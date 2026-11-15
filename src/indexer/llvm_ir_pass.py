"""Layer 3: Hardened Clang/LLVM SSA IR extractor and ProGraML graph builder."""

from __future__ import annotations

import os
import re
import resource
import shutil
import subprocess
import tempfile
from pathlib import Path

from llvmlite import binding as llvm

from indexer.models import LLVMIRGraph, ProGraMLEdge, ProGraMLNode

ALLOWED_CLANG_PREFIXES: tuple[str, ...] = ("-I", "-isystem", "-D", "-U", "-std=", "-target ")
MAX_GRAPH_NODES: int = 10_000
MAX_COMPILER_TIMEOUT_S: float = 15.0
MAX_COMPILER_MEMORY_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB
BUILD_TMP_ROOT = Path("/tmp/ag-index-build")

_SSA_REG_RE = re.compile(r"%([A-Za-z0-9_.$-]+)")
_DBG_REF_RE = re.compile(r"!dbg\s+!(\d+)")
_DBG_LOC_RE = re.compile(r"^!(\d+)\s*=\s*!DILocation\(line:\s*(\d+)")
_OPCODE_RE = re.compile(r"^(?:%[^\s=]+\s*=\s*)?([a-zA-Z0-9_.]+)")


def sanitize_compile_flags(raw_flags: list[str]) -> list[str]:
    """Allowlist-filter compiler flags to prevent SEC-01 flag/plugin injection."""
    safe_flags: list[str] = []
    skip_next = False
    for i, flag in enumerate(raw_flags):
        if skip_next:
            skip_next = False
            continue
        if flag.startswith("@"):
            continue
        if flag in ("-I", "-isystem", "-D", "-U") and i + 1 < len(raw_flags):
            arg = raw_flags[i + 1]
            if not arg.startswith("@") and "-fplugin" not in arg:
                safe_flags.extend([flag, arg])
            skip_next = True
        elif flag.startswith(ALLOWED_CLANG_PREFIXES) and "-fplugin" not in flag:
            safe_flags.append(flag)
    return safe_flags


def _apply_resource_limits() -> None:
    """Enforce SEC-02 virtual memory cap on compiler subprocesses."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = MAX_COMPILER_MEMORY_BYTES
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except (OSError, ValueError):
        pass


def resolve_clang_binary(preferred: str = "clang-18") -> str | None:
    """Locate clang-18 or fallback clang binary on PATH."""
    for candidate in (preferred, "clang"):
        found = shutil.which(candidate)
        if found is not None:
            return found
    return None


def canonicalize_ssa_instruction(inst_text: str) -> tuple[str, list[str]]:
    """Replace local SSA register names with deterministic <SSA_REG_i> tokens."""
    reg_map: dict[str, str] = {}
    tokens: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in reg_map:
            reg_map[name] = f"<SSA_REG_{len(reg_map)}>"
        return reg_map[name]

    normalized = _SSA_REG_RE.sub(_replace, inst_text.strip())
    opcode_match = _OPCODE_RE.match(inst_text.strip())
    if opcode_match:
        tokens.append(opcode_match.group(1))
    tokens.extend(reg_map.values())
    return normalized, tokens


class LLVMIRPass:
    """Compiles C/C++ translation units to LLVM SSA IR and builds ProGraML graphs."""

    def __init__(self, clang_bin: str = "clang-18") -> None:
        self.clang_bin = resolve_clang_binary(clang_bin)
        self.rustc_bin = shutil.which("rustc")
        BUILD_TMP_ROOT.mkdir(parents=True, exist_ok=True)

    def compile_to_ir(
        self,
        source_path: str | Path,
        raw_flags: list[str] | None = None,
        opt_level: str = "-O1",
        target_triple: str | None = None,
    ) -> str:
        """Compile a source file to LLVM IR (.ll) text inside an isolated temp directory."""
        if self.clang_bin is None:
            raise RuntimeError("No clang compiler binary found on PATH")

        src = Path(source_path)
        safe_flags = sanitize_compile_flags(raw_flags or [])
        with tempfile.TemporaryDirectory(dir=BUILD_TMP_ROOT) as tmp_dir:
            out_ll = Path(tmp_dir) / f"{src.stem}.ll"
            cmd: list[str] = [
                self.clang_bin,
                "-S",
                "-emit-llvm",
                opt_level,
                "-gline-tables-only",
                "-fno-discard-value-names",
                "-ferror-limit=20",
                "-ftemplate-depth=64",
                *safe_flags,
            ]
            if target_triple:
                cmd.extend(["-target", target_triple])
            cmd.extend([str(src), "-o", str(out_ll)])

            proc = subprocess.run(
                cmd,
                shell=False,
                capture_output=True,
                text=True,
                timeout=MAX_COMPILER_TIMEOUT_S,
                preexec_fn=_apply_resource_limits if os.name == "posix" else None,
            )
            if proc.returncode != 0 or not out_ll.exists():
                raise RuntimeError(f"Clang IR emission failed: {proc.stderr.strip()}")
            return out_ll.read_text(encoding="utf-8", errors="replace")

    def compile_rust_to_ir(self, source_path: str | Path) -> str:
        """Compile a standalone Rust file to LLVM IR text via rustc."""
        if self.rustc_bin is None:
            raise RuntimeError("No rustc compiler binary found on PATH")
        src = Path(source_path)
        with tempfile.TemporaryDirectory(dir=BUILD_TMP_ROOT) as tmp_dir:
            out_ll = Path(tmp_dir) / f"{src.stem}.ll"
            cmd = [
                self.rustc_bin,
                "--crate-type",
                "lib",
                "--emit=llvm-ir",
                "-C",
                "debuginfo=1",
                "-C",
                "opt-level=1",
                str(src),
                "-o",
                str(out_ll),
            ]
            proc = subprocess.run(
                cmd,
                shell=False,
                capture_output=True,
                text=True,
                timeout=MAX_COMPILER_TIMEOUT_S,
                preexec_fn=_apply_resource_limits if os.name == "posix" else None,
            )
            if proc.returncode != 0 or not out_ll.exists():
                raise RuntimeError(f"Rustc IR emission failed: {proc.stderr.strip()}")
            return out_ll.read_text(encoding="utf-8", errors="replace")

    def parse_ir_module(self, ir_text: str, source_file: str = "<memory>") -> list[LLVMIRGraph]:
        """Parse LLVM IR assembly text into per-function ProGraML LLVMIRGraph models."""
        mod = llvm.parse_assembly(ir_text)
        dbg_lines = self._extract_dbg_locations(str(mod))
        graphs: list[LLVMIRGraph] = []

        for func in mod.functions:
            if func.is_declaration:
                continue
            graphs.append(self._build_function_graph(func, source_file, dbg_lines))
        return graphs

    def _build_function_graph(
        self,
        func: llvm.ValueRef,
        source_file: str,
        dbg_lines: dict[str, int],
    ) -> LLVMIRGraph:
        nodes: list[ProGraMLNode] = []
        edges: list[ProGraMLEdge] = []
        canonical_tokens: list[str] = []
        fn_dbg_map: dict[str, int] = {}
        reg_def_node: dict[str, int] = {}
        block_entry_node: dict[str, int] = {}
        block_exit_node: dict[str, int] = {}
        truncated = False

        # Pre-register basic block entry/exit nodes for control-flow linking.
        blocks = list(func.blocks)
        for bb in blocks:
            bb_name = bb.name or f"bb_{len(block_entry_node)}"
            prev_inst_id: int | None = None
            for inst in bb.instructions:
                if len(nodes) >= MAX_GRAPH_NODES:
                    truncated = True
                    break
                raw_inst = str(inst).strip()
                norm_inst, inst_tokens = canonicalize_ssa_instruction(raw_inst)
                opcode = inst.opcode if hasattr(inst, "opcode") else ""
                if opcode:
                    canonical_tokens.append(opcode)
                canonical_tokens.extend(t for t in inst_tokens if t != opcode)

                dbg_match = _DBG_REF_RE.search(raw_inst)
                dbg_id = f"!{dbg_match.group(1)}" if dbg_match else None
                if dbg_id and dbg_id in dbg_lines:
                    fn_dbg_map[dbg_id] = dbg_lines[dbg_id]

                nid = len(nodes)
                nodes.append(
                    ProGraMLNode(
                        node_id=nid,
                        node_type="inst",
                        text=norm_inst,
                        opcode=opcode,
                        dbg_id=dbg_id,
                    )
                )
                if bb_name not in block_entry_node:
                    block_entry_node[bb_name] = nid
                block_exit_node[bb_name] = nid

                if prev_inst_id is not None:
                    edges.append(
                        ProGraMLEdge(source=prev_inst_id, target=nid, flow_type="ctrl", position=0)
                    )
                prev_inst_id = nid

                # Track SSA definition if instruction assigns %name = ...
                if "=" in raw_inst:
                    lhs = raw_inst.split("=", 1)[0].strip()
                    if lhs.startswith("%"):
                        reg_def_node[lhs] = nid

                # Link SSA data-dependency edges from operands to this instruction.
                rhs = raw_inst.split("=", 1)[1] if "=" in raw_inst else raw_inst
                for pos, op_match in enumerate(_SSA_REG_RE.finditer(rhs)):
                    op_reg = f"%{op_match.group(1)}"
                    if op_reg in reg_def_node and reg_def_node[op_reg] != nid:
                        edges.append(
                            ProGraMLEdge(
                                source=reg_def_node[op_reg],
                                target=nid,
                                flow_type="data",
                                position=pos,
                            )
                        )
                    elif op_reg not in reg_def_node and len(nodes) < MAX_GRAPH_NODES:
                        val_id = len(nodes)
                        nodes.append(
                            ProGraMLNode(
                                node_id=val_id,
                                node_type="val",
                                text="<SSA_PARAM>",
                                opcode="",
                            )
                        )
                        reg_def_node[op_reg] = val_id
                        edges.append(
                            ProGraMLEdge(source=val_id, target=nid, flow_type="data", position=pos)
                        )

                if opcode == "call":
                    edges.append(ProGraMLEdge(source=nid, target=nid, flow_type="call", position=0))

            if truncated:
                break

        # Link inter-block control-flow edges (including phi/loop back-edges).
        for bb in blocks:
            bb_name = bb.name
            exit_id = block_exit_node.get(bb_name)
            if exit_id is None:
                continue
            bb_text = str(bb)
            for pos, label_match in enumerate(re.finditer(r"label\s+%([A-Za-z0-9_.$-]+)", bb_text)):
                target_bb = label_match.group(1)
                entry_id = block_entry_node.get(target_bb)
                if entry_id is not None:
                    edges.append(
                        ProGraMLEdge(
                            source=exit_id, target=entry_id, flow_type="ctrl", position=pos
                        )
                    )

        return LLVMIRGraph(
            function_name=func.name,
            source_file=source_file,
            nodes=nodes,
            edges=edges,
            canonical_tokens=canonical_tokens,
            dbg_line_map=fn_dbg_map,
            truncated=truncated,
            raw_ir=str(func),
        )

    @staticmethod
    def _extract_dbg_locations(ir_text: str) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for line in ir_text.splitlines():
            m = _DBG_LOC_RE.match(line.strip())
            if m:
                mapping[f"!{m.group(1)}"] = int(m.group(2))
        return mapping
