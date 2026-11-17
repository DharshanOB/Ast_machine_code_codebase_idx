"""Layer 3: Hardened Clang/LLVM SSA IR extractor and ProGraML graph builder."""

from __future__ import annotations

import atexit
import functools
import os
import re
import resource
import shutil
import subprocess
import tempfile
from pathlib import Path

from llvmlite import binding as llvm

from indexer.models import LLVMIRGraph, ProGraMLEdge, ProGraMLNode

ALLOWED_CLANG_PREFIXES: tuple[str, ...] = ("-I", "-isystem", "-D", "-U", "-std=", "--target=")
MAX_GRAPH_NODES: int = 10_000
MAX_COMPILER_TIMEOUT_S: float = 15.0
MAX_COMPILER_MEMORY_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB


@functools.lru_cache(maxsize=1)
def get_build_tmp_root() -> Path:
    """Lazily create and register cleanup for the compiler temporary directory root."""
    root = Path(tempfile.mkdtemp(prefix="astraea-build-"))
    atexit.register(shutil.rmtree, root, True)
    return root

_SSA_REG_RE = re.compile(r"%([A-Za-z0-9_.$-]+)")
_DBG_REF_RE = re.compile(r"!dbg\s+!(\d+)")
_DBG_LOC_RE = re.compile(r"^!(\d+)\s*=\s*!DILocation\(line:\s*(\d+)")
_DISUBPROGRAM_RE = re.compile(
    r"^!(\d+)\s*=\s*(?:distinct\s+)?!DISubprogram\((.+)\)\s*$"
)
_DISUB_NAME_RE = re.compile(r'\bname:\s*"([^"]+)"')
_DISUB_LINKAGE_RE = re.compile(r'\blinkageName:\s*"([^"]+)"')
_DISUB_LINE_RE = re.compile(r"\bline:\s*(\d+)")
_DEFINE_DBG_RE = re.compile(
    r'^define\b[^@]*@(?:"([^"]+)"|([A-Za-z0-9_.$-]+)\b)(?:.*?\s(!dbg\s+!(\d+)))?'
)
_RUST_HASH_SUFFIX_RE = re.compile(r"::h[0-9a-fA-F]{16}$")
_LHS_ASSIGN_RE = re.compile(r"^\s*(%[A-Za-z0-9_.$-]+)\s*=\s*(.*)$", re.DOTALL)
_LABEL_OPERAND_RE = re.compile(r"\blabel\s+%([A-Za-z0-9_.$-]+)")
_PHI_BLOCK_RE = re.compile(r",\s*%([A-Za-z0-9_.$-]+)\s*\]")
_OPCODE_RE = re.compile(r"^(?:%[^\s=]+\s*=\s*)?([a-zA-Z0-9_.]+)")


@functools.lru_cache(maxsize=2048)
def extract_leaf_symbol_name(raw_name: str) -> str:
    """Best-effort demangle/strip C++ Itanium or Rust mangled symbol to leaf identifier."""
    clean = raw_name.strip().lstrip("@").strip('"')
    if not clean:
        return ""
    # Fast path for unmangled symbols
    if not clean.startswith(("_Z", "__Z", "_R")):
        clean = _RUST_HASH_SUFFIX_RE.sub("", clean)
        return clean.split("::")[-1]
    cppfilt = shutil.which("c++filt")
    if cppfilt is not None and clean.startswith(("_Z", "__Z")):
        try:
            proc = subprocess.run(
                [cppfilt, clean],
                shell=False,
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            demangled = proc.stdout.strip()
            if proc.returncode == 0 and demangled and demangled != clean:
                base = demangled.split("(", 1)[0].strip()
                base = _RUST_HASH_SUFFIX_RE.sub("", base)
                return base.split("::")[-1].split()[-1]
        except (OSError, subprocess.SubprocessError):
            pass
    # Rust v0 mangling: _RNv...<len>_?<ident>...
    if clean.startswith("_R"):
        pos = 2
        idents: list[str] = []
        num_re = re.compile(r"(\d+)")
        while pos < len(clean):
            m = num_re.match(clean, pos)
            if not m:
                pos += 1
                continue
            length = int(m.group(1))
            end_num = m.end()
            if end_num < len(clean) and clean[end_num] == "_":
                end_num += 1
            if length > 0 and end_num + length <= len(clean):
                cand = clean[end_num : end_num + length]
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cand):
                    idents.append(cand)
                    pos = end_num + length
                    continue
            pos = end_num + 1
        if idents:
            return idents[-1]
    return clean


_FORBIDDEN_FLAG_SUBSTRINGS: tuple[str, ...] = ("@", "-fplugin", "-xclang", "-mllvm", "-specs=")


def _has_forbidden_flag_token(token: str) -> bool:
    low = token.lower()
    return any(bad in low for bad in _FORBIDDEN_FLAG_SUBSTRINGS)


def sanitize_compile_flags(raw_flags: list[str]) -> list[str]:
    """Allowlist-filter compiler flags to prevent SEC-01 flag/plugin/response-file injection."""
    safe_flags: list[str] = []
    skip_next = False
    for i, flag in enumerate(raw_flags):
        if skip_next:
            skip_next = False
            continue
        if _has_forbidden_flag_token(flag):
            continue
        if flag in ("-I", "-isystem", "-D", "-U", "-target") and i + 1 < len(raw_flags):
            arg = raw_flags[i + 1]
            if not arg.startswith("-") and not _has_forbidden_flag_token(arg):
                safe_flags.extend([flag, arg])
            skip_next = True
        elif flag.startswith(ALLOWED_CLANG_PREFIXES):
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
        get_build_tmp_root().mkdir(parents=True, exist_ok=True)

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
        with tempfile.TemporaryDirectory(dir=get_build_tmp_root()) as tmp_dir:
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
            return self._run_ir_compiler(cmd, out_ll, "Clang")

    def compile_rust_to_ir(self, source_path: str | Path) -> str:
        """Compile a standalone Rust file to LLVM IR text via rustc."""
        if self.rustc_bin is None:
            raise RuntimeError("No rustc compiler binary found on PATH")
        src = Path(source_path)
        with tempfile.TemporaryDirectory(dir=get_build_tmp_root()) as tmp_dir:
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
                "-C",
                "link-dead-code",
                str(src),
                "-o",
                str(out_ll),
            ]
            return self._run_ir_compiler(cmd, out_ll, "Rustc")

    @staticmethod
    def _run_ir_compiler(cmd: list[str], out_ll: Path, label: str) -> str:
        """Execute an IR-emitting compiler subprocess with resource limits."""
        proc = subprocess.run(
            cmd,
            shell=False,
            capture_output=True,
            text=True,
            timeout=MAX_COMPILER_TIMEOUT_S,
            preexec_fn=_apply_resource_limits if os.name == "posix" else None,
        )
        if proc.returncode != 0 or not out_ll.exists():
            raise RuntimeError(f"{label} IR emission failed: {proc.stderr.strip()}")
        return out_ll.read_text(encoding="utf-8", errors="replace")

    def parse_ir_module(self, ir_text: str, source_file: str = "<memory>") -> list[LLVMIRGraph]:
        """Parse LLVM IR assembly text into per-function ProGraML LLVMIRGraph models."""
        ctx = llvm.create_context()
        try:
            mod = llvm.parse_assembly(ir_text, context=ctx)
            try:
                mod_text = str(mod)
                combined_text = f"{ir_text}\n{mod_text}"
                dbg_lines = self._extract_dbg_locations(combined_text)
                subprogram_meta = self._extract_subprogram_metadata(combined_text)
                graphs: list[LLVMIRGraph] = []

                for func in mod.functions:
                    if func.is_declaration:
                        continue
                    graphs.append(
                        self._build_function_graph(func, source_file, dbg_lines, subprogram_meta)
                    )
                return graphs
            finally:
                mod.close()
        finally:
            ctx.close()

    def _build_function_graph(
        self,
        func: llvm.ValueRef,
        source_file: str,
        dbg_lines: dict[str, int],
        subprogram_meta: dict[str, tuple[str, int | None]] | None = None,
    ) -> LLVMIRGraph:
        nodes: list[ProGraMLNode] = []
        edges: list[ProGraMLEdge] = []
        canonical_tokens: list[str] = []
        fn_dbg_map: dict[str, int] = {}
        reg_def_node: dict[str, int] = {}
        block_entry_node: dict[str, int] = {}
        block_exit_node: dict[str, int] = {}
        inst_rhs_list: list[tuple[int, str]] = []
        truncated = False

        # Pass 1: register instruction nodes, intra-block ctrl edges, and all SSA definitions
        # so loop-carried phi back-edges resolve to defining instructions rather than params.
        blocks = list(func.blocks)
        bb_names: list[str] = []
        for idx, bb in enumerate(blocks):
            bb_name = bb.name or f"bb_{idx}"
            bb_names.append(bb_name)
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

                assign_m = _LHS_ASSIGN_RE.match(raw_inst)
                if assign_m:
                    reg_def_node[assign_m.group(1)] = nid
                    rhs = assign_m.group(2)
                else:
                    rhs = raw_inst
                inst_rhs_list.append((nid, rhs))

                if opcode == "call":
                    edges.append(ProGraMLEdge(source=nid, target=nid, flow_type="call", position=0))

            if truncated:
                break

        non_value_regs: set[str] = {f"%{b}" for b in block_entry_node}
        for _, rhs in inst_rhs_list:
            for lm in _LABEL_OPERAND_RE.finditer(rhs):
                non_value_regs.add(f"%{lm.group(1)}")
            for pm in _PHI_BLOCK_RE.finditer(rhs):
                non_value_regs.add(f"%{pm.group(1)}")

        # Pass 2: link SSA data-dependency edges using the complete per-function def map.
        for nid, rhs in inst_rhs_list:
            for pos, op_match in enumerate(_SSA_REG_RE.finditer(rhs)):
                op_reg = f"%{op_match.group(1)}"
                if op_reg in non_value_regs or op_reg.startswith(
                    ("%struct.", "%class.", "%union.")
                ):
                    continue
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

        # Link inter-block control-flow edges (including phi/loop back-edges).
        for bb, bb_name in zip(blocks, bb_names, strict=True):
            exit_id = block_exit_node.get(bb_name)
            if exit_id is None:
                continue
            bb_text = str(bb)
            for pos, label_match in enumerate(_LABEL_OPERAND_RE.finditer(bb_text)):
                target_bb = label_match.group(1)
                entry_id = block_entry_node.get(target_bb) or block_entry_node.get(
                    f"bb_{target_bb}"
                )
                if entry_id is not None:
                    edges.append(
                        ProGraMLEdge(
                            source=exit_id, target=entry_id, flow_type="ctrl", position=pos
                        )
                    )

        src_name, decl_line = (subprogram_meta or {}).get(func.name, ("", None))
        if not src_name:
            src_name = extract_leaf_symbol_name(func.name)

        return LLVMIRGraph(
            function_name=func.name,
            source_name=src_name,
            decl_line=decl_line,
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

    @staticmethod
    def _extract_subprogram_metadata(ir_text: str) -> dict[str, tuple[str, int | None]]:
        """Map LLVM function symbol names to (unmangled_name, decl_line) via !DISubprogram."""
        subprograms: dict[str, tuple[str, str, int | None]] = {}
        by_linkage: dict[str, tuple[str, int | None]] = {}
        for line in ir_text.splitlines():
            m = _DISUBPROGRAM_RE.match(line.strip())
            if not m:
                continue
            dbg_id = f"!{m.group(1)}"
            body = m.group(2)
            name_m = _DISUB_NAME_RE.search(body)
            if not name_m:
                continue
            name = name_m.group(1)
            line_m = _DISUB_LINE_RE.search(body)
            decl_line = int(line_m.group(1)) if line_m else None
            link_m = _DISUB_LINKAGE_RE.search(body)
            linkage = link_m.group(1) if link_m else ""
            subprograms[dbg_id] = (name, linkage, decl_line)
            if linkage:
                by_linkage[linkage] = (name, decl_line)

        result: dict[str, tuple[str, int | None]] = dict(by_linkage)
        for line in ir_text.splitlines():
            dm = _DEFINE_DBG_RE.match(line.strip())
            if not dm:
                continue
            fn_name = dm.group(1) or dm.group(2) or ""
            dbg_num = dm.group(4)
            if fn_name and dbg_num and f"!{dbg_num}" in subprograms:
                sp_name, _, sp_line = subprograms[f"!{dbg_num}"]
                result[fn_name] = (sp_name, sp_line)
        return result
