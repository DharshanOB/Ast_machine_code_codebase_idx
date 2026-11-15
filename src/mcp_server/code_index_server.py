"""Phase 4: Astraea FastMCP server & multi-layer indexing pipeline."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from indexer.ast_chunker import ASTChunker, detect_language
from indexer.binary_lifter import BinaryLifter
from indexer.dwarf_aligner import DWARFAligner
from indexer.llvm_ir_pass import BUILD_TMP_ROOT, LLVMIRPass
from indexer.models import (
    ASTChunkNode,
    BinaryBasicBlock,
    DWARFLineMapping,
    IRStatus,
    Language,
    LLVMIRGraph,
    UnifiedCodeUnit,
)
from indexer.python_bytecode import lower_python_file
from indexer.semantic_cpg import SemanticCPGBuilder
from storage.hybrid_store import HybridCodeStore

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_BUDGET = 2048
CHARS_PER_TOKEN_ESTIMATE = 4
_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        "node_modules",
        "build",
        "dist",
        "target",
        "venv",
        ".venv",
        "__pycache__",
        "vendor",
    }
)

mcp = MCPServer("astraea-code-indexer")

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


@functools.lru_cache(maxsize=16)
def _get_compiler_version(bin_path: str | None) -> str:
    """Return the first line of `<bin_path> --version`, or empty string if unavailable."""
    if not bin_path:
        return ""
    try:
        proc = subprocess.run(
            [bin_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    output = (proc.stdout or proc.stderr or "").strip()
    return output.splitlines()[0].strip() if output else ""


def _transitive_include_hashes(
    src_path: Path, root: Path, seen: set[Path] | None = None
) -> list[str]:
    """Recursively collect SHA-256 digests of `#include "..."` headers inside root."""
    if seen is None:
        seen = set()
    src_resolved = src_path.resolve()
    if src_resolved in seen:
        return []
    seen.add(src_resolved)

    root_resolved = root.resolve()
    try:
        text = src_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    digests: list[str] = []
    for match in _INCLUDE_RE.finditer(text):
        inc_path = (src_path.parent / match.group(1)).resolve()
        if not inc_path.is_file() or not inc_path.is_relative_to(root_resolved):
            continue
        if inc_path in seen:
            continue
        try:
            digests.append(hashlib.sha256(inc_path.read_bytes()).hexdigest())
        except OSError:
            continue
        digests.extend(_transitive_include_hashes(inc_path, root_resolved, seen))
    return digests


def _sanitize_compiler_error(exc: Exception) -> str:
    """Return a concise failure category without echoing raw compiler stderr or included lines."""
    msg = str(exc)
    if "TimeoutExpired" in type(exc).__name__ or "timed out" in msg.lower():
        return "Compiler subprocess timed out"
    for prefix in (
        "Clang IR emission failed",
        "Rustc IR emission failed",
        "Object compilation failed",
        "Rustc object compilation failed",
    ):
        if msg.startswith(prefix):
            return prefix
    return f"Lowering failed ({type(exc).__name__})"


def _truncate_to_token_budget(
    payload: dict[str, Any], token_budget: int = DEFAULT_TOKEN_BUDGET
) -> dict[str, Any]:
    """Enforce SEC-03 token budget on MCP responses by trimming verbose IR/asm fields."""
    max_chars = token_budget * CHARS_PER_TOKEN_ESTIMATE
    serialized = json.dumps(payload)
    if len(serialized) <= max_chars:
        return payload

    trimmed = dict(payload)
    if "llvm_ir" in trimmed and isinstance(trimmed["llvm_ir"], dict):
        ir_copy = dict(trimmed["llvm_ir"])
        ir_copy["raw_ir"] = str(ir_copy.get("raw_ir", ""))[:512] + "...[truncated]"
        trimmed["llvm_ir"] = ir_copy
    trimmed["truncated_for_budget"] = True
    return trimmed


class CodeIndexPipeline:
    """Orchestrates all 4 indexing layers and serves queries via HybridCodeStore."""

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self.chunker = ASTChunker()
        self.cpg = SemanticCPGBuilder()
        self.ir_pass = LLVMIRPass()
        self.lifter = BinaryLifter()
        self.dwarf = DWARFAligner()
        self.store = HybridCodeStore(storage_dir=storage_dir)
        self.symbol_to_unit: dict[str, str] = {}
        self._file_hashes: dict[str, str] = {}
        self._file_chunks: dict[str, list[ASTChunkNode]] = {}
        self._file_unit_ids: dict[str, list[str]] = {}
        self._restore_persistent_state()

    def _restore_persistent_state(self) -> None:
        """Restore incremental file hashes, chunks, and symbol mappings from SQLite."""
        for path_str, (file_hash, unit_ids) in self.store.load_file_state().items():
            self._file_hashes[path_str] = file_hash
            self._file_unit_ids[path_str] = list(unit_ids)
            chunks: list[ASTChunkNode] = []
            for uid in unit_ids:
                unit = self.store.units.get(uid)
                if unit is not None:
                    chunks.append(unit.ast_chunk)
            if chunks:
                self._file_chunks[path_str] = chunks
        for unit in self.store.units.values():
            if unit.scip_symbol is not None:
                self.symbol_to_unit[unit.scip_symbol.symbol_uri] = unit.unit_id
            if unit.ast_chunk.symbol_name:
                self.symbol_to_unit[unit.ast_chunk.symbol_name] = unit.unit_id
        if self._file_chunks:
            all_chunks = [c for cl in self._file_chunks.values() for c in cl]
            self.cpg.build_from_ast_chunks(all_chunks)

    def index_workspace(
        self,
        repo_path: str | Path,
        compile_commands_path: str | Path | None = None,
        incremental: bool = True,
    ) -> dict[str, Any]:
        """Index repo_path across AST, CPG, IR, and Binary layers (incremental by default)."""
        root = Path(repo_path).resolve()
        if not root.exists() or not root.is_dir():
            return {"indexed_units": 0, "error": f"Directory does not exist: {repo_path}"}

        compile_flags_by_file = self._load_compile_commands(compile_commands_path)
        current_files: dict[str, tuple[Path, str]] = {}

        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            rel_parts = file_path.relative_to(root).parts
            if any(p.startswith(".") or p in _IGNORED_DIRS for p in rel_parts[:-1]):
                continue
            lang = detect_language(file_path)
            if lang is None:
                continue
            hasher = hashlib.sha256(file_path.read_bytes())
            flags = compile_flags_by_file.get(str(file_path.resolve()), [])
            compiler_bin = ""
            if lang in (Language.C, Language.CPP):
                compiler_bin = self.ir_pass.clang_bin or ""
            elif lang == Language.RUST:
                compiler_bin = self.ir_pass.rustc_bin or ""
            compiler_ver = _get_compiler_version(compiler_bin or None)
            include_hashes = (
                _transitive_include_hashes(file_path, root)
                if lang in (Language.C, Language.CPP)
                else []
            )
            hasher.update(
                b"\0"
                + json.dumps(flags).encode("utf-8")
                + b"\0"
                + compiler_bin.encode("utf-8")
                + b"\0"
                + compiler_ver.encode("utf-8")
                + b"\0"
                + json.dumps(include_hashes).encode("utf-8")
            )
            digest = hasher.hexdigest()
            current_files[str(file_path)] = (file_path, digest)

        deleted_paths = [p for p in self._file_hashes if p not in current_files]
        stale_unit_ids: list[str] = []
        for p in deleted_paths:
            self._file_hashes.pop(p, None)
            self._file_chunks.pop(p, None)
            stale_unit_ids.extend(self._file_unit_ids.pop(p, []))

        changed_files: dict[Path, list[ASTChunkNode]] = {}
        updated_paths: list[str] = []
        skipped_unchanged = 0
        for p_str, (file_path, digest) in current_files.items():
            if incremental and self._file_hashes.get(p_str) == digest:
                skipped_unchanged += 1
                continue
            stale_unit_ids.extend(self._file_unit_ids.pop(p_str, []))
            chunks = self.chunker.chunk_file(file_path)
            self._file_hashes[p_str] = digest
            updated_paths.append(p_str)
            if chunks:
                self._file_chunks[p_str] = chunks
                changed_files[file_path] = chunks
            else:
                self._file_chunks.pop(p_str, None)

        if stale_unit_ids:
            stale_set = set(stale_unit_ids)
            self.symbol_to_unit = {
                k: v for k, v in self.symbol_to_unit.items() if v not in stale_set
            }
        all_chunks = [c for chunk_list in self._file_chunks.values() for c in chunk_list]
        self.cpg = SemanticCPGBuilder()
        symbols = self.cpg.build_from_ast_chunks(all_chunks)
        chunk_to_symbol = {sym.chunk_id: sym for sym in symbols.values() if sym.chunk_id}

        units: list[UnifiedCodeUnit] = []
        newly_lowered_ir: list[LLVMIRGraph] = []
        degraded_count = 0

        for src_path, chunks in changed_files.items():
            ir_by_fn: dict[str, LLVMIRGraph] = {}
            bin_by_fn: dict[str, list[BinaryBasicBlock]] = {}
            dwarf_maps: list[DWARFLineMapping] = []
            lang = detect_language(src_path)
            file_error = ""

            if lang in (Language.C, Language.CPP):
                if self.ir_pass.clang_bin is None:
                    file_error = "clang binary not found on PATH"
                else:
                    flags = compile_flags_by_file.get(str(src_path.resolve()), [])
                    try:
                        ir_text = self.ir_pass.compile_to_ir(src_path, raw_flags=flags)
                        for g in self.ir_pass.parse_ir_module(ir_text, source_file=str(src_path)):
                            ir_by_fn[g.function_name] = g
                        with tempfile.TemporaryDirectory(dir=BUILD_TMP_ROOT) as tmp_obj_dir:
                            obj_path = self.lifter.compile_to_object(
                                src_path,
                                out_obj_path=Path(tmp_obj_dir) / f"{src_path.stem}.o",
                                raw_flags=flags,
                            )
                            bin_by_fn = self.lifter.lift_elf(obj_path)
                            dwarf_maps = self.dwarf.extract_mappings(
                                obj_path, default_file=str(src_path)
                            )
                        self.dwarf.attach_ir_dbg_ids(dwarf_maps, list(ir_by_fn.values()))
                    except Exception as exc:
                        file_error = _sanitize_compiler_error(exc)
                        logger.warning(
                            "Native C/C++ lowering failed for %s: %s", src_path, file_error
                        )
            elif lang == Language.RUST:
                if self.ir_pass.rustc_bin is None:
                    file_error = "rustc binary not found on PATH"
                else:
                    try:
                        ir_text = self.ir_pass.compile_rust_to_ir(src_path)
                        for g in self.ir_pass.parse_ir_module(ir_text, source_file=str(src_path)):
                            ir_by_fn[g.function_name] = g
                        with tempfile.TemporaryDirectory(dir=BUILD_TMP_ROOT) as tmp_obj_dir:
                            obj_path = self.lifter.compile_rust_to_object(
                                src_path,
                                out_obj_path=Path(tmp_obj_dir) / f"{src_path.stem}.o",
                            )
                            bin_by_fn = self.lifter.lift_elf(obj_path)
                            dwarf_maps = self.dwarf.extract_mappings(
                                obj_path, default_file=str(src_path)
                            )
                        self.dwarf.attach_ir_dbg_ids(dwarf_maps, list(ir_by_fn.values()))
                    except Exception as exc:
                        file_error = _sanitize_compiler_error(exc)
                        logger.warning("Rust lowering failed for %s: %s", src_path, file_error)
            elif lang == Language.PYTHON:
                try:
                    ir_by_fn, bin_by_fn = lower_python_file(src_path)
                except Exception as exc:
                    file_error = str(exc)
                    logger.warning("Python bytecode lowering failed for %s: %s", src_path, exc)

            newly_lowered_ir.extend(ir_by_fn.values())

            for chunk in chunks:
                sym = chunk_to_symbol.get(chunk.chunk_id)
                ir_graph = ir_by_fn.get(chunk.symbol_name)
                bin_blocks = bin_by_fn.get(chunk.symbol_name, [])
                reason = ""
                if ir_graph is not None or bin_blocks:
                    status = IRStatus.OK
                elif lang == Language.JAVASCRIPT:
                    status = IRStatus.NOT_APPLICABLE
                else:
                    status = IRStatus.DEGRADED
                    reason = file_error or f"No IR/binary symbol matched '{chunk.symbol_name}'"
                if status == IRStatus.DEGRADED:
                    degraded_count += 1
                unit_mappings = self.dwarf.lookup_va_for_line(
                    dwarf_maps, chunk.line_start, chunk.line_end
                )
                unit = UnifiedCodeUnit(
                    ast_chunk=chunk,
                    scip_symbol=sym,
                    ir_graph=ir_graph,
                    binary_blocks=bin_blocks,
                    dwarf_mappings=unit_mappings,
                    ir_status=status,
                    degraded_reason=reason,
                )
                units.append(unit)
                self._file_unit_ids.setdefault(str(src_path), []).append(unit.unit_id)
                if sym is not None:
                    self.symbol_to_unit[sym.symbol_uri] = unit.unit_id
                if chunk.symbol_name:
                    self.symbol_to_unit[chunk.symbol_name] = unit.unit_id

        updated_file_states = {
            p_str: (self._file_hashes[p_str], self._file_unit_ids.get(p_str, []))
            for p_str in updated_paths
        }
        count, deleted_count = self.store.apply_batch(
            upsert_units=units,
            delete_unit_ids=stale_unit_ids,
            save_file_states=updated_file_states,
            delete_file_paths=deleted_paths,
        )

        seen_ir_ids: set[int] = set()
        all_ir_graphs: list[LLVMIRGraph] = []
        for u in self.store.units.values():
            if u.ir_graph is not None and id(u.ir_graph) not in seen_ir_ids:
                seen_ir_ids.add(id(u.ir_graph))
                all_ir_graphs.append(u.ir_graph)
        for g in newly_lowered_ir:
            if id(g) not in seen_ir_ids:
                seen_ir_ids.add(id(g))
                all_ir_graphs.append(g)
        ir_cpg_validation = self.cpg.validate_against_ir_calls(all_ir_graphs)

        return {
            "indexed_units": count,
            "skipped_unchanged_files": skipped_unchanged,
            "deleted_units": deleted_count,
            "degraded_units": degraded_count,
            "symbols_count": len(symbols),
            "ir_cpg_validation": {
                "confirmed": len(ir_cpg_validation["confirmed"]),
                "cpg_only": len(ir_cpg_validation["cpg_only"]),
                "ir_only": len(ir_cpg_validation["ir_only"]),
                "ambiguous": len(ir_cpg_validation["ambiguous"]),
            },
        }

    def inspect_lowering_pipeline(
        self,
        symbol_or_function: str,
        bb_cursor: int = 0,
        bb_page_size: int = 5,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> dict[str, Any]:
        """Return AST, SCIP symbol, LLVM IR, and paginated binary CFG for a function."""
        unit_id = self.symbol_to_unit.get(symbol_or_function, symbol_or_function)
        unit = self.store.units.get(unit_id)
        if unit is None:
            for u in self.store.units.values():
                if u.ast_chunk.symbol_name == symbol_or_function:
                    unit = u
                    break
        if unit is None:
            return {"error": f"Symbol or unit not found: {symbol_or_function}"}

        blocks = unit.binary_blocks
        page = blocks[bb_cursor : bb_cursor + bb_page_size]
        next_cursor = bb_cursor + bb_page_size if (bb_cursor + bb_page_size) < len(blocks) else None

        payload: dict[str, Any] = {
            "unit_id": unit.unit_id,
            "ir_status": unit.ir_status.value,
            "degraded_reason": unit.degraded_reason,
            "ast": {
                "symbol_name": unit.ast_chunk.symbol_name,
                "file_path": unit.ast_chunk.file_path,
                "line_span": [unit.ast_chunk.line_start, unit.ast_chunk.line_end],
                "cyclomatic_complexity": unit.ast_chunk.cyclomatic_complexity,
            },
            "scip_symbol": unit.scip_symbol.symbol_uri if unit.scip_symbol else None,
            "llvm_ir": (
                {
                    "ir_id": unit.ir_graph.ir_id,
                    "canonical_tokens": unit.ir_graph.canonical_tokens[:64],
                    "node_count": len(unit.ir_graph.nodes),
                    "edge_count": len(unit.ir_graph.edges),
                }
                if unit.ir_graph
                else None
            ),
            "binary_cfg": {
                "total_blocks": len(blocks),
                "bb_cursor": bb_cursor,
                "next_bb_cursor": next_cursor,
                "blocks": [b.model_dump() for b in page],
            },
            "dwarf_mappings_count": len(unit.dwarf_mappings),
        }
        return _truncate_to_token_budget(payload, token_budget=token_budget)

    @staticmethod
    def _load_compile_commands(cc_path: str | Path | None) -> dict[str, list[str]]:
        if cc_path is None:
            return {}
        path = Path(cc_path)
        if not path.exists():
            return {}
        entries = json.loads(path.read_text(encoding="utf-8"))
        out: dict[str, list[str]] = {}
        for entry in entries:
            file_str = str(Path(entry.get("file", "")).resolve())
            args = entry.get("arguments")
            if isinstance(args, list):
                out[file_str] = [str(a) for a in args]
            elif isinstance(entry.get("command"), str):
                out[file_str] = entry["command"].split()
        return out


_PIPELINE = CodeIndexPipeline()


@mcp.tool()
def index_workspace(
    repo_path: str, compile_commands_path: str | None = None, incremental: bool = True
) -> dict[str, Any]:
    """Index a repository across AST, SCIP/CPG, LLVM IR, and disassembled Machine Code."""
    return _PIPELINE.index_workspace(
        repo_path, compile_commands_path, incremental=incremental
    )


@mcp.tool()
def search_code_hybrid(
    query: str, layers: list[str] | None = None, top_k: int = 10
) -> list[dict[str, Any]]:
    """Search indexed code units using weighted Reciprocal Rank Fusion across modalities."""
    hits = _PIPELINE.store.rrf_search(query=query, layers=layers, top_k=top_k)
    return [h.model_dump() for h in hits]


@mcp.tool()
def inspect_lowering_pipeline(
    symbol_or_function: str, bb_cursor: int = 0
) -> dict[str, Any]:
    """Inspect AST, SCIP symbol, LLVM IR, and paginated machine-code CFG for a function."""
    return _PIPELINE.inspect_lowering_pipeline(symbol_or_function, bb_cursor=bb_cursor)


@mcp.tool()
def find_ir_or_binary_clones(
    function_id: str, threshold: float = 0.75, modality: str = "ir_structural"
) -> list[dict[str, Any]]:
    """Find semantically equivalent functions across optimization levels or syntax variants."""
    uid = _PIPELINE.symbol_to_unit.get(function_id, function_id)
    clones = _PIPELINE.store.find_clones(uid, modality=modality, threshold=threshold)
    return [{"unit_id": cid, "similarity": score} for cid, score in clones]


@mcp.tool()
def trace_symbol_dataflow(symbol_id: str, max_hops: int = 2) -> list[dict[str, Any]]:
    """Trace cycle-safe k-hop caller/callee and data-flow edges for a SCIP symbol URI."""
    return _PIPELINE.cpg.k_hop_slice(symbol_id, max_hops=max_hops)


@mcp.tool()
def get_index_health() -> dict[str, Any]:
    """Return storage consistency, SQLite health, and CPG symbol/edge counts."""
    return {
        **_PIPELINE.store.health_check(),
        "cpg_symbols": len(_PIPELINE.cpg.symbols),
        "cpg_edges": _PIPELINE.cpg.graph.number_of_edges(),
    }


def main() -> None:
    """Run the Astraea code-multi-indexer MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
