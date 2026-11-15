"""Phase 4: Astraea FastMCP server & multi-layer indexing pipeline."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from indexer.ast_chunker import MAX_FILE_BYTES, ASTChunker, detect_language, is_header_file
from indexer.binary_lifter import BinaryLifter
from indexer.dwarf_aligner import DWARFAligner
from indexer.llvm_ir_pass import BUILD_TMP_ROOT, LLVMIRPass, extract_leaf_symbol_name
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
    src_path: Path,
    root: Path,
    include_dirs: tuple[Path, ...] = (),
    seen: set[Path] | None = None,
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
        if src_resolved.stat().st_size > MAX_FILE_BYTES:
            return []
        text = src_resolved.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    search_dirs = (src_resolved.parent, *include_dirs)
    digests: list[str] = []
    for match in _INCLUDE_RE.finditer(text):
        inc_rel = match.group(1)
        inc_path: Path | None = None
        for base_dir in search_dirs:
            cand = (base_dir / inc_rel).resolve()
            if cand.is_file() and cand.is_relative_to(root_resolved):
                inc_path = cand
                break
        if inc_path is None or inc_path in seen:
            continue
        try:
            if inc_path.stat().st_size <= MAX_FILE_BYTES:
                digests.append(hashlib.sha256(inc_path.read_bytes()).hexdigest())
        except OSError:
            continue
        digests.extend(
            _transitive_include_hashes(
                inc_path, root_resolved, include_dirs=include_dirs, seen=seen
            )
        )
    return digests


def _extract_include_dirs(flags: list[str], root: Path) -> tuple[Path, ...]:
    """Extract safe `-I` / `-isystem` directories inside root from compiler flags."""
    dirs: list[Path] = []
    i = 0
    while i < len(flags):
        flag = flags[i]
        raw_dir = ""
        if flag in ("-I", "-isystem") and i + 1 < len(flags):
            raw_dir = flags[i + 1]
            i += 2
        elif flag.startswith("-I") and len(flag) > 2:
            raw_dir = flag[2:]
            i += 1
        elif flag.startswith("-isystem") and len(flag) > 8:
            raw_dir = flag[8:]
            i += 1
        else:
            i += 1
            continue
        try:
            cand = Path(raw_dir)
            resolved = (cand if cand.is_absolute() else (root / cand)).resolve()
            if resolved.is_dir() and resolved.is_relative_to(root.resolve()):
                dirs.append(resolved)
        except OSError:
            continue
    return tuple(dirs)


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
    max_chars = max(256, token_budget * CHARS_PER_TOKEN_ESTIMATE)
    if len(json.dumps(payload)) <= max_chars:
        return payload

    trimmed = dict(payload)
    trimmed["truncated_for_budget"] = True

    if isinstance(trimmed.get("llvm_ir"), dict):
        ir_copy = dict(trimmed["llvm_ir"])
        if "raw_ir" in ir_copy:
            ir_copy["raw_ir"] = str(ir_copy["raw_ir"])[:256] + "...[truncated]"
        if (
            len(json.dumps(trimmed)) > max_chars
            and isinstance(ir_copy.get("canonical_tokens"), list)
        ):
            ir_copy["canonical_tokens"] = ir_copy["canonical_tokens"][:16]
        trimmed["llvm_ir"] = ir_copy

    if len(json.dumps(trimmed)) > max_chars and isinstance(trimmed.get("binary_cfg"), dict):
        cfg_copy = dict(trimmed["binary_cfg"])
        blocks = list(cfg_copy.get("blocks") or [])
        while blocks:
            candidate = {**trimmed, "binary_cfg": {**cfg_copy, "blocks": blocks}}
            if len(json.dumps(candidate)) <= max_chars:
                break
            blocks.pop()
        # If even 1 block exceeds budget, trim its instruction lists.
        if not blocks and cfg_copy.get("blocks"):
            first = dict(cfg_copy["blocks"][0])
            first["raw_instructions"] = list(first.get("raw_instructions") or [])[:16]
            first["normalized_instructions"] = list(first.get("normalized_instructions") or [])[:16]
            blocks = [first]
        cfg_copy["blocks"] = blocks
        trimmed["binary_cfg"] = cfg_copy

    if len(json.dumps(trimmed)) > max_chars and isinstance(trimmed.get("degraded_reason"), str):
        trimmed["degraded_reason"] = trimmed["degraded_reason"][:256]

    return trimmed


def _is_path_allowed(target: Path) -> bool:
    """Verify target is within ASTRAEA_ALLOWED_ROOTS if that env var is configured."""
    raw = os.environ.get("ASTRAEA_ALLOWED_ROOTS", "").strip()
    if not raw:
        return True
    resolved = target.resolve()
    for entry in raw.split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        try:
            if resolved.is_relative_to(Path(entry).resolve()):
                return True
        except OSError:
            continue
    return False


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
        self._lock = threading.RLock()
        self._restore_persistent_state()

    def _restore_persistent_state(self) -> None:
        """Restore incremental file hashes, chunks, CPG, and IR validation from SQLite."""
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
                self.symbol_to_unit.setdefault(unit.ast_chunk.symbol_name, unit.unit_id)
        if self._file_chunks:
            all_chunks = [c for cl in self._file_chunks.values() for c in cl]
            self.cpg.build_from_ast_chunks(all_chunks)
            ir_graphs = [u.ir_graph for u in self.store.units.values() if u.ir_graph is not None]
            if ir_graphs:
                self.cpg.validate_against_ir_calls(ir_graphs)

    def index_workspace(
        self,
        repo_path: str | Path,
        compile_commands_path: str | Path | None = None,
        incremental: bool = True,
    ) -> dict[str, Any]:
        """Index repo_path across AST, CPG, IR, and Binary layers (incremental by default)."""
        with self._lock:
            return self._index_workspace_locked(
                repo_path=repo_path,
                compile_commands_path=compile_commands_path,
                incremental=incremental,
            )

    def _index_workspace_locked(
        self,
        repo_path: str | Path,
        compile_commands_path: str | Path | None = None,
        incremental: bool = True,
    ) -> dict[str, Any]:
        root = Path(repo_path).resolve()
        if not _is_path_allowed(root):
            return {"indexed_units": 0, "error": f"Path outside allowed roots: {repo_path}"}
        if not root.exists() or not root.is_dir():
            return {"indexed_units": 0, "error": f"Directory does not exist: {repo_path}"}

        compile_flags_by_file = self._load_compile_commands(compile_commands_path)
        current_files: dict[str, tuple[Path, str]] = {}

        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            try:
                resolved_file = file_path.resolve()
                if not resolved_file.is_relative_to(root):
                    continue
                if resolved_file.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            rel_parts = file_path.relative_to(root).parts
            if any(p.startswith(".") or p in _IGNORED_DIRS for p in rel_parts[:-1]):
                continue
            lang = detect_language(file_path)
            if lang is None:
                continue
            hasher = hashlib.sha256(resolved_file.read_bytes())
            flags = compile_flags_by_file.get(str(resolved_file), [])
            compiler_bin = ""
            if lang in (Language.C, Language.CPP) and not is_header_file(file_path):
                compiler_bin = self.ir_pass.clang_bin or ""
            elif lang == Language.RUST:
                compiler_bin = self.ir_pass.rustc_bin or ""
            compiler_ver = _get_compiler_version(compiler_bin or None)
            include_dirs = (
                _extract_include_dirs(flags, root) if lang in (Language.C, Language.CPP) else ()
            )
            include_hashes = (
                _transitive_include_hashes(resolved_file, root, include_dirs=include_dirs)
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

        staged_hashes = dict(self._file_hashes)
        staged_chunks = dict(self._file_chunks)
        staged_unit_ids = {k: list(v) for k, v in self._file_unit_ids.items()}

        deleted_paths = [p for p in staged_hashes if p not in current_files]
        stale_unit_ids: list[str] = []
        for p in deleted_paths:
            staged_hashes.pop(p, None)
            staged_chunks.pop(p, None)
            stale_unit_ids.extend(staged_unit_ids.pop(p, []))

        changed_files: dict[Path, list[ASTChunkNode]] = {}
        updated_paths: list[str] = []
        skipped_unchanged = 0
        for p_str, (file_path, digest) in current_files.items():
            if incremental and staged_hashes.get(p_str) == digest:
                skipped_unchanged += 1
                continue
            stale_unit_ids.extend(staged_unit_ids.pop(p_str, []))
            chunks = self.chunker.chunk_file(file_path)
            staged_hashes[p_str] = digest
            updated_paths.append(p_str)
            if chunks:
                staged_chunks[p_str] = chunks
                changed_files[file_path] = chunks
            else:
                staged_chunks.pop(p_str, None)

        if not changed_files and not deleted_paths and not updated_paths:
            return {
                "indexed_units": 0,
                "skipped_unchanged_files": skipped_unchanged,
                "deleted_units": 0,
                "degraded_units": 0,
                "symbols_count": len(self.cpg.symbols),
                "ir_cpg_validation": {
                    "confirmed": 0,
                    "cpg_only": 0,
                    "ir_only": 0,
                    "ambiguous": 0,
                },
            }

        staged_symbol_to_unit = dict(self.symbol_to_unit)
        if stale_unit_ids:
            stale_set = set(stale_unit_ids)
            staged_symbol_to_unit = {
                k: v for k, v in staged_symbol_to_unit.items() if v not in stale_set
            }

        all_chunks = [c for chunk_list in staged_chunks.values() for c in chunk_list]
        next_cpg = SemanticCPGBuilder()
        symbols = next_cpg.build_from_ast_chunks(all_chunks)
        chunk_to_symbol = {sym.chunk_id: sym for sym in symbols.values() if sym.chunk_id}

        lowered_results: list[
            tuple[
                Path,
                list[ASTChunkNode],
                Language | None,
                dict[str, LLVMIRGraph],
                dict[str, list[BinaryBasicBlock]],
                list[DWARFLineMapping],
                dict[str, tuple[str, int | None]],
                str,
            ]
        ] = []
        changed_items = list(changed_files.items())
        if len(changed_items) > 1:
            workers = min(4, len(changed_items))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                lowered_results = list(
                    pool.map(
                        lambda item: self._lower_single_file(
                            item[0], item[1], compile_flags_by_file
                        ),
                        changed_items,
                    )
                )
        else:
            for src_path, chunks in changed_items:
                lowered_results.append(
                    self._lower_single_file(src_path, chunks, compile_flags_by_file)
                )

        units: list[UnifiedCodeUnit] = []
        newly_lowered_ir: list[LLVMIRGraph] = []
        degraded_count = 0

        for (
            src_path,
            chunks,
            lang,
            ir_by_fn,
            bin_by_fn,
            dwarf_maps,
            elf_subprograms,
            file_error,
        ) in lowered_results:
            newly_lowered_ir.extend(ir_by_fn.values())
            for chunk in chunks:
                sym = chunk_to_symbol.get(chunk.chunk_id)
                ir_graph, bin_blocks = self._match_ir_and_binary(
                    chunk, ir_by_fn, bin_by_fn, elf_subprograms
                )
                reason = ""
                if ir_graph is not None or bin_blocks:
                    status = IRStatus.OK
                elif lang == Language.JAVASCRIPT or is_header_file(src_path):
                    status = IRStatus.NOT_APPLICABLE
                    if is_header_file(src_path):
                        reason = "Header file; lowered via including translation units"
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
                staged_unit_ids.setdefault(str(src_path), []).append(unit.unit_id)
                if sym is not None:
                    staged_symbol_to_unit[sym.symbol_uri] = unit.unit_id
                if chunk.symbol_name:
                    staged_symbol_to_unit.setdefault(chunk.symbol_name, unit.unit_id)

        updated_file_states = {
            p_str: (staged_hashes[p_str], staged_unit_ids.get(p_str, []))
            for p_str in updated_paths
        }
        # Persist batch BEFORE committing in-memory incremental cache state.
        count, deleted_count = self.store.apply_batch(
            upsert_units=units,
            delete_unit_ids=stale_unit_ids,
            save_file_states=updated_file_states,
            delete_file_paths=deleted_paths,
        )

        self._file_hashes = staged_hashes
        self._file_chunks = staged_chunks
        self._file_unit_ids = staged_unit_ids
        self.symbol_to_unit = staged_symbol_to_unit
        self.cpg = next_cpg

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

    def _lower_single_file(
        self,
        src_path: Path,
        chunks: list[ASTChunkNode],
        compile_flags_by_file: dict[str, list[str]],
    ) -> tuple[
        Path,
        list[ASTChunkNode],
        Language | None,
        dict[str, LLVMIRGraph],
        dict[str, list[BinaryBasicBlock]],
        list[DWARFLineMapping],
        dict[str, tuple[str, int | None]],
        str,
    ]:
        ir_by_fn: dict[str, LLVMIRGraph] = {}
        bin_by_fn: dict[str, list[BinaryBasicBlock]] = {}
        dwarf_maps: list[DWARFLineMapping] = []
        elf_subprograms: dict[str, tuple[str, int | None]] = {}
        lang = detect_language(src_path)
        file_error = ""

        if lang in (Language.C, Language.CPP) and not is_header_file(src_path):
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
                        elf_subprograms = self.dwarf.extract_subprogram_info(obj_path)
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
                        elf_subprograms = self.dwarf.extract_subprogram_info(obj_path)
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

        return (
            src_path,
            chunks,
            lang,
            ir_by_fn,
            bin_by_fn,
            dwarf_maps,
            elf_subprograms,
            file_error,
        )

    @staticmethod
    def _match_ir_and_binary(
        chunk: ASTChunkNode,
        ir_by_fn: dict[str, LLVMIRGraph],
        bin_by_fn: dict[str, list[BinaryBasicBlock]],
        elf_subprograms: dict[str, tuple[str, int | None]],
    ) -> tuple[LLVMIRGraph | None, list[BinaryBasicBlock]]:
        """Match an AST chunk to its lowered IR graph and ELF/bytecode blocks."""
        sym = chunk.symbol_name
        if not sym:
            return ir_by_fn.get(""), bin_by_fn.get("", [])

        # Fast path for exact matches (e.g. unmangled C functions or top-level Python).
        matched_ir = ir_by_fn.get(sym)
        matched_bin = bin_by_fn.get(sym)

        # Python qualified scope lookup (ClassName.method).
        if matched_ir is None and chunk.language == Language.PYTHON:
            scope_classes = [
                part.split("class_definition ", 1)[1].strip()
                for part in chunk.scope_header.splitlines()
                if part.startswith("class_definition ")
            ]
            if scope_classes:
                qual = f"{'.'.join(scope_classes)}.{sym}"
                matched_ir = ir_by_fn.get(qual)
                if matched_bin is None:
                    matched_bin = bin_by_fn.get(qual)

        # Line-aware fallback for mangled C++/Rust IR graphs or duplicated Python keys.
        if matched_ir is None:
            ir_candidates: list[tuple[int, str, LLVMIRGraph]] = []
            for key, g in ir_by_fn.items():
                leaf = g.source_name or extract_leaf_symbol_name(g.function_name)
                if leaf != sym:
                    continue
                if g.decl_line is not None and chunk.line_start <= g.decl_line <= chunk.line_end:
                    ir_candidates.append((0, key, g))
                else:
                    ir_candidates.append((1, key, g))
            if ir_candidates:
                ir_candidates.sort(key=lambda item: item[0])
                best_key, matched_ir = ir_candidates[0][1], ir_candidates[0][2]
                if matched_bin is None and best_key in bin_by_fn:
                    matched_bin = bin_by_fn[best_key]

        if matched_bin is None:
            bin_candidates: list[tuple[int, list[BinaryBasicBlock]]] = []
            for elf_sym, blocks in bin_by_fn.items():
                sp_name, sp_line = elf_subprograms.get(elf_sym, ("", None))
                leaf = sp_name or extract_leaf_symbol_name(elf_sym)
                if leaf != sym:
                    continue
                if sp_line is not None and chunk.line_start <= sp_line <= chunk.line_end:
                    bin_candidates.append((0, blocks))
                else:
                    bin_candidates.append((1, blocks))
            if bin_candidates:
                bin_candidates.sort(key=lambda item: item[0])
                matched_bin = bin_candidates[0][1]

        return matched_ir, (matched_bin or [])

    def inspect_lowering_pipeline(
        self,
        symbol_or_function: str,
        bb_cursor: int = 0,
        bb_page_size: int = 5,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> dict[str, Any]:
        """Return AST, SCIP symbol, LLVM IR, and paginated binary CFG for a function."""
        with self._lock:
            bb_cursor = max(0, bb_cursor)
            bb_page_size = max(1, min(bb_page_size, 50))
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
            next_cursor = (
                bb_cursor + bb_page_size if (bb_cursor + bb_page_size) < len(blocks) else None
            )

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
            raw_file = str(entry.get("file", ""))
            if not raw_file:
                continue
            file_p = Path(raw_file)
            if not file_p.is_absolute() and entry.get("directory"):
                file_p = Path(str(entry["directory"])) / file_p
            file_str = str(file_p.resolve())
            args = entry.get("arguments")
            if isinstance(args, list):
                out[file_str] = [str(a) for a in args]
            elif isinstance(entry.get("command"), str):
                try:
                    out[file_str] = shlex.split(entry["command"])
                except ValueError:
                    out[file_str] = entry["command"].split()
        return out


_PIPELINE = CodeIndexPipeline(
    storage_dir=os.environ.get("ASTRAEA_STORAGE_DIR") or None
)


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
    bounded_k = max(1, min(top_k, 100))
    hits = _PIPELINE.store.rrf_search(query=query, layers=layers, top_k=bounded_k)
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
    bounded_hops = max(1, min(max_hops, 5))
    slices = _PIPELINE.cpg.k_hop_slice(symbol_id, max_hops=bounded_hops)
    max_chars = DEFAULT_TOKEN_BUDGET * CHARS_PER_TOKEN_ESTIMATE
    out: list[dict[str, Any]] = []
    running_chars = 2
    for item in slices:
        item_len = len(json.dumps(item)) + 2
        if running_chars + item_len > max_chars:
            break
        out.append(item)
        running_chars += item_len
    return out


@mcp.tool()
def get_index_health() -> dict[str, Any]:
    """Return storage consistency, SQLite health, and CPG symbol/edge counts."""
    with _PIPELINE._lock:
        return {
            **_PIPELINE.store.health_check(),
            "cpg_symbols": len(_PIPELINE.cpg.symbols),
            "cpg_edges": _PIPELINE.cpg.graph.number_of_edges(),
        }


def main(argv: list[str] | None = None) -> None:
    """Run the Astraea code-multi-indexer MCP server over stdio."""
    global _PIPELINE
    parser = argparse.ArgumentParser(prog="astraea-index", description="Astraea MCP Server")
    parser.add_argument(
        "--storage-dir",
        default=os.environ.get("ASTRAEA_STORAGE_DIR")
        or str(Path.home() / ".cache" / "astraea" / "index"),
        help="Persistent directory for SQLite, Tantivy, and Qdrant storage.",
    )
    args, _ = parser.parse_known_args(argv)
    target_dir = Path(args.storage_dir) if args.storage_dir else None
    if target_dir != _PIPELINE.store.storage_dir:
        _PIPELINE.store.close()
        _PIPELINE = CodeIndexPipeline(storage_dir=target_dir)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
