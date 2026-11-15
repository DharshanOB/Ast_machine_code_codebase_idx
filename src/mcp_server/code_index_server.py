"""Phase 4: Antigravity FastMCP server & multi-layer indexing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from indexer.ast_chunker import ASTChunker, detect_language
from indexer.binary_lifter import BinaryLifter
from indexer.dwarf_aligner import DWARFAligner
from indexer.llvm_ir_pass import LLVMIRPass
from indexer.models import (
    ASTChunkNode,
    BinaryBasicBlock,
    DWARFLineMapping,
    IRStatus,
    Language,
    LLVMIRGraph,
    UnifiedCodeUnit,
)
from indexer.semantic_cpg import SemanticCPGBuilder
from storage.hybrid_store import HybridCodeStore

DEFAULT_TOKEN_BUDGET = 2048
CHARS_PER_TOKEN_ESTIMATE = 4

mcp = MCPServer("code-multi-indexer")


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

    def index_workspace(
        self, repo_path: str | Path, compile_commands_path: str | Path | None = None
    ) -> dict[str, Any]:
        """Index all supported source files in repo_path across AST, CPG, IR, and Binary layers."""
        root = Path(repo_path)
        if not root.exists():
            return {"indexed_units": 0, "error": f"Path does not exist: {repo_path}"}

        compile_flags_by_file = self._load_compile_commands(compile_commands_path)
        all_chunks: list[ASTChunkNode] = []
        file_chunks: dict[Path, list[ASTChunkNode]] = {}

        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            rel_parts = file_path.relative_to(root).parts
            if any(p.startswith(".") for p in rel_parts):
                continue
            if detect_language(file_path) is None:
                continue
            chunks = self.chunker.chunk_file(file_path)
            if chunks:
                all_chunks.extend(chunks)
                file_chunks[file_path] = chunks

        symbols = self.cpg.build_from_ast_chunks(all_chunks)
        chunk_to_symbol = {sym.chunk_id: sym for sym in symbols.values() if sym.chunk_id}

        units: list[UnifiedCodeUnit] = []
        degraded_count = 0

        for src_path, chunks in file_chunks.items():
            ir_by_fn: dict[str, LLVMIRGraph] = {}
            bin_by_fn: dict[str, list[BinaryBasicBlock]] = {}
            dwarf_maps: list[DWARFLineMapping] = []
            lang = detect_language(src_path)

            if lang in (Language.C, Language.CPP) and self.ir_pass.clang_bin is not None:
                flags = compile_flags_by_file.get(str(src_path.resolve()), [])
                try:
                    ir_text = self.ir_pass.compile_to_ir(src_path, raw_flags=flags)
                    for g in self.ir_pass.parse_ir_module(ir_text, source_file=str(src_path)):
                        ir_by_fn[g.function_name] = g
                    obj_path = self.lifter.compile_to_object(src_path, raw_flags=flags)
                    bin_by_fn = self.lifter.lift_elf(obj_path)
                    dwarf_maps = self.dwarf.extract_mappings(obj_path, default_file=str(src_path))
                    self.dwarf.attach_ir_dbg_ids(dwarf_maps, list(ir_by_fn.values()))
                except Exception:
                    pass

            for chunk in chunks:
                sym = chunk_to_symbol.get(chunk.chunk_id)
                ir_graph = ir_by_fn.get(chunk.symbol_name)
                bin_blocks = bin_by_fn.get(chunk.symbol_name, [])
                status = (
                    IRStatus.OK
                    if (ir_graph is not None or bin_blocks)
                    else IRStatus.DEGRADED
                )
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
                )
                units.append(unit)
                if sym is not None:
                    self.symbol_to_unit[sym.symbol_uri] = unit.unit_id
                if chunk.symbol_name:
                    self.symbol_to_unit[chunk.symbol_name] = unit.unit_id

        count = self.store.upsert_units(units)
        return {
            "indexed_units": count,
            "degraded_units": degraded_count,
            "symbols_count": len(symbols),
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
def index_workspace(repo_path: str, compile_commands_path: str | None = None) -> dict[str, Any]:
    """Index a repository across AST, SCIP/CPG, LLVM IR, and disassembled Machine Code."""
    return _PIPELINE.index_workspace(repo_path, compile_commands_path)


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


def main() -> None:
    """Run the Antigravity code-multi-indexer MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
