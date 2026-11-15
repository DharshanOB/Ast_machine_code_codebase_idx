"""Phase 4 E2E, MCP Server, and Hypothesis Property Tests (PT-01 & E2E-01)."""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from indexer.ast_chunker import ASTChunker
from indexer.binary_lifter import BinaryLifter
from indexer.llvm_ir_pass import BUILD_TMP_ROOT
from indexer.models import IRStatus
from indexer.semantic_cpg import SemanticCPGBuilder
from mcp_server.code_index_server import CodeIndexPipeline, mcp


@given(raw_bytes=st.binary(min_size=0, max_size=256))
@settings(max_examples=100)
def test_pt01_binary_lifter_random_bytes_never_crashes(raw_bytes: bytes) -> None:
    """PT-01: Arbitrary random byte streams never crash or hang the disassembler."""
    lifter = BinaryLifter()
    blocks = lifter.lift_bytes(raw_bytes, base_va=0x1000, function_name="fuzz_fn", arch="x86_64")
    assert isinstance(blocks, list)


def test_e2e01_pipeline_indexing_and_degraded_fallback(tmp_path: Path) -> None:
    """E2E-01: Index a workspace, verify AST/CPG/RRF/inspect, and graceful DEGRADED fallback."""
    (tmp_path / "math.c").write_text(
        "int scale_pos(int x) {\n    return x * 4;\n}\n",
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text(
        "def helper_fn(n: int) -> int:\n    return n + 1\n",
        encoding="utf-8",
    )
    (tmp_path / "broken.c").write_text(
        "int broken_fn(int x) { return unknown_symbol_xyz; }\n",
        encoding="utf-8",
    )

    pipeline = CodeIndexPipeline(storage_dir=tmp_path / ".idx")
    summary = pipeline.index_workspace(tmp_path)
    assert summary["indexed_units"] == 3
    assert summary["symbols_count"] >= 3

    hits = pipeline.store.rrf_search("scale_pos", top_k=5)
    assert any(h.symbol_name == "scale_pos" for h in hits)

    inspected = pipeline.inspect_lowering_pipeline("scale_pos")
    assert inspected["ast"]["symbol_name"] == "scale_pos"
    assert inspected["scip_symbol"] is not None
    assert inspected["ir_status"] in (IRStatus.OK.value, IRStatus.DEGRADED.value)
    assert len(json.dumps(inspected)) <= 2048 * 4

    # Python file lowers to bytecode CFG + basic blocks
    py_inspected = pipeline.inspect_lowering_pipeline("helper_fn")
    assert py_inspected["ir_status"] == IRStatus.OK.value

    # Broken C file gracefully degrades to Layer 1 + Layer 2 without aborting batch
    broken_inspected = pipeline.inspect_lowering_pipeline("broken_fn")
    assert broken_inspected["ir_status"] == IRStatus.DEGRADED.value
    assert broken_inspected["degraded_reason"] != ""


def test_workspace_skips_ignored_dirs_and_cleans_tmp(tmp_path: Path) -> None:
    """Verify node_modules/build are skipped and no temp dirs leak in BUILD_TMP_ROOT."""
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text(
        "function dep() { return 1; }\n", encoding="utf-8"
    )
    (tmp_path / "app.py").write_text("def main_fn():\n    return 42\n", encoding="utf-8")

    before_dirs = set(BUILD_TMP_ROOT.iterdir()) if BUILD_TMP_ROOT.exists() else set()
    pipeline = CodeIndexPipeline()
    summary = pipeline.index_workspace(tmp_path)
    after_dirs = set(BUILD_TMP_ROOT.iterdir()) if BUILD_TMP_ROOT.exists() else set()

    assert summary["indexed_units"] == 1
    assert after_dirs == before_dirs


def test_cpg_ignores_calls_in_comments_and_strings() -> None:
    """Verify AST-based CPG call extraction ignores comments and string literals."""
    chunker = ASTChunker()
    chunks = chunker.chunk_file(
        "sample.py",
        source=(
            "def target_fn():\n"
            "    return 1\n\n"
            "def caller_fn():\n"
            "    # target_fn() in comment\n"
            '    msg = "target_fn() in string"\n'
            "    return msg\n"
        ),
    )
    cpg = SemanticCPGBuilder()
    symbols = cpg.build_from_ast_chunks(chunks)
    caller_uri = next(u for u, s in symbols.items() if s.display_name == "caller_fn")
    assert cpg.k_hop_slice(caller_uri, max_hops=1) == []


def test_cpg_restricts_calls_to_compatible_languages() -> None:
    """Verify a C call to helper() does not link to an unrelated Python helper()."""
    chunker = ASTChunker()
    c_chunks = chunker.chunk_file(
        "mod.c", source="int helper(int x);\nint run(int x) { return helper(x); }\n"
    )
    py_chunks = chunker.chunk_file("other.py", source="def helper(x: int) -> int:\n    return x\n")

    cpg = SemanticCPGBuilder()
    symbols = cpg.build_from_ast_chunks(c_chunks + py_chunks)
    run_uri = next(u for u, s in symbols.items() if s.display_name == "run")
    assert cpg.k_hop_slice(run_uri, max_hops=1) == []


def test_binary_lifter_cleanup_removes_owned_tmp_dirs(tmp_path: Path) -> None:
    """Verify BinaryLifter.cleanup() removes temp dirs created by direct compile_to_object()."""
    from indexer.llvm_ir_pass import resolve_clang_binary

    if resolve_clang_binary() is None:
        return
    c_file = tmp_path / "ok.c"
    c_file.write_text("int ok(int x) { return x + 1; }\n", encoding="utf-8")
    lifter = BinaryLifter()
    obj_path = lifter.compile_to_object(c_file)
    assert obj_path.exists()
    lifter.cleanup()
    assert not obj_path.parent.exists()


def test_degraded_reason_sanitizes_compiler_source_echo(tmp_path: Path) -> None:
    """Verify degraded_reason does not echo raw #included file contents."""
    secret_hdr = tmp_path / "secret.h"
    secret_hdr.write_text("SECRET_TOKEN_SHOULD_NOT_LEAK\n", encoding="utf-8")
    bad_c = tmp_path / "bad.c"
    bad_c.write_text('#include "secret.h"\nint f(void) { return 0; }\n', encoding="utf-8")

    pipeline = CodeIndexPipeline()
    pipeline.index_workspace(tmp_path)
    inspected = pipeline.inspect_lowering_pipeline("f")
    assert "SECRET_TOKEN_SHOULD_NOT_LEAK" not in inspected.get("degraded_reason", "")


def test_mcp_tools_registered() -> None:
    """Verify all 5 MCP tools are registered on MCPServer."""
    tool_names = {t.name for t in mcp._tool_manager.list_tools()}
    assert {
        "index_workspace",
        "search_code_hybrid",
        "inspect_lowering_pipeline",
        "find_ir_or_binary_clones",
        "trace_symbol_dataflow",
    }.issubset(tool_names)
