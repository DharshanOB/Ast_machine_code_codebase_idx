"""Phase 4 E2E, MCP Server, and Hypothesis Property Tests (PT-01 & E2E-01)."""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from indexer.binary_lifter import BinaryLifter
from indexer.models import IRStatus
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

    pipeline = CodeIndexPipeline(storage_dir=tmp_path / ".idx")
    summary = pipeline.index_workspace(tmp_path)
    assert summary["indexed_units"] == 2
    assert summary["symbols_count"] >= 2

    hits = pipeline.store.rrf_search("scale_pos", top_k=5)
    assert any(h.symbol_name == "scale_pos" for h in hits)

    inspected = pipeline.inspect_lowering_pipeline("scale_pos")
    assert inspected["ast"]["symbol_name"] == "scale_pos"
    assert inspected["scip_symbol"] is not None
    assert inspected["ir_status"] in (IRStatus.OK.value, IRStatus.DEGRADED.value)
    assert len(json.dumps(inspected)) <= 2048 * 4

    # Python file gracefully degrades to Layer 1 + Layer 2 without aborting batch
    py_inspected = pipeline.inspect_lowering_pipeline("helper_fn")
    assert py_inspected["ir_status"] == IRStatus.DEGRADED.value


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
