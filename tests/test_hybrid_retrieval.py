"""Phase 3 Checkpoint 3 tests: Tri-modal hybrid store, WL IR vectors, and RRF retrieval."""

from __future__ import annotations

from pathlib import Path

from indexer.ast_chunker import ASTChunker
from indexer.binary_lifter import BinaryLifter
from indexer.llvm_ir_pass import LLVMIRPass
from indexer.models import IRStatus, Language, UnifiedCodeUnit
from storage.hybrid_store import HybridCodeStore, compute_wl_ir_vector, cosine_similarity

_SAMPLE_IR = """
define i32 @clamp_branch(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, 0
  %cond = select i1 %cmp, i32 %x, i32 0
  ret i32 %cond
}

define i32 @clamp_ternary(i32 %v) {
entry:
  %c = icmp sgt i32 %v, 0
  %res = select i1 %c, i32 %v, i32 0
  ret i32 %res
}

define i32 @scale_pos(i32 %x) {
entry:
  %mul = shl nsw i32 %x, 2
  ret i32 %mul
}
"""


def test_wl_ir_vector_isomorphism() -> None:
    ir_pass = LLVMIRPass()
    graphs = {g.function_name: g for g in ir_pass.parse_ir_module(_SAMPLE_IR)}
    v_branch = compute_wl_ir_vector(graphs["clamp_branch"])
    v_ternary = compute_wl_ir_vector(graphs["clamp_ternary"])
    v_scale = compute_wl_ir_vector(graphs["scale_pos"])

    assert cosine_similarity(v_branch, v_ternary) >= 0.99
    assert cosine_similarity(v_branch, v_scale) < 0.95


def test_hybrid_store_rrf_and_clone_detection(tmp_path: Path) -> None:
    chunker = ASTChunker()
    ir_pass = LLVMIRPass()
    lifter = BinaryLifter()

    ir_graphs = {g.function_name: g for g in ir_pass.parse_ir_module(_SAMPLE_IR)}
    clamp_bin = lifter.lift_bytes(
        bytes.fromhex("31 c0 85 ff 0f 4f c7 c3"), function_name="clamp"
    )
    scale_bin = lifter.lift_bytes(
        bytes.fromhex("89 f8 c1 e0 02 c3"), function_name="scale_pos"
    )

    c_branch = chunker.chunk_source(
        "int clamp_branch(int x) { if (x > 0) return x; return 0; }\n",
        Language.C,
        "clamp.c",
    )[0]
    c_ternary = chunker.chunk_source(
        "int clamp_ternary(int v) { return (v > 0) ? v : 0; }\n",
        Language.C,
        "clamp_ternary.c",
    )[0]
    c_scale = chunker.chunk_source(
        "int scale_pos(int x) { return x * 4; }\n",
        Language.C,
        "scale.c",
    )[0]

    u_branch = UnifiedCodeUnit(
        ast_chunk=c_branch,
        ir_graph=ir_graphs["clamp_branch"],
        binary_blocks=clamp_bin,
        ir_status=IRStatus.OK,
    )
    u_ternary = UnifiedCodeUnit(
        ast_chunk=c_ternary,
        ir_graph=ir_graphs["clamp_ternary"],
        binary_blocks=clamp_bin,
        ir_status=IRStatus.OK,
    )
    u_scale = UnifiedCodeUnit(
        ast_chunk=c_scale,
        ir_graph=ir_graphs["scale_pos"],
        binary_blocks=scale_bin,
        ir_status=IRStatus.OK,
    )

    store = HybridCodeStore(storage_dir=tmp_path / "idx")
    count = store.upsert_units([u_branch, u_ternary, u_scale], commit_sha="abc123")
    assert count == 3

    hits = store.rrf_search("scale_pos", top_k=3)
    assert len(hits) >= 1
    assert hits[0].symbol_name == "scale_pos"

    clones = store.find_clones(u_branch.unit_id, modality="ir_structural", threshold=0.95)
    clone_ids = [cid for cid, _ in clones]
    assert u_ternary.unit_id in clone_ids
    assert u_scale.unit_id not in clone_ids
