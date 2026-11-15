"""Phase 2 Checkpoint 2 tests: LLVM SSA IR, Binary Lifting, DWARF, and SEC-01."""

from __future__ import annotations

from pathlib import Path

import pytest

from indexer.binary_lifter import BinaryLifter, normalize_instruction
from indexer.dwarf_aligner import DWARFAligner
from indexer.llvm_ir_pass import (
    LLVMIRPass,
    canonicalize_ssa_instruction,
    resolve_clang_binary,
    sanitize_compile_flags,
)
from indexer.models import DWARFLineMapping


def test_sec01_sanitize_compile_flags() -> None:
    """SEC-01: Strip -fplugin, @response_file, and arbitrary flags while keeping safe ones."""
    raw = [
        "-O3",
        "-fplugin=/tmp/evil.so",
        "@/tmp/args.rsp",
        "-I",
        "/usr/include/custom",
        "-isystem/opt/sys",
        "-DDEBUG=1",
        "-U",
        "NDEBUG",
        "-std=c17",
        "-Xclang",
        "-load",
    ]
    safe = sanitize_compile_flags(raw)
    assert safe == [
        "-I",
        "/usr/include/custom",
        "-isystem/opt/sys",
        "-DDEBUG=1",
        "-U",
        "NDEBUG",
        "-std=c17",
    ]
    assert all("-fplugin" not in f for f in safe)
    assert all(not f.startswith("@") for f in safe)


def test_g01_llvm_ir_clamp_convergence_and_g02_strength_reduction() -> None:
    """G-01 & G-02: Parse real LLVM IR via llvmlite without mocking."""
    ir_module = """
; ModuleID = 'clamp_and_scale.c'
define dso_local i32 @clamp_branch(i32 noundef %x) !dbg !10 {
entry:
  %cmp = icmp sgt i32 %x, 0, !dbg !11
  %cond = select i1 %cmp, i32 %x, i32 0, !dbg !11
  ret i32 %cond, !dbg !12
}

define dso_local i32 @clamp_ternary(i32 noundef %val) !dbg !20 {
entry:
  %c = icmp sgt i32 %val, 0, !dbg !21
  %res = select i1 %c, i32 %val, i32 0, !dbg !21
  ret i32 %res, !dbg !22
}

define dso_local i32 @scale_pos(i32 noundef %x) !dbg !30 {
entry:
  %mul = shl nsw i32 %x, 2, !dbg !31
  ret i32 %mul, !dbg !32
}

!llvm.module.flags = !{!2}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!llvm.dbg.cu = !{!0}
!0 = distinct !DICompileUnit(
    language: DW_LANG_C11, file: !1, isOptimized: true, emissionKind: LineTablesOnly
)
!1 = !DIFile(filename: "clamp.c", directory: "/tmp")
!10 = distinct !DISubprogram(name: "clamp_branch", scope: !1, file: !1, line: 1, unit: !0)
!11 = !DILocation(line: 2, column: 9, scope: !10)
!12 = !DILocation(line: 3, column: 5, scope: !10)
!20 = distinct !DISubprogram(name: "clamp_ternary", scope: !1, file: !1, line: 6, unit: !0)
!21 = !DILocation(line: 7, column: 12, scope: !20)
!22 = !DILocation(line: 7, column: 5, scope: !20)
!30 = distinct !DISubprogram(name: "scale_pos", scope: !1, file: !1, line: 10, unit: !0)
!31 = !DILocation(line: 11, column: 14, scope: !30)
!32 = !DILocation(line: 11, column: 5, scope: !30)
"""
    ir_pass = LLVMIRPass()
    graphs = {g.function_name: g for g in ir_pass.parse_ir_module(ir_module, source_file="clamp.c")}

    assert "clamp_branch" in graphs
    assert "clamp_ternary" in graphs
    assert "scale_pos" in graphs

    g_branch = graphs["clamp_branch"]
    g_ternary = graphs["clamp_ternary"]
    # G-01: Identical canonical token sequences & isomorphic data-flow edges
    assert g_branch.canonical_tokens == g_ternary.canonical_tokens
    assert "icmp" in g_branch.canonical_tokens
    assert "select" in g_branch.canonical_tokens
    assert 2 in g_branch.dbg_line_map.values()

    # G-02: Strength reduction shl with data dependency from %x
    g_scale = graphs["scale_pos"]
    assert "shl" in g_scale.canonical_tokens
    assert any(e.flow_type == "data" for e in g_scale.edges)


def test_ut03_phi_node_loop_backedge() -> None:
    """UT-03: Construct cyclic CFG back-edge and phi data-flow edges."""
    loop_ir = """
define i32 @sum_loop(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc_next, %loop ]
  %acc_next = add i32 %acc, %i
  %next = add i32 %i, 1
  %cond = icmp slt i32 %next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret i32 %acc_next
}
"""
    ir_pass = LLVMIRPass()
    graphs = ir_pass.parse_ir_module(loop_ir, source_file="loop.c")
    assert len(graphs) == 1
    g = graphs[0]
    assert "phi" in g.canonical_tokens
    ctrl_edges = [e for e in g.edges if e.flow_type == "ctrl"]
    # Back-edge exists where target <= source node_id
    assert any(e.target <= e.source for e in ctrl_edges)


def test_g01_x86_machine_code_clamp_identity() -> None:
    """G-01: Disassemble x86-64 branchless clamp sequence via Capstone without mocking."""
    # xor eax, eax (31 c0); test edi, edi (85 ff); cmovg eax, edi (0f 4f c7); ret (c3)
    clamp_bytes = bytes.fromhex("31 c0 85 ff 0f 4f c7 c3")
    lifter = BinaryLifter()
    blocks = lifter.lift_bytes(clamp_bytes, base_va=0x1000, function_name="clamp", arch="x86_64")
    assert len(blocks) == 1
    assert blocks[0].normalized_instructions == [
        "XOR REG32_A, REG32_A",
        "TEST REG32_ARG1, REG32_ARG1",
        "CMOVG REG32_A, REG32_ARG1",
        "RET",
    ]


def test_ut04_instruction_normalization_oov() -> None:
    """UT-04: Stack displacements, jump targets, and immediates normalize deterministically."""
    assert normalize_instruction("mov", "eax, dword ptr [rbp - 0x18]", "x86_64") == (
        "MOV REG32_A, dword ptr [REG_FP - DISP_LOCAL]"
    )
    assert normalize_instruction("add", "edi, 0x1234", "x86_64") == "ADD REG32_ARG1, <IMM_CONST>"
    assert normalize_instruction("jne", "0x401050", "x86_64") == "JNE <BB_TARGET>"
    norm_ssa, _ = canonicalize_ssa_instruction("%cmp = icmp sgt i32 %x, 0")
    assert norm_ssa == "<SSA_REG_0> = icmp sgt i32 <SSA_REG_1>, 0"


def test_dwarf_aligner_line_lookup() -> None:
    aligner = DWARFAligner()
    mappings = [
        DWARFLineMapping(
            file_path="clamp.c",
            line_start=2,
            line_end=2,
            va_start=0x1000,
            va_end=0x1008,
        )
    ]
    assert len(aligner.lookup_va_for_line(mappings, 2)) == 1
    assert aligner.lookup_line_for_va(mappings, 0x1004) is not None
    assert aligner.lookup_line_for_va(mappings, 0x2000) is None


@pytest.mark.skipif(resolve_clang_binary() is None, reason="clang-18/clang not installed in VM")
def test_end_to_end_clang_ir_and_elf_lifting(tmp_path: Path) -> None:
    src = tmp_path / "clamp.c"
    src.write_text(
        "int clamp_branch(int x) {\n  if (x > 0) return x;\n  return 0;\n}\n",
        encoding="utf-8",
    )
    ir_pass = LLVMIRPass()
    ir_text = ir_pass.compile_to_ir(src)
    graphs = ir_pass.parse_ir_module(ir_text, source_file=str(src))
    assert any(g.function_name == "clamp_branch" for g in graphs)

    lifter = BinaryLifter()
    obj_path = lifter.compile_to_object(src, tmp_path / "clamp.o")
    lifted = lifter.lift_elf(obj_path)
    assert "clamp_branch" in lifted

    aligner = DWARFAligner()
    mappings = aligner.extract_mappings(obj_path, default_file=str(src))
    assert len(mappings) > 0


@pytest.mark.skipif(resolve_clang_binary() is None, reason="clang-18/clang not installed in VM")
def test_optimized_o2_and_stripped_binary_dwarf(tmp_path: Path) -> None:
    """Verify -O2 inlining DWARF extraction and graceful handling of stripped binaries."""
    import shutil
    import subprocess

    src = tmp_path / "inlined.c"
    src.write_text(
        "static inline int helper_add(int a, int b) { return a + b; }\n"
        "int entry_fn(int x) { return helper_add(x, 7); }\n",
        encoding="utf-8",
    )
    lifter = BinaryLifter()
    o2_obj = lifter.compile_to_object(src, tmp_path / "inlined_o2.o", opt_level="-O2")
    lifted = lifter.lift_elf(o2_obj)
    assert "entry_fn" in lifted

    aligner = DWARFAligner()
    o2_mappings = aligner.extract_mappings(o2_obj, default_file=str(src))
    assert len(o2_mappings) > 0

    strip_bin = shutil.which("strip")
    if strip_bin is not None:
        stripped_obj = tmp_path / "stripped.o"
        shutil.copy2(o2_obj, stripped_obj)
        subprocess.run([strip_bin, "--strip-debug", str(stripped_obj)], check=True)
        assert aligner.extract_mappings(stripped_obj, default_file=str(src)) == []
        assert "entry_fn" in lifter.lift_elf(stripped_obj)


@pytest.mark.skipif(resolve_clang_binary() is None, reason="clang-18/clang not installed in VM")
def test_validate_cpg_edges_against_llvm_ir_calls(tmp_path: Path) -> None:
    """Validate CPG CALLS edges against compiler-generated LLVM IR call instructions."""
    from indexer.ast_chunker import ASTChunker
    from indexer.semantic_cpg import SemanticCPGBuilder

    src = tmp_path / "calls.c"
    src.write_text(
        "__attribute__((noinline)) int leaf(int x) { return x + 1; }\n"
        "int caller(int y) { return leaf(y) * 2; }\n",
        encoding="utf-8",
    )
    chunks = ASTChunker().chunk_file(src)
    cpg = SemanticCPGBuilder()
    cpg.build_from_ast_chunks(chunks)

    ir_pass = LLVMIRPass()
    ir_text = ir_pass.compile_to_ir(src, opt_level="-O0")
    graphs = ir_pass.parse_ir_module(ir_text, source_file=str(src))

    validation = cpg.validate_against_ir_calls(graphs)
    assert ("caller", "leaf") in validation["confirmed"]


def test_phi_loop_backedge_and_target_flag_sanitization() -> None:
    """Two-pass SSA builder preserves loop-carried phi back-edges and allows target flags."""
    ir_sample = """
define i32 @loop_sum(i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %i, 1
  %cond = icmp slt i32 %next, %n
  br i1 %cond, label %loop, label %exit
exit:
  ret i32 %next
}
"""
    graphs = LLVMIRPass().parse_ir_module(ir_sample, source_file="loop.c")
    assert len(graphs) == 1
    nodes_by_id = {n.node_id: n for n in graphs[0].nodes}
    phi_back_edges = [
        e
        for e in graphs[0].edges
        if e.flow_type == "data"
        and nodes_by_id[e.target].opcode == "phi"
        and nodes_by_id[e.source].opcode == "add"
    ]
    assert len(phi_back_edges) >= 1

    safe = sanitize_compile_flags(
        ["--target=x86_64-unknown-linux-gnu", "-target", "aarch64-linux-gnu", "-fplugin=bad.so"]
    )
    assert safe == ["--target=x86_64-unknown-linux-gnu", "-target", "aarch64-linux-gnu"]


@pytest.mark.skipif(resolve_clang_binary() is None, reason="clang-18/clang not installed in VM")
def test_cpp_function_sections_and_header_not_applicable(tmp_path: Path) -> None:
    """C++ mangled functions with -ffunction-sections and .hpp headers lower cleanly."""
    from indexer.models import IRStatus
    from mcp_server.code_index_server import CodeIndexPipeline

    (tmp_path / "ops.hpp").write_text(
        "namespace math { int square(int x); }\n",
        encoding="utf-8",
    )
    (tmp_path / "ops.cpp").write_text(
        '#include "ops.hpp"\n'
        "namespace math {\n"
        "int square(int x) { return x * x + 1; }\n"
        "}\n",
        encoding="utf-8",
    )
    cc_path = tmp_path / "compile_commands.json"
    cc_path.write_text(
        '[{"directory": "'
        + str(tmp_path)
        + '", "file": "'
        + str(tmp_path / "ops.cpp")
        + '", "arguments": ["clang++", "-O1", "-ffunction-sections"]}]',
        encoding="utf-8",
    )
    pipeline = CodeIndexPipeline(storage_dir=tmp_path / ".astraea")
    try:
        res = pipeline.index_workspace(str(tmp_path), compile_commands_path=str(cc_path))
        assert res["indexed_units"] >= 2
        units = list(pipeline.store.units.values())
        header_units = [u for u in units if u.ast_chunk.file_path.endswith("ops.hpp")]
        cpp_units = [u for u in units if u.ast_chunk.file_path.endswith("ops.cpp")]
        assert header_units and all(u.ir_status == IRStatus.NOT_APPLICABLE for u in header_units)
        sq = next(u for u in cpp_units if u.ast_chunk.symbol_name == "square")
        assert sq.ir_status == IRStatus.OK
        assert sq.ir_graph is not None
        assert len(sq.binary_blocks) > 0
    finally:
        pipeline.store.close()


def test_rust_text_section_elf_lifting_and_indexing(tmp_path: Path) -> None:
    """Rust per-function .text.* ELF sections and v0 mangled IR match end-to-end."""
    import shutil

    if shutil.which("rustc") is None:
        pytest.skip("rustc not installed in VM")

    from indexer.models import IRStatus
    from mcp_server.code_index_server import CodeIndexPipeline

    (tmp_path / "lib.rs").write_text(
        "#[inline(never)]\n"
        "pub fn double_plus(x: i32) -> i32 {\n"
        "    if x > 0 { x * 2 + 1 } else { 1 }\n"
        "}\n",
        encoding="utf-8",
    )
    pipeline = CodeIndexPipeline(storage_dir=tmp_path / ".astraea")
    try:
        res = pipeline.index_workspace(str(tmp_path))
        assert res["indexed_units"] >= 1
        units = [
            u for u in pipeline.store.units.values() if u.ast_chunk.symbol_name == "double_plus"
        ]
        assert len(units) == 1
        unit = units[0]
        assert unit.ir_status == IRStatus.OK
        assert unit.ir_graph is not None
        assert len(unit.binary_blocks) > 0
    finally:
        pipeline.store.close()
