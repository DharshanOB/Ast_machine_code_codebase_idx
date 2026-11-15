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
