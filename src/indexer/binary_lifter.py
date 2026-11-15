"""Layer 4: Machine code lifter, BB-CFG reconstructor, and instruction normalizer."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from itertools import islice
from pathlib import Path

import capstone
import lief

from indexer.llvm_ir_pass import (
    BUILD_TMP_ROOT,
    MAX_COMPILER_TIMEOUT_S,
    resolve_clang_binary,
    sanitize_compile_flags,
)
from indexer.models import BinaryBasicBlock

_X86_REG_MAP: dict[str, str] = {
    "rax": "REG64_A",
    "eax": "REG32_A",
    "ax": "REG16_A",
    "al": "REG8_A",
    "rdi": "REG64_ARG1",
    "edi": "REG32_ARG1",
    "rsi": "REG64_ARG2",
    "esi": "REG32_ARG2",
    "rdx": "REG64_ARG3",
    "edx": "REG32_ARG3",
    "rcx": "REG64_ARG4",
    "ecx": "REG32_ARG4",
    "r8": "REG64_ARG5",
    "r8d": "REG32_ARG5",
    "r9": "REG64_ARG6",
    "r9d": "REG32_ARG6",
    "rbx": "REG64_CALLEE1",
    "ebx": "REG32_CALLEE1",
    "rbp": "REG_FP",
    "ebp": "REG_FP",
    "rsp": "REG_SP",
    "esp": "REG_SP",
    "rip": "REG_PC",
}

_ARM64_REG_MAP: dict[str, str] = {
    "x0": "REG64_ARG1",
    "w0": "REG32_ARG1",
    "x1": "REG64_ARG2",
    "w1": "REG32_ARG2",
    "x2": "REG64_ARG3",
    "w2": "REG32_ARG3",
    "x3": "REG64_ARG4",
    "w3": "REG32_ARG4",
    "x29": "REG_FP",
    "fp": "REG_FP",
    "x30": "REG_LR",
    "lr": "REG_LR",
    "sp": "REG_SP",
}

_X86_JUMP_MNEMONICS = frozenset(
    {
        "jmp",
        "je",
        "jne",
        "jz",
        "jnz",
        "jg",
        "jge",
        "jl",
        "jle",
        "ja",
        "jae",
        "jb",
        "jbe",
        "js",
        "jns",
        "ret",
        "retq",
    }
)
_ARM64_JUMP_PREFIXES = ("b.", "b", "bl", "cbz", "cbnz", "tbz", "tbnz", "ret")

_HEX_OR_DEC_RE = re.compile(r"(?<![A-Za-z0-9_])(-?0x[0-9a-fA-F]+|-?\d+)(?![A-Za-z0-9_])")
_MEM_DISP_RE = re.compile(r"\[\s*(REG_FP|REG_SP|REG_PC)\s*([+-])\s*(?:0x[0-9a-fA-F]+|\d+)\s*\]")


def normalize_instruction(mnemonic: str, op_str: str, arch: str = "x86_64") -> str:
    """Apply deterministic instruction normalization Phi(I) to mitigate OOV explosion."""
    mn = mnemonic.strip().upper()
    ops = op_str.strip()
    if not ops:
        return mn

    is_jump = (
        mnemonic.lower() in _X86_JUMP_MNEMONICS
        if arch == "x86_64"
        else mnemonic.lower().startswith(_ARM64_JUMP_PREFIXES)
    )
    if is_jump and re.fullmatch(r"0x[0-9a-fA-F]+|#?0x[0-9a-fA-F]+|\d+", ops):
        return f"{mn} <BB_TARGET>"

    reg_map = _X86_REG_MAP if arch == "x86_64" else _ARM64_REG_MAP
    norm_ops = ops
    for reg, canonical in reg_map.items():
        norm_ops = re.sub(rf"\b{re.escape(reg)}\b", canonical, norm_ops, flags=re.IGNORECASE)

    norm_ops = _MEM_DISP_RE.sub(r"[\1 \2 DISP_LOCAL]", norm_ops)

    def _norm_imm(match: re.Match[str]) -> str:
        raw = match.group(1)
        try:
            val = int(raw, 16) if "x" in raw.lower() else int(raw, 10)
        except ValueError:
            return raw
        if abs(val) > 256:
            return "<IMM_CONST>"
        return raw

    norm_ops = _HEX_OR_DEC_RE.sub(_norm_imm, norm_ops)
    norm_ops = re.sub(r"\s+", " ", norm_ops).strip()
    return f"{mn} {norm_ops}"


MAX_DISASM_BYTES: int = 1_048_576  # 1 MiB cap per function
MAX_DISASM_INSNS: int = 50_000


class BinaryLifter:
    """Compiles or opens ELF binaries, disassembles functions, and extracts normalized BB-CFGs."""

    def __init__(self, clang_bin: str = "clang-18") -> None:
        self.clang_bin = resolve_clang_binary(clang_bin)
        self.rustc_bin = shutil.which("rustc")
        BUILD_TMP_ROOT.mkdir(parents=True, exist_ok=True)

    def compile_to_object(
        self,
        source_path: str | Path,
        out_obj_path: str | Path | None = None,
        raw_flags: list[str] | None = None,
        opt_level: str = "-O1",
        target_triple: str | None = None,
    ) -> Path:
        """Compile a C/C++ source file to a relocatable ELF object file with DWARF info."""
        if self.clang_bin is None:
            raise RuntimeError("No clang compiler binary found on PATH")

        src = Path(source_path)
        safe_flags = sanitize_compile_flags(raw_flags or [])
        target_path = (
            Path(out_obj_path)
            if out_obj_path is not None
            else Path(tempfile.mkdtemp(dir=BUILD_TMP_ROOT)) / f"{src.stem}.o"
        )
        cmd = [
            self.clang_bin,
            "-c",
            opt_level,
            "-g",
            "-ferror-limit=20",
            "-ftemplate-depth=64",
            *safe_flags,
        ]
        if target_triple:
            cmd.extend(["-target", target_triple])
        cmd.extend([str(src), "-o", str(target_path)])

        proc = subprocess.run(
            cmd,
            shell=False,
            capture_output=True,
            text=True,
            timeout=MAX_COMPILER_TIMEOUT_S,
        )
        if proc.returncode != 0 or not target_path.exists():
            raise RuntimeError(f"Object compilation failed: {proc.stderr.strip()}")
        return target_path

    def compile_rust_to_object(
        self,
        source_path: str | Path,
        out_obj_path: str | Path | None = None,
    ) -> Path:
        """Compile a standalone Rust file to an ELF object file with DWARF info."""
        if self.rustc_bin is None:
            raise RuntimeError("No rustc compiler binary found on PATH")
        src = Path(source_path)
        target_path = (
            Path(out_obj_path)
            if out_obj_path is not None
            else Path(tempfile.mkdtemp(dir=BUILD_TMP_ROOT)) / f"{src.stem}.o"
        )
        cmd = [
            self.rustc_bin,
            "--crate-type",
            "lib",
            "--emit=obj",
            "-C",
            "debuginfo=2",
            "-C",
            "opt-level=1",
            str(src),
            "-o",
            str(target_path),
        ]
        proc = subprocess.run(
            cmd,
            shell=False,
            capture_output=True,
            text=True,
            timeout=MAX_COMPILER_TIMEOUT_S,
        )
        if proc.returncode != 0 or not target_path.exists():
            raise RuntimeError(f"Rustc object compilation failed: {proc.stderr.strip()}")
        return target_path

    def lift_elf(
        self, elf_path: str | Path, function_filter: str | None = None
    ) -> dict[str, list[BinaryBasicBlock]]:
        """Parse an ELF binary/object with LIEF and disassemble its .text function symbols."""
        path = Path(elf_path)
        if not path.exists():
            return {}

        binary = lief.parse(str(path))
        if binary is None or not isinstance(binary, lief.ELF.Binary):
            return {}

        arch = self._detect_arch(binary)
        text_section = binary.get_section(".text")
        if text_section is None:
            return {}

        text_bytes = bytes(text_section.content)
        text_va = int(text_section.virtual_address)
        result: dict[str, list[BinaryBasicBlock]] = {}

        for sym in binary.symbols:
            if getattr(sym.type, "name", "") != "FUNC" or sym.size <= 0:
                continue
            sym_name = (
                sym.name.decode("utf-8", errors="replace")
                if isinstance(sym.name, bytes)
                else str(sym.name)
            )
            if function_filter and sym_name != function_filter:
                continue
            offset = int(sym.value) - text_va
            if offset < 0 or offset >= len(text_bytes):
                continue
            func_bytes = text_bytes[offset : offset + int(sym.size)]
            blocks = self.lift_bytes(
                code=func_bytes,
                base_va=int(sym.value),
                function_name=sym_name,
                arch=arch,
            )
            if blocks:
                result[sym_name] = blocks
        return result

    def lift_bytes(
        self,
        code: bytes,
        base_va: int = 0x1000,
        function_name: str = "<anonymous>",
        arch: str = "x86_64",
    ) -> list[BinaryBasicBlock]:
        """Disassemble raw machine code bytes into normalized BinaryBasicBlocks (PT-01 safe)."""
        if not code:
            return []

        cs_arch, cs_mode = (
            (capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
            if arch == "aarch64"
            else (capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        )
        try:
            md = capstone.Cs(cs_arch, cs_mode)
            insns = list(islice(md.disasm(code[:MAX_DISASM_BYTES], base_va), MAX_DISASM_INSNS))
        except Exception:
            return []

        if not insns:
            return []

        leaders: set[int] = {insns[0].address}
        for idx, insn in enumerate(insns):
            if self._is_branch(insn.mnemonic, arch):
                target = self._parse_branch_target(insn.op_str)
                if target is not None:
                    leaders.add(target)
                if idx + 1 < len(insns):
                    leaders.add(insns[idx + 1].address)

        blocks: list[BinaryBasicBlock] = []
        curr_raw: list[str] = []
        curr_norm: list[str] = []
        curr_start = insns[0].address
        curr_end = insns[0].address

        for idx, insn in enumerate(insns):
            if idx > 0 and insn.address in leaders and curr_raw:
                blocks.append(
                    BinaryBasicBlock(
                        function_name=function_name,
                        arch=arch,
                        va_start=curr_start,
                        va_end=curr_end,
                        raw_instructions=curr_raw,
                        normalized_instructions=curr_norm,
                        successor_vas=[insn.address],
                    )
                )
                curr_raw = []
                curr_norm = []
                curr_start = insn.address

            curr_raw.append(f"{insn.mnemonic} {insn.op_str}".strip())
            curr_norm.append(normalize_instruction(insn.mnemonic, insn.op_str, arch=arch))
            curr_end = insn.address + insn.size

            if self._is_branch(insn.mnemonic, arch):
                succs: list[int] = []
                target = self._parse_branch_target(insn.op_str)
                if target is not None:
                    succs.append(target)
                if not self._is_unconditional_terminator(insn.mnemonic, arch) and idx + 1 < len(
                    insns
                ):
                    succs.append(insns[idx + 1].address)
                blocks.append(
                    BinaryBasicBlock(
                        function_name=function_name,
                        arch=arch,
                        va_start=curr_start,
                        va_end=curr_end,
                        raw_instructions=curr_raw,
                        normalized_instructions=curr_norm,
                        successor_vas=succs,
                    )
                )
                curr_raw = []
                curr_norm = []
                if idx + 1 < len(insns):
                    curr_start = insns[idx + 1].address

        if curr_raw:
            blocks.append(
                BinaryBasicBlock(
                    function_name=function_name,
                    arch=arch,
                    va_start=curr_start,
                    va_end=curr_end,
                    raw_instructions=curr_raw,
                    normalized_instructions=curr_norm,
                    successor_vas=[],
                )
            )
        return blocks

    @staticmethod
    def _detect_arch(binary: lief.ELF.Binary) -> str:
        machine = binary.header.machine_type
        if machine == lief.ELF.ARCH.AARCH64:
            return "aarch64"
        return "x86_64"

    @staticmethod
    def _is_branch(mnemonic: str, arch: str) -> bool:
        m = mnemonic.lower()
        if arch == "x86_64":
            return m in _X86_JUMP_MNEMONICS
        return m == "ret" or m.startswith(("b.", "cbz", "cbnz", "tbz", "tbnz")) or m == "b"

    @staticmethod
    def _is_unconditional_terminator(mnemonic: str, arch: str) -> bool:
        m = mnemonic.lower()
        if arch == "x86_64":
            return m in ("jmp", "ret", "retq")
        return m in ("b", "ret")

    @staticmethod
    def _parse_branch_target(op_str: str) -> int | None:
        m = re.fullmatch(r"#?(0x[0-9a-fA-F]+)", op_str.strip())
        if m:
            return int(m.group(1), 16)
        return None
