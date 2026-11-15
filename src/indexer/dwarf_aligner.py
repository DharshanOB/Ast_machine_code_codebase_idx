"""Cross-layer DWARF debug metadata aligner (.debug_line & inlined subroutines)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from elftools.elf.elffile import ELFFile

from indexer.models import DWARFLineMapping, LLVMIRGraph


class DWARFAligner:
    """Parses DWARF .debug_line and .debug_info to align source lines, IR !dbg, and binary VAs."""

    def extract_mappings(
        self,
        elf_path: str | Path,
        default_file: str = "",
        function_entry_lines: dict[tuple[int, int], int] | None = None,
    ) -> list[DWARFLineMapping]:
        """Extract DWARFLineMapping records from an ELF object or binary."""
        path = Path(elf_path)
        if not path.exists():
            return []

        mappings: list[DWARFLineMapping] = []
        with path.open("rb") as fh:
            elf = ELFFile(fh)
            if not elf.has_dwarf_info():
                return []
            dwarf = elf.get_dwarf_info()
            inlined_ranges = self._extract_inlined_subroutines(dwarf)

            for cu in dwarf.iter_CUs():
                line_prog = dwarf.line_program_for_CU(cu)
                if line_prog is None:
                    continue
                file_entries = line_prog["file_entry"]
                entries = line_prog.get_entries()
                prev_state: Any = None

                for entry in entries:
                    state = entry.state
                    if state is None:
                        continue
                    if prev_state is not None and prev_state.address < state.address:
                        file_name = self._resolve_file_name(
                            file_entries, prev_state.file, default_file
                        )
                        line_no = int(prev_state.line)
                        col_no = int(prev_state.column)
                        is_prologue = line_no == 0
                        if is_prologue and function_entry_lines:
                            line_no = self._lookup_entry_line(
                                prev_state.address, function_entry_lines
                            )
                        inlined = self._lookup_inlined(prev_state.address, inlined_ranges)
                        mappings.append(
                            DWARFLineMapping(
                                file_path=file_name,
                                line_start=line_no,
                                line_end=line_no,
                                column=col_no,
                                va_start=int(prev_state.address),
                                va_end=int(state.address),
                                inlined_from=inlined,
                                is_prologue_or_epilogue=is_prologue,
                            )
                        )
                    prev_state = None if state.end_sequence else state

        return mappings

    def attach_ir_dbg_ids(
        self, mappings: list[DWARFLineMapping], ir_graphs: list[LLVMIRGraph]
    ) -> list[DWARFLineMapping]:
        """Link LLVM IR !dbg IDs to DWARFLineMapping records by matching source line numbers."""
        line_to_dbg: dict[int, list[str]] = {}
        for graph in ir_graphs:
            for dbg_id, line_no in graph.dbg_line_map.items():
                line_to_dbg.setdefault(line_no, [])
                if dbg_id not in line_to_dbg[line_no]:
                    line_to_dbg[line_no].append(dbg_id)

        for m in mappings:
            if m.line_start in line_to_dbg:
                m.dbg_ids = list(line_to_dbg[m.line_start])
        return mappings

    def lookup_va_for_line(
        self, mappings: list[DWARFLineMapping], line_start: int, line_end: int | None = None
    ) -> list[DWARFLineMapping]:
        """Return all DWARF mappings overlapping [line_start, line_end]."""
        end = line_start if line_end is None else line_end
        return [m for m in mappings if m.line_end >= line_start and m.line_start <= end]

    def lookup_line_for_va(
        self, mappings: list[DWARFLineMapping], va: int
    ) -> DWARFLineMapping | None:
        """Return the DWARF mapping covering virtual address `va`."""
        for m in mappings:
            if m.va_start <= va < m.va_end:
                return m
        return None

    @staticmethod
    def _resolve_file_name(file_entries: list[Any], file_index: int, default_file: str) -> str:
        if 1 <= file_index <= len(file_entries):
            raw_name = file_entries[file_index - 1].name
            if isinstance(raw_name, bytes):
                return raw_name.decode("utf-8", errors="replace")
            return str(raw_name)
        return default_file

    @staticmethod
    def _lookup_entry_line(va: int, function_entry_lines: dict[tuple[int, int], int]) -> int:
        for (start_va, end_va), entry_line in function_entry_lines.items():
            if start_va <= va < end_va:
                return entry_line
        return 0

    @staticmethod
    def _extract_inlined_subroutines(dwarf: Any) -> list[tuple[int, int, str]]:
        ranges: list[tuple[int, int, str]] = []
        for cu in dwarf.iter_CUs():
            for die in cu.iter_DIEs():
                if die.tag != "DW_TAG_inlined_subroutine":
                    continue
                low_pc = die.attributes.get("DW_AT_low_pc")
                high_pc = die.attributes.get("DW_AT_high_pc")
                if low_pc is None or high_pc is None:
                    continue
                start = int(low_pc.value)
                end = (
                    start + int(high_pc.value)
                    if high_pc.form.startswith("DW_FORM_data")
                    else int(high_pc.value)
                )
                origin_name = DWARFAligner._resolve_inlined_name(die)
                ranges.append((start, end, origin_name))
        return ranges

    @staticmethod
    def _resolve_inlined_name(die: Any) -> str:
        name_attr = die.attributes.get("DW_AT_name")
        if name_attr is not None:
            val = name_attr.value
            return val.decode("utf-8", errors="replace") if isinstance(val, bytes) else str(val)
        call_line = die.attributes.get("DW_AT_call_line")
        if call_line is not None:
            return f"inlined@line:{call_line.value}"
        return "inlined_subroutine"

    @staticmethod
    def _lookup_inlined(va: int, inlined_ranges: list[tuple[int, int, str]]) -> str | None:
        for start, end, name in inlined_ranges:
            if start <= va < end:
                return name
        return None
