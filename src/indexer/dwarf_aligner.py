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
        """Link LLVM IR !dbg IDs to DWARFLineMapping records by file + source line."""
        scoped_dbg: dict[tuple[str, int], list[str]] = {}
        line_to_dbg: dict[int, list[str]] = {}
        for graph in ir_graphs:
            src_base = Path(graph.source_file).name if graph.source_file != "<memory>" else ""
            for dbg_id, line_no in graph.dbg_line_map.items():
                line_to_dbg.setdefault(line_no, [])
                if dbg_id not in line_to_dbg[line_no]:
                    line_to_dbg[line_no].append(dbg_id)
                if src_base:
                    scoped_dbg.setdefault((src_base, line_no), [])
                    if dbg_id not in scoped_dbg[(src_base, line_no)]:
                        scoped_dbg[(src_base, line_no)].append(dbg_id)

        for m in mappings:
            m_base = Path(m.file_path).name if m.file_path else ""
            matched = scoped_dbg.get((m_base, m.line_start)) if m_base else None
            if matched is None and (not m_base or not scoped_dbg):
                matched = line_to_dbg.get(m.line_start)
            if matched:
                m.dbg_ids = list(matched)
        return mappings

    def extract_subprogram_info(self, elf_path: str | Path) -> dict[str, tuple[str, int | None]]:
        """Map ELF symbol/linkage names to (unmangled_name, decl_line) from DWARF subprograms."""
        path = Path(elf_path)
        if not path.exists():
            return {}
        info: dict[str, tuple[str, int | None]] = {}
        with path.open("rb") as fh:
            elf = ELFFile(fh)
            if not elf.has_dwarf_info():
                return {}
            dwarf = elf.get_dwarf_info()
            for cu in dwarf.iter_CUs():
                for die in cu.iter_DIEs():
                    if die.tag != "DW_TAG_subprogram":
                        continue
                    name = self._die_str_attr(die, "DW_AT_name")
                    linkage = self._die_str_attr(die, "DW_AT_linkage_name") or self._die_str_attr(
                        die, "DW_AT_MIPS_linkage_name"
                    )
                    if not name and not linkage:
                        continue
                    decl_attr = die.attributes.get("DW_AT_decl_line")
                    decl_line = int(decl_attr.value) if decl_attr is not None else None
                    resolved_name = name or linkage
                    if linkage:
                        info[linkage] = (resolved_name, decl_line)
                    if name:
                        info.setdefault(name, (resolved_name, decl_line))
        return info

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
    def _die_str_attr(die: Any, attr_name: str) -> str:
        attr = die.attributes.get(attr_name)
        if attr is None:
            return ""
        val = attr.value
        return val.decode("utf-8", errors="replace") if isinstance(val, bytes) else str(val)

    @staticmethod
    def _extract_inlined_subroutines(dwarf: Any) -> list[tuple[int, int, str]]:
        ranges: list[tuple[int, int, str]] = []
        try:
            range_lists = dwarf.range_lists()
        except Exception:
            range_lists = None

        for cu in dwarf.iter_CUs():
            cu_base = 0
            try:
                top_die = cu.get_top_DIE()
                cu_low = top_die.attributes.get("DW_AT_low_pc")
                if cu_low is not None:
                    cu_base = int(cu_low.value)
            except Exception:
                pass

            for die in cu.iter_DIEs():
                if die.tag != "DW_TAG_inlined_subroutine":
                    continue
                origin_name = DWARFAligner._resolve_inlined_name(die)
                low_pc = die.attributes.get("DW_AT_low_pc")
                high_pc = die.attributes.get("DW_AT_high_pc")
                if low_pc is not None and high_pc is not None:
                    start = int(low_pc.value)
                    end = (
                        start + int(high_pc.value)
                        if high_pc.form.startswith("DW_FORM_data")
                        else int(high_pc.value)
                    )
                    ranges.append((start, end, origin_name))
                    continue

                ranges_attr = die.attributes.get("DW_AT_ranges")
                if ranges_attr is not None and range_lists is not None:
                    try:
                        rlist = range_lists.get_range_list_at_offset(int(ranges_attr.value), cu=cu)
                        base = cu_base
                        for entry in rlist:
                            if hasattr(entry, "base_address"):
                                base = int(entry.base_address)
                            elif hasattr(entry, "begin_offset") and hasattr(entry, "end_offset"):
                                ranges.append(
                                    (
                                        base + int(entry.begin_offset),
                                        base + int(entry.end_offset),
                                        origin_name,
                                    )
                                )
                    except Exception:
                        pass
        return ranges

    @staticmethod
    def _resolve_inlined_name(die: Any) -> str:
        name = DWARFAligner._die_str_attr(die, "DW_AT_name")
        if name:
            return name
        if "DW_AT_abstract_origin" in die.attributes:
            try:
                origin = die.get_DIE_from_attribute("DW_AT_abstract_origin")
                if origin is not None:
                    origin_name = DWARFAligner._die_str_attr(origin, "DW_AT_name")
                    if origin_name:
                        return origin_name
            except Exception:
                pass
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
