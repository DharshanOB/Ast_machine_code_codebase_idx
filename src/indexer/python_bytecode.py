"""Layer 3/4 equivalent for Python: bytecode CFG and basic-block lowering via `dis`."""

from __future__ import annotations

import dis
import types
from pathlib import Path

from indexer.models import BinaryBasicBlock, LLVMIRGraph, ProGraMLEdge, ProGraMLNode

_MAX_PY_BYTES = 2_000_000


def _normalize_py_inst(inst: dis.Instruction) -> str:
    """Normalize a Python bytecode instruction for structural comparison."""
    op = inst.opname
    if op in ("LOAD_CONST", "RETURN_CONST"):
        return f"{op} <CONST>"
    if op in ("LOAD_FAST", "STORE_FAST", "DELETE_FAST", "LOAD_FAST_CHECK"):
        return f"{op} <VAR>"
    if op in ("LOAD_GLOBAL", "LOAD_ATTR", "STORE_ATTR", "LOAD_METHOD", "LOAD_SUPER_ATTR"):
        return f"{op} <SYM>"
    if "JUMP" in op or op == "FOR_ITER":
        return f"{op} <TARGET>"
    return op


def _build_code_cfg_and_blocks(
    code_obj: types.CodeType,
    source_file: str,
) -> tuple[LLVMIRGraph, list[BinaryBasicBlock]]:
    """Build a ProGraML-compatible CFG and basic blocks for a single Python code object."""
    insns = list(dis.get_instructions(code_obj))
    if not insns:
        graph = LLVMIRGraph(
            function_name=code_obj.co_name,
            source_file=source_file,
            canonical_tokens=[],
        )
        return graph, []

    nodes: list[ProGraMLNode] = []
    edges: list[ProGraMLEdge] = []
    canonical_tokens: list[str] = []
    offset_to_idx: dict[int, int] = {}

    for idx, inst in enumerate(insns):
        offset_to_idx[inst.offset] = idx
        norm = _normalize_py_inst(inst)
        canonical_tokens.append(inst.opname)
        nodes.append(
            ProGraMLNode(
                node_id=idx,
                node_type="inst",
                opcode=inst.opname,
                text=norm,
            )
        )

    for idx, inst in enumerate(insns):
        if idx + 1 < len(insns) and inst.opname not in ("RETURN_VALUE", "RETURN_CONST", "RERAISE"):
            if not inst.opname.startswith("JUMP_"):
                edges.append(ProGraMLEdge(source=idx, target=idx + 1, flow_type="ctrl"))
        if ("JUMP" in inst.opname or inst.opname == "FOR_ITER") and isinstance(inst.argval, int):
            target_idx = offset_to_idx.get(inst.argval)
            if target_idx is not None:
                edges.append(ProGraMLEdge(source=idx, target=target_idx, flow_type="ctrl"))

    graph = LLVMIRGraph(
        function_name=code_obj.co_name,
        source_file=source_file,
        nodes=nodes,
        edges=edges,
        canonical_tokens=canonical_tokens,
    )

    blocks: list[BinaryBasicBlock] = []
    curr_raw: list[str] = []
    curr_norm: list[str] = []
    curr_start = insns[0].offset
    curr_end = insns[0].offset

    for idx, inst in enumerate(insns):
        if idx > 0 and inst.is_jump_target and curr_raw:
            blocks.append(
                BinaryBasicBlock(
                    function_name=code_obj.co_name,
                    arch="python_bytecode",
                    va_start=curr_start,
                    va_end=curr_end,
                    raw_instructions=curr_raw,
                    normalized_instructions=curr_norm,
                )
            )
            curr_raw = []
            curr_norm = []
            curr_start = inst.offset

        raw_repr = f"{inst.opname} {inst.argrepr}".strip() if inst.argrepr else inst.opname
        curr_raw.append(raw_repr)
        curr_norm.append(_normalize_py_inst(inst))
        curr_end = inst.offset

    if curr_raw:
        blocks.append(
            BinaryBasicBlock(
                function_name=code_obj.co_name,
                arch="python_bytecode",
                va_start=curr_start,
                va_end=curr_end,
                raw_instructions=curr_raw,
                normalized_instructions=curr_norm,
            )
        )

    return graph, blocks


def lower_python_file(
    source_path: str | Path,
    source_text: str | None = None,
) -> tuple[dict[str, LLVMIRGraph], dict[str, list[BinaryBasicBlock]]]:
    """Compile a Python file and extract bytecode CFGs and basic blocks per function."""
    path_str = str(source_path)
    text = (
        Path(path_str).read_text(encoding="utf-8", errors="replace")
        if source_text is None
        else source_text
    )
    if len(text.encode("utf-8", errors="ignore")) > _MAX_PY_BYTES:
        return {}, {}

    try:
        root_code = compile(text, path_str, "exec")
    except (SyntaxError, ValueError, TypeError):
        return {}, {}

    ir_by_fn: dict[str, LLVMIRGraph] = {}
    blocks_by_fn: dict[str, list[BinaryBasicBlock]] = {}

    stack: list[types.CodeType] = [root_code]
    while stack:
        code_obj = stack.pop()
        for const in code_obj.co_consts:
            if isinstance(const, types.CodeType):
                stack.append(const)

        name = code_obj.co_name
        if name.startswith("<"):
            continue
        graph, blocks = _build_code_cfg_and_blocks(code_obj, path_str)
        if name not in ir_by_fn:
            ir_by_fn[name] = graph
            blocks_by_fn[name] = blocks

    return ir_by_fn, blocks_by_fn
