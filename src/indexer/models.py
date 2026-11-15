"""Core Pydantic v2 data models and deterministic content-addressable IDs."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Language(StrEnum):
    """Supported source languages for syntactic and semantic indexing."""

    C = "c"
    CPP = "cpp"
    RUST = "rust"
    PYTHON = "python"
    JAVASCRIPT = "javascript"


class IRStatus(StrEnum):
    """Status of compiler IR and binary lowering for a code unit."""

    OK = "OK"
    DEGRADED = "DEGRADED"
    SKIPPED = "SKIPPED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CPGEdgeKind(StrEnum):
    """Cross-file and intraprocedural Code Property Graph edge types."""

    CALLS = "CALLS"
    IMPLEMENTS = "IMPLEMENTS"
    REACHES = "REACHES"
    CFG_NEXT = "CFG_NEXT"
    CONTAINS = "CONTAINS"


def compute_content_sha256(*parts: Any) -> str:
    """Compute a deterministic SHA-256 hex digest over normalized JSON-serializable parts."""
    normalized = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class ASTChunkNode(BaseModel):
    """Layer 1 structure-aware cAST chunk with scope context and complexity metrics."""

    chunk_id: str = Field(default="", description="Deterministic SHA-256 content ID.")
    file_path: str
    language: Language
    node_kind: str
    symbol_name: str = ""
    byte_start: int = Field(ge=0)
    byte_end: int = Field(ge=0)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    scope_header: str = ""
    docstring: str = ""
    source_text: str
    cyclomatic_complexity: int = Field(default=1, ge=1)
    lexical_tokens: list[str] = Field(default_factory=list)
    has_syntax_error: bool = False

    def model_post_init(self, __context: Any) -> None:
        if not self.chunk_id:
            self.chunk_id = compute_content_sha256(
                "ast",
                self.language.value,
                self.node_kind,
                self.scope_header,
                self.source_text,
            )


class SCIPEdge(BaseModel):
    """Directed semantic edge in the SCIP / Code Property Graph."""

    source_id: str
    target_id: str
    kind: CPGEdgeKind
    line: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class SCIPSymbolNode(BaseModel):
    """Layer 2 canonical SCIP symbol node and its local/cross-file edges."""

    symbol_uri: str = Field(
        description="Canonical URI: <scheme>:<package>/<version>/<descriptor_chain>"
    )
    display_name: str
    kind: str
    file_path: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    chunk_id: str = ""
    documentation: str = ""
    edges: list[SCIPEdge] = Field(default_factory=list)


class ProGraMLNode(BaseModel):
    """Node in a ProGraML-style LLVM SSA IR multigraph."""

    node_id: int = Field(ge=0)
    node_type: str = Field(description="One of: inst, val, type")
    text: str
    opcode: str = ""
    dbg_id: str | None = None


class ProGraMLEdge(BaseModel):
    """Typed, position-encoded edge in a ProGraML LLVM SSA IR multigraph."""

    source: int = Field(ge=0)
    target: int = Field(ge=0)
    flow_type: str = Field(description="One of: ctrl, data, call")
    position: int = Field(default=0, ge=0)


class LLVMIRGraph(BaseModel):
    """Layer 3 LLVM SSA IR representation in ProGraML graph topology."""

    ir_id: str = ""
    function_name: str
    source_file: str
    nodes: list[ProGraMLNode] = Field(default_factory=list)
    edges: list[ProGraMLEdge] = Field(default_factory=list)
    canonical_tokens: list[str] = Field(default_factory=list)
    dbg_line_map: dict[str, int] = Field(default_factory=dict)
    truncated: bool = False
    raw_ir: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.ir_id:
            self.ir_id = compute_content_sha256(
                "llvm_ir",
                self.function_name,
                self.canonical_tokens,
                [(e.source, e.target, e.flow_type, e.position) for e in self.edges],
            )


class BinaryBasicBlock(BaseModel):
    """Layer 4 disassembled and normalized machine-code basic block."""

    block_id: str = ""
    function_name: str
    arch: str = "x86_64"
    va_start: int = Field(ge=0)
    va_end: int = Field(ge=0)
    raw_instructions: list[str] = Field(default_factory=list)
    normalized_instructions: list[str] = Field(default_factory=list)
    successor_vas: list[int] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.block_id:
            self.block_id = compute_content_sha256(
                "bin_bb",
                self.arch,
                self.normalized_instructions,
            )


class DWARFLineMapping(BaseModel):
    """Bidirectional alignment record linking source lines, LLVM !dbg, and binary VAs."""

    file_path: str
    line_start: int = Field(ge=0)
    line_end: int = Field(ge=0)
    column: int = Field(default=0, ge=0)
    va_start: int = Field(ge=0)
    va_end: int = Field(ge=0)
    dbg_ids: list[str] = Field(default_factory=list)
    inlined_from: str | None = None
    is_prologue_or_epilogue: bool = False


class UnifiedCodeUnit(BaseModel):
    """Cross-layer indexed code unit unifying AST, SCIP/CPG, LLVM IR, and Binary CFG."""

    unit_id: str = ""
    commit_sha: str = "working-tree"
    ast_chunk: ASTChunkNode
    scip_symbol: SCIPSymbolNode | None = None
    ir_graph: LLVMIRGraph | None = None
    binary_blocks: list[BinaryBasicBlock] = Field(default_factory=list)
    dwarf_mappings: list[DWARFLineMapping] = Field(default_factory=list)
    ir_status: IRStatus = IRStatus.DEGRADED

    def model_post_init(self, __context: Any) -> None:
        if not self.unit_id:
            self.unit_id = compute_content_sha256(
                "unit",
                self.ast_chunk.chunk_id,
                self.scip_symbol.symbol_uri if self.scip_symbol else "",
                self.ir_graph.ir_id if self.ir_graph else "",
                [b.block_id for b in self.binary_blocks],
            )
