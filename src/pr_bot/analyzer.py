"""Compiler-aware GitHub PR diff analyzer (AST + CPG + LLVM IR + Machine Code)."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from indexer.models import ASTChunkNode, CPGEdgeKind, IRStatus, UnifiedCodeUnit
from mcp_server.code_index_server import CodeIndexPipeline
from storage.hybrid_store import compute_wl_ir_vector, cosine_similarity

_DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$")
_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_CONTAINER_NODE_TYPES: frozenset[str] = frozenset(
    {
        "class_definition",
        "class_specifier",
        "struct_specifier",
        "impl_item",
        "struct_item",
        "trait_item",
        "enum_specifier",
    }
)


def _is_nested_chunk(a: ASTChunkNode, b: ASTChunkNode) -> bool:
    """Return True if chunk `a` and `b` are in the same file and one encloses the other."""
    if a.file_path != b.file_path:
        return False
    a_in_b = b.line_start <= a.line_start and a.line_end <= b.line_end
    b_in_a = a.line_start <= b.line_start and b.line_end <= a.line_end
    return a_in_b or b_in_a


class ReviewFinding(BaseModel):
    """Actionable PR finding anchored to a file and line."""

    severity: str = Field(description="HIGH, MEDIUM, or INFO")
    category: str = Field(
        description="BLAST_RADIUS, CLONE, LOWERING_REGRESSION, COMPLEXITY_SPIKE, or NFC_VERIFIED"
    )
    file_path: str
    line: int = Field(ge=1)
    symbol_name: str
    message: str


class FunctionDelta(BaseModel):
    """Cross-layer comparison of a modified function between base and head."""

    file_path: str
    symbol_name: str
    line_start: int
    line_end: int
    complexity_before: int | None = None
    complexity_after: int
    ir_similarity: float | None = None
    binary_blocks_before: int | None = None
    binary_blocks_after: int = 0
    nfc_verified: bool = False
    blast_radius_callers: list[str] = Field(default_factory=list)
    semantic_clones: list[str] = Field(default_factory=list)


class PRReviewReport(BaseModel):
    """Complete multi-layer PR review report with Markdown and GitHub inline comments."""

    base_ref: str = "base"
    head_ref: str = "head"
    changed_functions: list[FunctionDelta] = Field(default_factory=list)
    findings: list[ReviewFinding] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render a rich GitHub PR summary comment with compiler & CPG evidence."""
        lines: list[str] = [
            "## Antigravity 4-Layer PR Review (AST -> CPG -> LLVM IR -> Machine Code)",
            "",
            f"**Base:** `{self.base_ref}` | **Head:** `{self.head_ref}` | "
            f"**Modified Functions:** {len(self.changed_functions)} | "
            f"**Findings:** {len(self.findings)}",
            "",
        ]
        if self.changed_functions:
            lines.extend(
                [
                    "### Modified Functions & Lowering Delta",
                    "",
                    "| Function | File | Complexity (Δ) | IR Sim | BBs | Callers | Status |",
                    "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
                ]
            )
            for fn in self.changed_functions:
                c_before = fn.complexity_before if fn.complexity_before is not None else "-"
                c_delta = (
                    f"{c_before} -> {fn.complexity_after}"
                    if fn.complexity_before is not None
                    else f"new ({fn.complexity_after})"
                )
                ir_sim = f"{fn.ir_similarity:.3f}" if fn.ir_similarity is not None else "n/a"
                bb_str = (
                    f"{fn.binary_blocks_before} -> {fn.binary_blocks_after}"
                    if fn.binary_blocks_before is not None
                    else str(fn.binary_blocks_after)
                )
                status = "NFC Verified" if fn.nfc_verified else "Modified"
                lines.append(
                    f"| `{fn.symbol_name}` | `{fn.file_path}` | {c_delta} | {ir_sim} | "
                    f"{bb_str} | {len(fn.blast_radius_callers)} callers | {status} |"
                )
            lines.append("")

        if self.findings:
            lines.extend(["### Findings", ""])
            for f in self.findings:
                lines.append(
                    f"- **[{f.severity}] {f.category}** (`{f.file_path}:{f.line}` "
                    f"in `{f.symbol_name}`): {f.message}"
                )
            lines.append("")

        callers_edges = [
            (fn.symbol_name, caller)
            for fn in self.changed_functions
            for caller in fn.blast_radius_callers
        ]
        if callers_edges:
            lines.extend(["### Blast Radius Graph", "", "```mermaid", "flowchart LR"])
            for sym, caller in callers_edges[:20]:
                lines.append(f'    "{caller}" -->|CALLS| "{sym}"')
            lines.extend(["```", ""])

        return "\n".join(lines)

    def to_github_review_comments(self) -> list[dict[str, Any]]:
        """Format actionable findings as GitHub PR inline review comment payloads."""
        return [
            {
                "path": f.file_path,
                "line": f.line,
                "side": "RIGHT",
                "body": f"**[{f.severity}] {f.category}** (`{f.symbol_name}`)\n\n{f.message}",
            }
            for f in self.findings
            if f.severity in ("HIGH", "MEDIUM")
        ]


class RepoReviewReport(BaseModel):
    """Full-repository periodic health, complexity, clone, and blast-radius report."""

    repo_path: str
    indexed_units: int
    degraded_units: int
    hotspots: list[dict[str, Any]] = Field(default_factory=list)
    clone_pairs: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[ReviewFinding] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render a periodic repository health review in Markdown."""
        lines: list[str] = [
            "## Antigravity Periodic 4-Layer Repository Review",
            "",
            f"**Indexed Units:** {self.indexed_units} | "
            f"**Degraded Units:** {self.degraded_units} | "
            f"**Findings:** {len(self.findings)}",
            "",
        ]
        if self.hotspots:
            lines.extend(
                [
                    "### Complexity & Blast-Radius Hotspots",
                    "",
                    "| Function | File | Complexity | Callers |",
                    "| :--- | :--- | :---: | :---: |",
                ]
            )
            for h in self.hotspots:
                lines.append(
                    f"| `{h['symbol_name']}` | `{h['file_path']}` | "
                    f"{h['complexity']} | {h['callers']} |"
                )
            lines.append("")

        if self.clone_pairs:
            lines.extend(["### Semantic / IR Clone Pairs", ""])
            for c in self.clone_pairs:
                lines.append(
                    f"- `{c['source']}` ↔ `{c['target']}` (similarity={c['similarity']:.3f})"
                )
            lines.append("")

        if self.findings:
            lines.extend(["### Findings", ""])
            for f in self.findings:
                lines.append(
                    f"- **[{f.severity}] {f.category}** (`{f.file_path}:{f.line}` "
                    f"in `{f.symbol_name}`): {f.message}"
                )
            lines.append("")
        return "\n".join(lines)


def parse_unified_diff(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Parse unified diff text into {relative_file_path: [(start_line, end_line), ...]}."""
    changed: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None
    for raw_line in diff_text.splitlines():
        file_match = _DIFF_FILE_RE.match(raw_line)
        if file_match:
            current_file = file_match.group(1)
            changed.setdefault(current_file, [])
            continue
        if current_file is None:
            continue
        hunk_match = _DIFF_HUNK_RE.match(raw_line)
        if hunk_match:
            start = int(hunk_match.group(1))
            count = int(hunk_match.group(2) or "1")
            end = start + max(count, 1) - 1
            changed[current_file].append((start, end))
    return changed


def _rel_path(full_path: str, root: Path) -> str:
    try:
        return str(Path(full_path).resolve().relative_to(root.resolve()))
    except ValueError:
        return full_path


def _index_units_by_key(
    pipeline: CodeIndexPipeline, root: Path
) -> dict[tuple[str, str], UnifiedCodeUnit]:
    mapping: dict[tuple[str, str], UnifiedCodeUnit] = {}
    for unit in pipeline.store.units.values():
        rel = _rel_path(unit.ast_chunk.file_path, root)
        if unit.ast_chunk.symbol_name:
            mapping[(rel, unit.ast_chunk.symbol_name)] = unit
    return mapping


def analyze_pr_directories(
    base_dir: Path,
    head_dir: Path,
    diff_text: str,
    base_ref: str = "base",
    head_ref: str = "head",
    compile_commands_path: Path | None = None,
) -> PRReviewReport:
    """Compare base_dir vs head_dir across all 4 layers for files/lines touched in diff_text."""
    changed_hunks = parse_unified_diff(diff_text)

    base_pipe = CodeIndexPipeline()
    base_pipe.index_workspace(base_dir, compile_commands_path=compile_commands_path)
    base_units = _index_units_by_key(base_pipe, base_dir)

    head_pipe = CodeIndexPipeline()
    head_pipe.index_workspace(head_dir, compile_commands_path=compile_commands_path)
    head_units = _index_units_by_key(head_pipe, head_dir)

    deltas: list[FunctionDelta] = []
    findings: list[ReviewFinding] = []

    for (rel_file, sym_name), head_unit in sorted(head_units.items()):
        hunks = changed_hunks.get(rel_file)
        if hunks is None:
            continue
        ast = head_unit.ast_chunk
        if not any(h_end >= ast.line_start and h_start <= ast.line_end for h_start, h_end in hunks):
            continue

        base_unit = base_units.get((rel_file, sym_name))
        c_before = base_unit.ast_chunk.cyclomatic_complexity if base_unit else None
        c_after = ast.cyclomatic_complexity

        ir_sim: float | None = None
        nfc_verified = False
        if (
            base_unit is not None
            and base_unit.ir_status == IRStatus.OK
            and head_unit.ir_status == IRStatus.OK
            and base_unit.ir_graph is not None
            and head_unit.ir_graph is not None
        ):
            v_base = compute_wl_ir_vector(base_unit.ir_graph)
            v_head = compute_wl_ir_vector(head_unit.ir_graph)
            ir_sim = round(cosine_similarity(v_base, v_head), 4)
            if ir_sim >= 0.99 and (
                base_unit.ir_graph.canonical_tokens == head_unit.ir_graph.canonical_tokens
            ):
                nfc_verified = True

        bb_before = len(base_unit.binary_blocks) if base_unit else None
        bb_after = len(head_unit.binary_blocks)

        callers: list[str] = []
        if head_unit.scip_symbol is not None:
            slices = head_pipe.cpg.k_hop_slice(
                head_unit.scip_symbol.symbol_uri,
                max_hops=2,
                edge_kinds=frozenset({CPGEdgeKind.CALLS}),
                direction="backward",
            )
            callers = sorted({str(s["source"]) for s in slices})

        clones_raw = head_pipe.store.find_clones(
            head_unit.unit_id, modality="ir_structural", threshold=0.92, top_k=3
        )
        if not clones_raw:
            clones_raw = head_pipe.store.find_clones(
                head_unit.unit_id, modality="ast_dense", threshold=0.92, top_k=3
            )
        clone_names = [
            f"{head_pipe.store.units[cid].ast_chunk.symbol_name} ({score:.2f})"
            for cid, score in clones_raw
            if cid in head_pipe.store.units
            and head_pipe.store.units[cid].ast_chunk.symbol_name != sym_name
        ]

        deltas.append(
            FunctionDelta(
                file_path=rel_file,
                symbol_name=sym_name,
                line_start=ast.line_start,
                line_end=ast.line_end,
                complexity_before=c_before,
                complexity_after=c_after,
                ir_similarity=ir_sim,
                binary_blocks_before=bb_before,
                binary_blocks_after=bb_after,
                nfc_verified=nfc_verified,
                blast_radius_callers=callers,
                semantic_clones=clone_names,
            )
        )

        if nfc_verified:
            findings.append(
                ReviewFinding(
                    severity="INFO",
                    category="NFC_VERIFIED",
                    file_path=rel_file,
                    line=ast.line_start,
                    symbol_name=sym_name,
                    message=(
                        f"LLVM IR equivalence verified (similarity={ir_sim:.4f}): "
                        "refactor produces identical lowered SSA IR."
                    ),
                )
            )
        if c_before is not None and c_after - c_before >= 3:
            findings.append(
                ReviewFinding(
                    severity="MEDIUM",
                    category="COMPLEXITY_SPIKE",
                    file_path=rel_file,
                    line=ast.line_start,
                    symbol_name=sym_name,
                    message=f"Cyclomatic complexity jumped from {c_before} to {c_after}.",
                )
            )
        if bb_before is not None and bb_after > bb_before:
            findings.append(
                ReviewFinding(
                    severity="MEDIUM",
                    category="LOWERING_REGRESSION",
                    file_path=rel_file,
                    line=ast.line_start,
                    symbol_name=sym_name,
                    message=(
                        f"Compiled machine-code basic blocks increased from {bb_before} "
                        f"to {bb_after} (possible lost branchless lowering)."
                    ),
                )
            )
        if callers:
            findings.append(
                ReviewFinding(
                    severity="HIGH" if len(callers) >= 3 else "INFO",
                    category="BLAST_RADIUS",
                    file_path=rel_file,
                    line=ast.line_start,
                    symbol_name=sym_name,
                    message=f"Modified function has {len(callers)} downstream caller(s) in CPG.",
                )
            )
        if clone_names:
            findings.append(
                ReviewFinding(
                    severity="MEDIUM",
                    category="CLONE",
                    file_path=rel_file,
                    line=ast.line_start,
                    symbol_name=sym_name,
                    message=f"Potential semantic clone(s) detected: {', '.join(clone_names)}.",
                )
            )

    return PRReviewReport(
        base_ref=base_ref,
        head_ref=head_ref,
        changed_functions=deltas,
        findings=findings,
    )


def analyze_git_pr(
    repo_path: str | Path,
    base_ref: str = "HEAD~1",
    head_ref: str = "HEAD",
    compile_commands_path: str | Path | None = None,
) -> PRReviewReport:
    """Run base-vs-head 4-layer PR analysis using isolated git worktrees."""
    repo = Path(repo_path).resolve()
    diff_proc = subprocess.run(
        ["git", "-C", str(repo), "diff", "-U0", base_ref, head_ref],
        shell=False,
        capture_output=True,
        text=True,
        check=True,
    )
    cc_path = Path(compile_commands_path) if compile_commands_path else None

    with tempfile.TemporaryDirectory(prefix="ag-pr-") as tmp_root:
        base_wt = Path(tmp_root) / "base"
        head_wt = Path(tmp_root) / "head"
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(base_wt), base_ref],
            shell=False,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(head_wt), head_ref],
            shell=False,
            capture_output=True,
            check=True,
        )
        try:
            return analyze_pr_directories(
                base_dir=base_wt,
                head_dir=head_wt,
                diff_text=diff_proc.stdout,
                base_ref=base_ref,
                head_ref=head_ref,
                compile_commands_path=cc_path,
            )
        finally:
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force", str(base_wt)],
                shell=False,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force", str(head_wt)],
                shell=False,
                capture_output=True,
            )


def analyze_repository(
    repo_path: str | Path,
    compile_commands_path: str | Path | None = None,
    complexity_threshold: int = 8,
    clone_threshold: float = 0.94,
) -> RepoReviewReport:
    """Run a full-repository 4-layer audit for periodic or manual reviews."""
    root = Path(repo_path).resolve()
    pipe = CodeIndexPipeline()
    summary = pipe.index_workspace(root, compile_commands_path=compile_commands_path)

    hotspots: list[dict[str, Any]] = []
    clone_pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    findings: list[ReviewFinding] = []

    for unit in pipe.store.units.values():
        ast = unit.ast_chunk
        rel = _rel_path(ast.file_path, root)
        sym_name = ast.symbol_name or "<anonymous>"
        is_container = ast.node_kind in _CONTAINER_NODE_TYPES
        callers = 0
        if unit.scip_symbol is not None:
            slices = pipe.cpg.k_hop_slice(
                unit.scip_symbol.symbol_uri,
                max_hops=2,
                edge_kinds=frozenset({CPGEdgeKind.CALLS}),
                direction="backward",
            )
            callers = len({str(s["source"]) for s in slices})

        if not is_container and (ast.cyclomatic_complexity >= complexity_threshold or callers >= 3):
            hotspots.append(
                {
                    "symbol_name": sym_name,
                    "file_path": rel,
                    "complexity": ast.cyclomatic_complexity,
                    "callers": callers,
                }
            )
        if not is_container and ast.cyclomatic_complexity >= complexity_threshold:
            findings.append(
                ReviewFinding(
                    severity="MEDIUM",
                    category="COMPLEXITY_SPIKE",
                    file_path=rel,
                    line=ast.line_start,
                    symbol_name=sym_name,
                    message=f"High cyclomatic complexity ({ast.cyclomatic_complexity}).",
                )
            )
        if ast.has_syntax_error:
            findings.append(
                ReviewFinding(
                    severity="HIGH",
                    category="SYNTAX_ERROR",
                    file_path=rel,
                    line=ast.line_start,
                    symbol_name=sym_name,
                    message="Tree-sitter detected syntax errors in this code unit.",
                )
            )

        modality = "ir_structural" if unit.ir_status == IRStatus.OK else "ast_dense"
        for cid, score in pipe.store.find_clones(
            unit.unit_id, modality=modality, threshold=clone_threshold, top_k=2
        ):
            other = pipe.store.units.get(cid)
            if other is None or _is_nested_chunk(ast, other.ast_chunk):
                continue
            other_name = other.ast_chunk.symbol_name or "<anonymous>"
            if other_name == sym_name:
                continue
            pair = (min(sym_name, other_name), max(sym_name, other_name))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                clone_pairs.append(
                    {"source": sym_name, "target": other_name, "similarity": score}
                )
                findings.append(
                    ReviewFinding(
                        severity="INFO",
                        category="CLONE",
                        file_path=rel,
                        line=ast.line_start,
                        symbol_name=sym_name,
                        message=f"Semantic clone of `{other_name}` (similarity={score:.3f}).",
                    )
                )

    hotspots.sort(key=lambda h: (int(h["complexity"]), int(h["callers"])), reverse=True)
    return RepoReviewReport(
        repo_path=str(root),
        indexed_units=int(summary.get("indexed_units", 0)),
        degraded_units=int(summary.get("degraded_units", 0)),
        hotspots=hotspots[:20],
        clone_pairs=clone_pairs[:20],
        findings=findings,
    )
