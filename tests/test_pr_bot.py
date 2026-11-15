"""Tests for the compiler-aware GitHub PR Review Bot (NFC proof, regressions, blast radius)."""

from __future__ import annotations

from pathlib import Path

from pr_bot.analyzer import analyze_pr_directories, analyze_repository, parse_unified_diff
from pr_bot.gemini_reviewer import build_review_prompt


def test_parse_unified_diff() -> None:
    diff = """diff --git a/src/clamp.c b/src/clamp.c
--- a/src/clamp.c
+++ b/src/clamp.c
@@ -1,4 +1,3 @@
"""
    hunks = parse_unified_diff(diff)
    assert hunks == {"src/clamp.c": [(1, 3)]}


def test_pr_bot_nfc_proof_and_blast_radius(tmp_path: Path) -> None:
    base = tmp_path / "base"
    head = tmp_path / "head"
    (base / "src").mkdir(parents=True)
    (head / "src").mkdir(parents=True)

    # Base: if-else clamp + caller in main.c
    (base / "src" / "clamp.c").write_text(
        "int clamp(int x) {\n    if (x > 0) return x;\n    return 0;\n}\n",
        encoding="utf-8",
    )
    (base / "src" / "main.c").write_text(
        "int clamp(int x);\nint run(int v) { return clamp(v); }\n",
        encoding="utf-8",
    )

    # Head: refactored to ternary clamp (NFC) + main.c unchanged
    (head / "src" / "clamp.c").write_text(
        "int clamp(int x) {\n    return (x > 0) ? x : 0;\n}\n",
        encoding="utf-8",
    )
    (head / "src" / "main.c").write_text(
        "int clamp(int x);\nint run(int v) { return clamp(v); }\n",
        encoding="utf-8",
    )

    diff = """diff --git a/src/clamp.c b/src/clamp.c
--- a/src/clamp.c
+++ b/src/clamp.c
@@ -1,4 +1,3 @@
"""
    report = analyze_pr_directories(base, head, diff, base_ref="main", head_ref="feat/ternary")
    assert len(report.changed_functions) == 1
    fn = report.changed_functions[0]
    assert fn.symbol_name == "clamp"
    assert fn.nfc_verified is True
    assert fn.ir_similarity is not None and fn.ir_similarity >= 0.99
    assert len(fn.blast_radius_callers) >= 1

    md = report.to_markdown()
    assert "NFC Verified" in md
    assert "Blast Radius Graph" in md
    assert "mermaid" in md


def test_pr_bot_lowering_regression_and_inline_comments(tmp_path: Path) -> None:
    base = tmp_path / "base"
    head = tmp_path / "head"
    base.mkdir()
    head.mkdir()

    (base / "calc.c").write_text(
        "int compute(int x) {\n    return x * 4;\n}\n",
        encoding="utf-8",
    )
    # Head adds a loop + external call, preventing branchless lowering
    (head / "calc.c").write_text(
        "int ext_fn(int v);\n"
        "int compute(int x) {\n"
        "    int acc = 0;\n"
        "    for (int i = 0; i < x; i++) {\n"
        "        if (i % 2 == 0) acc += ext_fn(i);\n"
        "        else acc -= i;\n"
        "    }\n"
        "    if (acc < 0) return 0;\n"
        "    return acc;\n"
        "}\n",
        encoding="utf-8",
    )

    diff = """diff --git a/calc.c b/calc.c
--- a/calc.c
+++ b/calc.c
@@ -1,3 +1,10 @@
"""
    report = analyze_pr_directories(base, head, diff)
    categories = {f.category for f in report.findings}
    assert "COMPLEXITY_SPIKE" in categories
    assert "LOWERING_REGRESSION" in categories

    comments = report.to_github_review_comments()
    assert len(comments) >= 1
    assert comments[0]["path"] == "calc.c"
    assert comments[0]["side"] == "RIGHT"

    prompt = build_review_prompt(report, diff)
    assert "4-Layer Deterministic Report" in prompt
    assert "LOWERING_REGRESSION" in prompt
    assert "Unified Git Diff" in prompt


def test_periodic_repository_audit(tmp_path: Path) -> None:
    (tmp_path / "a.c").write_text(
        "int clamp_a(int x) {\n    if (x > 0) return x;\n    return 0;\n}\n",
        encoding="utf-8",
    )
    (tmp_path / "b.c").write_text(
        "int clamp_b(int y) {\n    return (y > 0) ? y : 0;\n}\n",
        encoding="utf-8",
    )
    repo_report = analyze_repository(tmp_path, clone_threshold=0.92)
    assert repo_report.indexed_units == 2
    assert len(repo_report.clone_pairs) >= 1
    md = repo_report.to_markdown()
    assert "Periodic 4-Layer Repository Review" in md
    assert "Semantic / IR Clone Pairs" in md
