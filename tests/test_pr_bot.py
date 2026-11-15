"""Tests for the compiler-aware GitHub PR Review Bot (NFC proof, regressions, blast radius)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from indexer.llvm_ir_pass import resolve_clang_binary
from pr_bot.analyzer import (
    analyze_git_pr,
    analyze_pr_directories,
    analyze_repository,
    parse_unified_diff,
)
from pr_bot.cli import main as cli_main
from pr_bot.gemini_reviewer import build_review_prompt


def test_parse_unified_diff() -> None:
    diff = """diff --git a/src/clamp.c b/src/clamp.c
--- a/src/clamp.c
+++ b/src/clamp.c
@@ -1,4 +1,3 @@
"""
    hunks = parse_unified_diff(diff)
    assert hunks == {"src/clamp.c": [(1, 3)]}


@pytest.mark.skipif(resolve_clang_binary() is None, reason="clang-18/clang not installed")
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


@pytest.mark.skipif(resolve_clang_binary() is None, reason="clang-18/clang not installed")
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
    assert "Deterministic Analysis Report" in prompt
    assert "LOWERING_REGRESSION" in prompt
    assert "Unified Git Diff" in prompt


@pytest.mark.skipif(resolve_clang_binary() is None, reason="clang-18/clang not installed")
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
    assert len(repo_report.clone_clusters) >= 1
    md = repo_report.to_markdown()
    assert "Periodic 4-Layer Repository Review" in md
    assert "Semantic / IR Clone Clusters" in md


def test_python_bytecode_and_javascript_indexing(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(
        "def compute(x: int) -> int:\n"
        "    if x > 0:\n"
        "        return x * 2\n"
        "    return 0\n",
        encoding="utf-8",
    )
    (tmp_path / "helper.js").write_text(
        "class Helper {\n"
        "  greet(name) {\n"
        "    return name ? `hi ${name}` : 'hi';\n"
        "  }\n"
        "}\n"
        "function add(a, b) {\n"
        "  return a + b;\n"
        "}\n",
        encoding="utf-8",
    )
    repo_report = analyze_repository(tmp_path)
    assert repo_report.indexed_units >= 4
    assert repo_report.degraded_units == 0


def test_pr_uses_merge_base_not_branch_tip(tmp_path: Path) -> None:
    """PR analysis must diff against merge-base so unrelated main commits are not flagged."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git = lambda *args: subprocess.run(  # noqa: E731
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    git("init", "-b", "main")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test User")

    (repo / "mod.py").write_text(
        "def f(x: int) -> int:\n    return x + 1\n\n"
        "def g(y: int) -> int:\n    return y * 2\n",
        encoding="utf-8",
    )
    git("add", "mod.py")
    git("commit", "-m", "initial")

    # Feature branch modifies only f
    git("checkout", "-b", "feat")
    (repo / "mod.py").write_text(
        "def f(x: int) -> int:\n    return x + 10\n\n"
        "def g(y: int) -> int:\n    return y * 2\n",
        encoding="utf-8",
    )
    git("commit", "-am", "feat modifies f")

    # Main advances independently by modifying g
    git("checkout", "main")
    (repo / "mod.py").write_text(
        "def f(x: int) -> int:\n    return x + 1\n\n"
        "def g(y: int) -> int:\n    return y * 3 + 1\n",
        encoding="utf-8",
    )
    git("commit", "-am", "main modifies g")

    report = analyze_git_pr(repo, base_ref="main", head_ref="feat")
    changed_names = {fn.symbol_name for fn in report.changed_functions}
    assert changed_names == {"f"}


def test_cli_repo_mode_and_corrupt_baseline(tmp_path: Path) -> None:
    """Verify CLI repo mode writes summary, baseline, and handles corrupt baseline JSON."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def alpha(x: int) -> int:\n    return x + 1\n", encoding="utf-8")

    out_md = tmp_path / "summary.md"
    out_comments = tmp_path / "comments.json"
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{not valid json", encoding="utf-8")

    rc = cli_main(
        [
            "--mode",
            "repo",
            "--repo",
            str(repo),
            "--baseline",
            str(baseline),
            "--out-baseline",
            str(baseline),
            "--out-md",
            str(out_md),
            "--out-comments",
            str(out_comments),
        ]
    )
    assert rc == 0
    assert "Baseline Warning" in out_md.read_text(encoding="utf-8")
    assert '"hotspots"' in baseline.read_text(encoding="utf-8")
