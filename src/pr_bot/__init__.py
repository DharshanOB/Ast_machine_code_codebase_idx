"""Compiler-aware GitHub PR review bot powered by AST, CPG, LLVM IR, and Binary CFG."""

from pr_bot.analyzer import (
    FunctionDelta,
    PRReviewReport,
    RepoReviewReport,
    ReviewFinding,
    analyze_git_pr,
    analyze_pr_directories,
    analyze_repository,
    parse_unified_diff,
)

__all__ = [
    "FunctionDelta",
    "PRReviewReport",
    "RepoReviewReport",
    "ReviewFinding",
    "analyze_git_pr",
    "analyze_pr_directories",
    "analyze_repository",
    "parse_unified_diff",
]
