"""CLI entrypoint for the Antigravity 4-Layer PR & Periodic Repository Review Bot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from pr_bot.analyzer import PRReviewReport, RepoReviewReport, analyze_git_pr, analyze_repository
from pr_bot.gemini_reviewer import generate_vertex_gemini_review


def main(argv: list[str] | None = None) -> int:
    """Run PR or full-repo 4-layer analysis and write Markdown + optional inline JSON."""
    parser = argparse.ArgumentParser(
        prog="ag-pr-review",
        description=(
            "Compiler-aware 4-layer GitHub PR & periodic repo reviewer "
            "(AST + CPG + LLVM IR + Machine Code)."
        ),
    )
    parser.add_argument("--mode", choices=("pr", "repo"), default="pr", help="Review mode.")
    parser.add_argument("--repo", default=".", help="Path to git repository root.")
    parser.add_argument("--base", default="HEAD~1", help="Base git ref/SHA (PR mode).")
    parser.add_argument("--head", default="HEAD", help="Head git ref/SHA (PR mode).")
    parser.add_argument("--compile-commands", default=None, help="Optional compile_commands.json.")
    parser.add_argument("--out-md", default="pr_review_summary.md", help="Output Markdown path.")
    parser.add_argument(
        "--out-comments", default="pr_review_comments.json", help="Output inline comments JSON."
    )
    parser.add_argument(
        "--vertex-project",
        default=None,
        help="Optional Google Cloud project ID for Vertex AI IAM/WIF auth.",
    )
    parser.add_argument("--vertex-location", default="global", help="Vertex AI region/endpoint.")
    parser.add_argument(
        "--gemini-model", default="gemini-4-hydrogen", help="Gemini model name."
    )
    args = parser.parse_args(argv)

    report: PRReviewReport | RepoReviewReport
    diff_text = ""
    if args.mode == "repo":
        report = analyze_repository(
            repo_path=args.repo,
            compile_commands_path=args.compile_commands,
        )
        inline_comments = []
    else:
        report = analyze_git_pr(
            repo_path=args.repo,
            base_ref=args.base,
            head_ref=args.head,
            compile_commands_path=args.compile_commands,
        )
        inline_comments = report.to_github_review_comments()
        diff_proc = subprocess.run(
            ["git", "-C", str(Path(args.repo).resolve()), "diff", args.base, args.head],
            shell=False,
            capture_output=True,
            text=True,
            check=False,
        )
        diff_text = diff_proc.stdout

    markdown = report.to_markdown()
    has_api_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if args.vertex_project or has_api_key:
        ai_review = generate_vertex_gemini_review(
            report=report,
            diff_text=diff_text,
            project=args.vertex_project,
            location=args.vertex_location,
            model=args.gemini_model,
        )
        if ai_review:
            markdown = (
                f"{markdown}\n### Gemini Review ({args.gemini_model})\n\n{ai_review}\n"
            )

    Path(args.out_md).write_text(markdown, encoding="utf-8")
    Path(args.out_comments).write_text(json.dumps(inline_comments, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
