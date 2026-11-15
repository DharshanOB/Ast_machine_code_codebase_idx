"""Vertex AI / Gemini API reviewer grounded in 4-layer compiler & CPG evidence."""

from __future__ import annotations

import os

from pr_bot.analyzer import PRReviewReport, RepoReviewReport

_SYSTEM_INSTRUCTION = (
    "You are a senior systems & security code reviewer. You are given deterministic "
    "4-layer analysis (AST complexity, SCIP/CPG blast radius, LLVM SSA IR similarity, "
    "and disassembled machine-code basic blocks) alongside optional diff context. "
    "Ground every claim in the provided evidence: highlight risks to downstream callers, "
    "explain compiler-lowering regressions or confirm NFC refactors, and suggest concrete fixes. "
    "Keep the review concise and actionable."
)


def build_review_prompt(
    report: PRReviewReport | RepoReviewReport,
    diff_text: str = "",
    max_diff_chars: int = 12_000,
) -> str:
    """Build a structured prompt combining 4-layer deterministic evidence and optional git diff."""
    sections = [f"### 4-Layer Deterministic Report\n\n{report.to_markdown()}"]
    if diff_text.strip():
        truncated_diff = (
            diff_text[:max_diff_chars] + "\n...[diff truncated]"
            if len(diff_text) > max_diff_chars
            else diff_text
        )
        sections.append(f"### Unified Git Diff\n\n```diff\n{truncated_diff}\n```")

    sections.append(
        "Provide:\n"
        "1. **Executive Risk Assessment** (based on blast radius, hotspots & lowering deltas)\n"
        "2. **Correctness, Security & Performance Observations**\n"
        "3. **Actionable Recommendations**\n"
    )
    return "\n\n".join(sections)


def generate_vertex_gemini_review(
    report: PRReviewReport | RepoReviewReport,
    diff_text: str = "",
    project: str | None = None,
    location: str = "global",
    model: str = "gemini-4-hydrogen",
    api_key: str | None = None,
) -> str:
    """Call Gemini via API key (GEMINI_API_KEY/GOOGLE_API_KEY) or Google Cloud Vertex AI."""
    from google import genai
    from google.genai import types

    resolved_key = (
        api_key
        or os.environ.get("VERTEX_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"}
    if resolved_key:
        client = (
            genai.Client(vertexai=True, api_key=resolved_key)
            if use_vertex
            else genai.Client(api_key=resolved_key)
        )
    elif project:
        client = genai.Client(vertexai=True, project=project, location=location)
    else:
        raise ValueError("Provide either GEMINI_API_KEY/GOOGLE_API_KEY or a GCP --vertex-project.")

    prompt = build_review_prompt(report, diff_text)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.2,
        ),
    )
    text = getattr(response, "text", None)
    return str(text).strip() if text else ""
