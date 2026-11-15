# Astraea Review Bot (`pr_bot`)

`pr_bot` powers Astraea's automated pull request reviews and scheduled full-repository health audits.

## Components

- `analyzer.py`: Implements `analyze_pr_directories`, `analyze_git_pr`, and `analyze_repository`. Collects AST complexity, CPG blast radius, clustered clones, LLVM IR / Python bytecode lowering, top hotspot sources, and bounded 1-hop CPG caller/callee context.
- `gemini_reviewer.py`: Builds grounded prompts (`build_review_prompt`) and calls Vertex AI / Gemini (`generate_vertex_gemini_review`).
- `cli.py`: CLI entrypoint (`astraea-review`).

## GitHub Actions Setup

1. Configure repository variables:
   - `GCP_PROJECT_ID`: Google Cloud project ID
   - `GCP_LOCATION`: Vertex AI location (for example `global`)
   - `GEMINI_MODEL`: Gemini model name (for example `gemini-4-hydrogen`)
   - `GOOGLE_GENAI_USE_VERTEXAI`: `true`
2. Configure repository secrets (for Workload Identity Federation):
   - `GCP_WIF_PROVIDER`: Workload Identity Provider resource name
   - `GCP_SERVICE_ACCOUNT`: Service account email with `roles/aiplatform.user`
   - Or `GEMINI_API_KEY` if authenticating with an API key instead of Vertex AI.
3. Add `.github/workflows/pr-review.yml` and `.github/workflows/periodic-review.yml` to your target repository.
