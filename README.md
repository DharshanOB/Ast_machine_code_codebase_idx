![Astraea](assets/astraea-banner.svg)

# Astraea

Astraea is a 4-layer code indexer, semantic graph analyzer, and AI code reviewer. It unifies syntax, call-graph, intermediate representation, and machine-code evidence so both automated CI reviews and MCP coding agents operate on grounded program structure rather than raw diffs or metrics alone.

## Architecture

Astraea indexes source repositories across four aligned layers:

1. **Layer 1 (AST & Lexical):** Tree-sitter parsing and structural chunking for C, C++, Rust, Python, and JavaScript, computing cyclomatic complexity and normalized token/AST vectors.
2. **Layer 2 (SCIP / CPG):** Cross-file Code Property Graph (`CALLS` and `DEF_USE` edges) enabling cycle-safe k-hop caller/callee impact slicing and blast-radius analysis.
3. **Layer 3 (IR / Bytecode):** Compiler lowering via `clang` / `rustc` to LLVM SSA IR (with Weisfeiler-Lehman graph vector isomorphism for NFC refactor proofs), plus `dis` bytecode lowering for Python modules and functions.
4. **Layer 4 (Machine Code & DWARF):** ELF disassembly via `lief` and `capstone` aligned back to source line ranges using DWARF debug metadata.

When generating an AI review with Gemini (via Vertex AI or `GEMINI_API_KEY`), Astraea attaches deterministic findings, source code for top complexity/blast-radius hotspots, and bounded 1-hop CPG caller/callee source snippets to prevent attention dilution and stay within token budgets.

## Installation

Requires Python 3.11+ and optionally `clang-18` / `rustc` for native C/C++/Rust IR lowering.

```bash
pip install -e ".[dev,gcp]"
```

## Usage

### 1. Pull Request Review (`--mode pr`)

Compares a base ref against a head ref across all four layers, verifies semantics-preserving (NFC) refactors, computes caller blast radius, and optionally generates a Gemini review:

```bash
astraea-review \
  --mode pr \
  --repo . \
  --base HEAD~1 \
  --head HEAD \
  --out-md pr_review_summary.md \
  --out-comments pr_review_comments.json
```

### 2. Periodic Repository Audit (`--mode repo`)

Scans the entire repository, clusters semantic clones, scales complexity/blast-radius findings, lists degraded units, and diffs against a previous baseline when `--baseline` is provided:

```bash
astraea-review \
  --mode repo \
  --repo . \
  --baseline .astraea_baseline.json \
  --out-baseline .astraea_baseline.json \
  --out-md repo_review_summary.md
```

### 3. Enabling Gemini via Vertex AI or API Key

Pass Vertex AI flags or set `GEMINI_API_KEY`:

```bash
astraea-review \
  --mode repo \
  --repo . \
  --vertex-project YOUR_GCP_PROJECT_ID \
  --vertex-location global \
  --gemini-model gemini-4-hydrogen \
  --out-md repo_review_summary.md
```

For GitHub Actions integration, see [.github/workflows/pr-review.yml](.github/workflows/pr-review.yml) and [.github/workflows/periodic-review.yml](.github/workflows/periodic-review.yml), and [src/pr_bot/README.md](src/pr_bot/README.md).

### 4. MCP Server (`astraea-index`)

Runs the stdio Model Context Protocol server exposing multi-layer indexing and k-hop CPG traversal tools:

```bash
astraea-index
```

## Development Checks

```bash
pytest -v --tb=short
ruff check src/ tests/
mypy --strict src/
```
