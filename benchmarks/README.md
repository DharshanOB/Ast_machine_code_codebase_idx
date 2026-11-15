# Astraea Benchmarks

This directory contains reproducible micro-benchmarks for:

1. **Cross-Language CPG Accuracy** (`C`, `C++`, `Rust`, `Python`, `JavaScript`):
   - **Multi-file corpus**: two source files per language (`core`/`math` and `app`) with cross-file call edges.
   - **Classes, structs, namespaces, modules, and methods**: covers `struct`/`class`/`namespace`/`impl`/`mod` scopes alongside top-level functions.
   - **Comment & string-literal false-positive traps**: embeds fake call expressions in line comments, block comments, string literals, and template strings to verify Tree-sitter AST filtering.
   - **Cross-language isolation**: indexes all 5 languages into a unified `SemanticCPGBuilder` with colliding symbol names (`compute` across `C`, `Rust`, `Python`, and `JavaScript`) and verifies zero `CALLS` edges cross incompatible language boundaries.
   - Reports per-language `symbol_recall`, `precision`, `recall`, plus `cross_language_isolation` (`isolation`, `violations`).
2. **Full vs. Incremental Indexing & Retrieval Latency**: measures full workspace indexing, incremental no-op re-indexing, and Tantivy + Qdrant RRF retrieval against a naive full-tree regex scan.
3. **Real-Repository Benchmark** (`run_real_repo_benchmark`): indexes the real repository into a temporary `storage_dir` and reports:
   - `cold_index_ms` & `incremental_noop_ms`
   - `incremental_speedup` (`cold_index_ms / max(incremental_noop_ms, 0.01)`)
   - `retrieval_p95_ms` across 20 `rrf_search` queries
   - `storage_bytes` (disk footprint of the temporary `storage_dir`) & `indexed_units`

## Running

```bash
PYTHONPATH=src python benchmarks/bench_astraea.py
```

To enforce CI accuracy, isolation, and retrieval latency thresholds (`symbol_recall >= 0.95`, `precision >= 0.95`, `recall >= 0.95`, `cross_language_isolation.violations == 0.0`, and `retrieval_p95_ms <= 500.0`):

```bash
PYTHONPATH=src python benchmarks/bench_astraea.py --check-thresholds
```

Or using the project virtual environment:

```bash
PYTHONPATH=src /path/to/venv/bin/python benchmarks/bench_astraea.py --check-thresholds
```
