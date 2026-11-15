# Astraea Benchmarks

This directory contains reproducible micro-benchmarks for:

1. **Cross-Language CPG Accuracy** (`C`, `C++`, `Rust`, `Python`, `JavaScript`):
   - **Multi-file corpus**: two source files per language (`core`/`math` and `app`) with cross-file call edges.
   - **Classes, structs, namespaces, modules, and methods**: covers `struct`/`class`/`namespace`/`impl`/`mod` scopes alongside top-level functions.
   - **Comment & string-literal false-positive traps**: embeds fake call expressions in line comments, block comments, string literals, and template strings to verify Tree-sitter AST filtering.
   - **Cross-language isolation**: indexes all 5 languages into a unified `SemanticCPGBuilder` with colliding symbol names (`compute` across `C`, `Rust`, `Python`, and `JavaScript`) and verifies zero `CALLS` edges cross incompatible language boundaries.
   - Reports per-language `symbol_recall`, `precision`, `recall`, plus `cross_language_isolation` (`isolation`, `violations`).
2. **Full vs. Incremental Indexing & Retrieval Latency**: measures full workspace indexing, incremental no-op re-indexing, and Tantivy + Qdrant RRF retrieval against a naive full-tree regex scan.

## Running

```bash
PYTHONPATH=src python benchmarks/bench_astraea.py
```

Or using the project virtual environment:

```bash
PYTHONPATH=src /path/to/venv/bin/python benchmarks/bench_astraea.py
```
