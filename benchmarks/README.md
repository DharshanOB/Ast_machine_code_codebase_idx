# Astraea Benchmarks

This directory contains reproducible micro-benchmarks for:

1. **Cross-Language CPG Accuracy** (`C`, `C++`, `Rust`, `Python`, `JavaScript`): verifies call-graph precision and recall while ignoring calls inside comments and strings and isolating incompatible languages.
2. **Full vs. Incremental Indexing & Retrieval Latency**: measures full workspace indexing, incremental no-op re-indexing, and Tantivy + Qdrant RRF retrieval against a naive full-tree regex scan.

## Running

```bash
python benchmarks/bench_astraea.py
```
