"""Performance & cross-language accuracy benchmarks for Astraea."""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path

from indexer.ast_chunker import ASTChunker
from indexer.models import Language
from indexer.semantic_cpg import SemanticCPGBuilder
from mcp_server.code_index_server import CodeIndexPipeline


def run_cross_language_accuracy_benchmark() -> dict[str, dict[str, float]]:
    """Measure AST chunk recall and CPG call-graph precision/recall across 5 languages."""
    samples: dict[Language, tuple[str, str, set[tuple[str, str]]]] = {
        Language.C: (
            "sample.c",
            "int leaf(int x) { return x + 1; }\n"
            "// leaf(0) in comment must be ignored\n"
            "int caller(int y) { return leaf(y); }\n",
            {("caller", "leaf")},
        ),
        Language.CPP: (
            "sample.cpp",
            "int leaf(int x) { return x + 1; }\n"
            "int caller(int y) { return leaf(y); }\n",
            {("caller", "leaf")},
        ),
        Language.RUST: (
            "sample.rs",
            "fn leaf(x: i32) -> i32 { x + 1 }\n"
            "fn caller(y: i32) -> i32 { leaf(y) }\n",
            {("caller", "leaf")},
        ),
        Language.PYTHON: (
            "sample.py",
            "def leaf(x: int) -> int:\n    return x + 1\n\n"
            "def caller(y: int) -> int:\n    # leaf(0)\n    return leaf(y)\n",
            {("caller", "leaf")},
        ),
        Language.JAVASCRIPT: (
            "sample.js",
            "function leaf(x) { return x + 1; }\n"
            "function caller(y) { return leaf(y); }\n",
            {("caller", "leaf")},
        ),
    }

    chunker = ASTChunker()
    results: dict[str, dict[str, float]] = {}
    for lang, (fname, src, expected_edges) in samples.items():
        chunks = chunker.chunk_file(fname, source=src)
        cpg = SemanticCPGBuilder()
        symbols = cpg.build_from_ast_chunks(chunks)
        found_edges: set[tuple[str, str]] = set()
        for u, v, data in cpg.graph.edges(data=True):
            if data.get("kind") == "CALLS" and u in symbols and v in symbols:
                found_edges.add((symbols[u].display_name, symbols[v].display_name))
        tp = len(found_edges & expected_edges)
        precision = tp / len(found_edges) if found_edges else 0.0
        recall = tp / len(expected_edges) if expected_edges else 1.0
        results[lang.value] = {"precision": precision, "recall": recall}
    return results


def run_indexing_and_search_benchmark(num_files: int = 40) -> dict[str, float]:
    """Benchmark full vs incremental indexing and RRF search vs naive regex scan."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i in range(num_files):
            (root / f"mod_{i}.py").write_text(
                f"def helper_{i}(x: int) -> int:\n    return x + {i}\n\n"
                f"def compute_{i}(v: int) -> int:\n    return helper_{i}(v) * 2\n",
                encoding="utf-8",
            )

        pipe = CodeIndexPipeline()
        t0 = time.perf_counter()
        pipe.index_workspace(root, incremental=False)
        full_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        pipe.index_workspace(root, incremental=True)
        inc_ms = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        pipe.store.rrf_search("compute_15", top_k=5)
        rrf_ms = (time.perf_counter() - t2) * 1000.0

        pat = re.compile(r"\bcompute_15\b")
        t3 = time.perf_counter()
        _ = [p for p in root.rglob("*.py") if pat.search(p.read_text(encoding="utf-8"))]
        grep_ms = (time.perf_counter() - t3) * 1000.0

        return {
            "full_index_ms": round(full_ms, 2),
            "incremental_noop_ms": round(inc_ms, 2),
            "rrf_search_ms": round(rrf_ms, 2),
            "naive_regex_scan_ms": round(grep_ms, 2),
        }


if __name__ == "__main__":
    print("Cross-language CPG accuracy:", run_cross_language_accuracy_benchmark())
    print("Indexing & search timings (ms):", run_indexing_and_search_benchmark())
