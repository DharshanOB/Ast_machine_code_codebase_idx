"""Performance & cross-language accuracy benchmarks for Astraea."""

from __future__ import annotations

import argparse
import math
import re
import tempfile
import time
from pathlib import Path

from indexer.ast_chunker import ASTChunker
from indexer.models import ASTChunkNode, Language
from indexer.semantic_cpg import SemanticCPGBuilder
from mcp_server.code_index_server import CodeIndexPipeline

_CALLABLE_KINDS: frozenset[str] = frozenset(
    {
        "function_definition",
        "function_item",
        "function_declaration",
        "generator_function_declaration",
        "method_definition",
    }
)

_INCOMPATIBLE_CHECK_LANGS: frozenset[Language] = frozenset(
    {Language.C, Language.CPP, Language.RUST, Language.PYTHON, Language.JAVASCRIPT}
)


def _language_corpus() -> dict[
    Language, tuple[list[tuple[str, str]], set[str], set[tuple[str, str]]]
]:
    """Return multi-file corpus per language: (files, expected_symbols, expected_call_edges)."""
    return {
        Language.C: (
            [
                (
                    "c_math.c",
                    "struct ScaleConfig { int factor; };\n"
                    "int compute(int x) { return x * 2; }\n"
                    "int scale_value(int v, struct ScaleConfig cfg) {\n"
                    "    return compute(v) + cfg.factor;\n"
                    "}\n",
                ),
                (
                    "c_app.c",
                    "int normalize(int v) { return v > 0 ? v : 0; }\n"
                    "int run_c_pipeline(int x) {\n"
                    "    // compute(99); normalize(0);\n"
                    "    /* scale_value(1, cfg); */\n"
                    '    const char *msg = "compute(42) scale_value(3)";\n'
                    "    struct ScaleConfig cfg = {3};\n"
                    "    int s = scale_value(x, cfg);\n"
                    "    return normalize(s);\n"
                    "}\n",
                ),
            ],
            {"ScaleConfig", "compute", "scale_value", "normalize", "run_c_pipeline"},
            {
                ("scale_value", "compute"),
                ("run_c_pipeline", "scale_value"),
                ("run_c_pipeline", "normalize"),
            },
        ),
        Language.CPP: (
            [
                (
                    "cpp_core.cpp",
                    "namespace math {\n"
                    "int cpp_leaf(int x) { return x + 1; }\n"
                    "int cpp_scale(int x) { return cpp_leaf(x) * 2; }\n"
                    "}\n"
                    "class Engine {\n"
                    "public:\n"
                    "    int apply(int v) { return math::cpp_scale(v); }\n"
                    "};\n",
                ),
                (
                    "cpp_app.cpp",
                    "int cpp_dispatch(int x) {\n"
                    "    // cpp_leaf(0); apply(0);\n"
                    '    const char *trap = "cpp_scale(10)";\n'
                    "    Engine eng;\n"
                    "    return eng.apply(x);\n"
                    "}\n",
                ),
            ],
            {"cpp_leaf", "cpp_scale", "Engine", "apply", "cpp_dispatch"},
            {
                ("cpp_scale", "cpp_leaf"),
                ("apply", "cpp_scale"),
                ("cpp_dispatch", "apply"),
            },
        ),
        Language.RUST: (
            [
                (
                    "rs_core.rs",
                    "pub struct Accumulator { base: i32 }\n"
                    "pub fn compute(x: i32) -> i32 { x + 10 }\n"
                    "impl Accumulator {\n"
                    "    pub fn step(&self, v: i32) -> i32 {\n"
                    "        compute(v) + self.base\n"
                    "    }\n"
                    "}\n",
                ),
                (
                    "rs_app.rs",
                    "mod pipeline {\n"
                    "    pub fn finalize(v: i32) -> i32 { v * 2 }\n"
                    "}\n"
                    "pub fn run_rs(v: i32) -> i32 {\n"
                    "    // compute(0); step(0);\n"
                    '    let _trap = "finalize(1) compute(2)";\n'
                    "    let acc = Accumulator { base: 1 };\n"
                    "    let s = acc.step(v);\n"
                    "    pipeline::finalize(s)\n"
                    "}\n",
                ),
            ],
            {"Accumulator", "compute", "step", "finalize", "run_rs"},
            {
                ("step", "compute"),
                ("run_rs", "step"),
                ("run_rs", "finalize"),
            },
        ),
        Language.PYTHON: (
            [
                (
                    "py_core.py",
                    "def compute(x: int) -> int:\n"
                    "    return x + 1\n\n"
                    "class Pipeline:\n"
                    "    def transform(self, v: int) -> int:\n"
                    "        return compute(v) * 2\n",
                ),
                (
                    "py_app.py",
                    "def validate(v: int) -> int:\n"
                    "    return v if v > 0 else 0\n\n"
                    "def run_py(x: int) -> int:\n"
                    "    # compute(0); transform(0)\n"
                    '    note = "compute(99) validate(1)"\n'
                    "    p = Pipeline()\n"
                    "    return validate(p.transform(x))\n",
                ),
            ],
            {"compute", "Pipeline", "transform", "validate", "run_py"},
            {
                ("transform", "compute"),
                ("run_py", "Pipeline"),
                ("run_py", "transform"),
                ("run_py", "validate"),
            },
        ),
        Language.JAVASCRIPT: (
            [
                (
                    "js_core.js",
                    "function compute(x) { return x + 5; }\n"
                    "class Runner {\n"
                    "    execute(v) { return compute(v); }\n"
                    "}\n",
                ),
                (
                    "js_app.js",
                    "function formatResult(v) { return v * 3; }\n"
                    "function runJs(x) {\n"
                    "    // compute(0); execute(0);\n"
                    "    const trap = `formatResult(1) compute(2)`;\n"
                    "    const r = new Runner();\n"
                    "    return formatResult(r.execute(x));\n"
                    "}\n",
                ),
            ],
            {"compute", "Runner", "execute", "formatResult", "runJs"},
            {
                ("execute", "compute"),
                ("runJs", "Runner"),
                ("runJs", "execute"),
                ("runJs", "formatResult"),
            },
        ),
    }


def _are_languages_isolated(src_lang: Language, dst_lang: Language) -> bool:
    """Return True if an edge between src_lang and dst_lang is language-compatible."""
    c_family = {Language.C, Language.CPP}
    if src_lang in c_family and dst_lang in c_family:
        return True
    return src_lang == dst_lang


def run_cross_language_accuracy_benchmark() -> dict[str, dict[str, float]]:
    """Measure AST symbol recall, CPG call precision/recall, and cross-language isolation."""
    corpus = _language_corpus()
    chunker = ASTChunker()
    results: dict[str, dict[str, float]] = {}

    all_chunks: list[ASTChunkNode] = []
    file_to_lang: dict[str, Language] = {}

    for lang, (files, expected_symbols, expected_edges) in corpus.items():
        lang_chunks: list[ASTChunkNode] = []
        for fname, src in files:
            file_to_lang[fname] = lang
            lang_chunks.extend(chunker.chunk_file(fname, source=src))
        all_chunks.extend(lang_chunks)

        cpg = SemanticCPGBuilder()
        symbols = cpg.build_from_ast_chunks(lang_chunks)

        found_symbols = {sym.display_name for sym in symbols.values()}
        symbol_tp = len(found_symbols & expected_symbols)
        symbol_recall = symbol_tp / len(expected_symbols) if expected_symbols else 1.0

        found_edges: set[tuple[str, str]] = set()
        for u, v, data in cpg.graph.edges(data=True):
            if (
                data.get("kind") == "CALLS"
                and u in symbols
                and v in symbols
                and symbols[u].kind in _CALLABLE_KINDS
            ):
                found_edges.add((symbols[u].display_name, symbols[v].display_name))

        edge_tp = len(found_edges & expected_edges)
        precision = edge_tp / len(found_edges) if found_edges else 0.0
        recall = edge_tp / len(expected_edges) if expected_edges else 1.0
        results[lang.value] = {
            "symbol_recall": symbol_recall,
            "precision": precision,
            "recall": recall,
        }

    # Unified multi-language CPG: verify shared symbol names (e.g. `compute`) do not cross-link.
    unified_cpg = SemanticCPGBuilder()
    unified_symbols = unified_cpg.build_from_ast_chunks(all_chunks)
    total_calls = 0
    cross_lang_violations = 0
    for u, v, data in unified_cpg.graph.edges(data=True):
        if data.get("kind") != "CALLS" or u not in unified_symbols or v not in unified_symbols:
            continue
        src_lang = file_to_lang.get(unified_symbols[u].file_path)
        dst_lang = file_to_lang.get(unified_symbols[v].file_path)
        if src_lang in _INCOMPATIBLE_CHECK_LANGS and dst_lang in _INCOMPATIBLE_CHECK_LANGS:
            total_calls += 1
            if not _are_languages_isolated(src_lang, dst_lang):
                cross_lang_violations += 1

    isolation = (
        (total_calls - cross_lang_violations) / total_calls if total_calls else 1.0
    )
    results["cross_language_isolation"] = {
        "isolation": isolation,
        "violations": float(cross_lang_violations),
    }
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


_REAL_REPO_QUERIES: tuple[str, ...] = (
    "ASTChunker",
    "SemanticCPGBuilder",
    "HybridCodeStore",
    "rrf_search",
    "index_workspace",
    "CodeIndexPipeline",
    "LLVMIRPass",
    "BinaryLifter",
    "DWARFAligner",
    "detect_language",
    "build_from_ast_chunks",
    "validate_against_ir_calls",
    "cross_language",
    "sqlite",
    "tantivy",
    "qdrant",
    "embedding",
    "symbol_uri",
    "compile_commands",
    "call_graph",
)


def run_real_repo_benchmark(repo_path: str | Path = ".") -> dict[str, float]:
    """Benchmark cold/incremental indexing, p95 RRF retrieval, and storage on a real repo."""
    target = Path(repo_path).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        storage_dir = Path(tmp) / "storage"
        pipe = CodeIndexPipeline(storage_dir=storage_dir)

        t0 = time.perf_counter()
        cold_stats = pipe.index_workspace(target, incremental=False)
        cold_index_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        pipe.index_workspace(target, incremental=True)
        incremental_noop_ms = (time.perf_counter() - t1) * 1000.0

        incremental_speedup = cold_index_ms / max(incremental_noop_ms, 0.01)

        latencies_ms: list[float] = []
        for query in _REAL_REPO_QUERIES:
            tq = time.perf_counter()
            pipe.store.rrf_search(query, top_k=5)
            latencies_ms.append((time.perf_counter() - tq) * 1000.0)

        sorted_latencies = sorted(latencies_ms)
        p95_idx = min(
            len(sorted_latencies) - 1,
            max(0, math.ceil(0.95 * len(sorted_latencies)) - 1),
        )
        retrieval_p95_ms = sorted_latencies[p95_idx]

        storage_bytes = float(
            sum(p.stat().st_size for p in storage_dir.rglob("*") if p.is_file())
        )
        indexed_units = float(cold_stats.get("indexed_units", 0))

        return {
            "cold_index_ms": round(cold_index_ms, 2),
            "incremental_noop_ms": round(incremental_noop_ms, 2),
            "incremental_speedup": round(incremental_speedup, 2),
            "retrieval_p95_ms": round(retrieval_p95_ms, 2),
            "storage_bytes": storage_bytes,
            "indexed_units": indexed_units,
        }


def _check_thresholds(
    accuracy: dict[str, dict[str, float]],
    real_repo: dict[str, float],
) -> None:
    """Assert accuracy, isolation, and retrieval latency thresholds."""
    for lang in _INCOMPATIBLE_CHECK_LANGS:
        metrics = accuracy[lang.value]
        assert metrics["symbol_recall"] >= 0.95, (
            f"{lang.value} symbol_recall {metrics['symbol_recall']} < 0.95"
        )
        assert metrics["precision"] >= 0.95, (
            f"{lang.value} precision {metrics['precision']} < 0.95"
        )
        assert metrics["recall"] >= 0.95, (
            f"{lang.value} recall {metrics['recall']} < 0.95"
        )

    violations = accuracy["cross_language_isolation"]["violations"]
    assert violations == 0.0, f"cross_language_isolation violations {violations} != 0.0"

    retrieval_p95_ms = real_repo["retrieval_p95_ms"]
    assert retrieval_p95_ms <= 500.0, f"retrieval_p95_ms {retrieval_p95_ms} > 500.0"


def main(argv: list[str] | None = None) -> int:
    """Run Astraea benchmarks and optionally enforce CI thresholds."""
    parser = argparse.ArgumentParser(description="Astraea performance & accuracy benchmarks.")
    parser.add_argument(
        "--check-thresholds",
        action="store_true",
        help="Assert accuracy, isolation, and p95 retrieval latency thresholds.",
    )
    args = parser.parse_args(argv)

    accuracy = run_cross_language_accuracy_benchmark()
    indexing = run_indexing_and_search_benchmark()
    real_repo = run_real_repo_benchmark(".")

    print("Cross-language CPG accuracy:", accuracy)
    print("Indexing & search timings (ms):", indexing)
    print("Real-repo benchmark:", real_repo)

    if args.check_thresholds:
        _check_thresholds(accuracy, real_repo)
        print("All benchmark thresholds passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
