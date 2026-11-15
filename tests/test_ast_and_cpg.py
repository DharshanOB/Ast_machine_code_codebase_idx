"""Phase 1 Checkpoint 1 tests: AST boundary chunking and cross-file SCIP/CPG linking."""

from __future__ import annotations

from indexer.ast_chunker import ASTChunker, split_identifier_tokens
from indexer.models import CPGEdgeKind, Language, compute_content_sha256
from indexer.semantic_cpg import SemanticCPGBuilder, make_symbol_uri


def test_deterministic_content_sha256() -> None:
    id_a = compute_content_sha256("ast", "c", "function_definition", "int foo() { return 1; }")
    id_b = compute_content_sha256("ast", "c", "function_definition", "int foo() { return 1; }")
    id_c = compute_content_sha256("ast", "c", "function_definition", "int foo() { return 2; }")
    assert id_a == id_b
    assert id_a != id_c
    assert len(id_a) == 64


def test_identifier_splitting_camel_and_snake() -> None:
    tokens = split_identifier_tokens("calculateTotalScore max_value_2")
    assert "calculate" in tokens
    assert "total" in tokens
    assert "score" in tokens
    assert "max" in tokens
    assert "value" in tokens
    assert "2" in tokens


def test_ast_chunker_multilanguage_and_scope_headers() -> None:
    chunker = ASTChunker()

    c_src = """
#include <stdint.h>

// Clamp value to non-negative
int clamp_branch(int x) {
    if (x > 0) return x;
    return 0;
}
"""
    c_chunks = chunker.chunk_source(c_src, Language.C, file_path="clamp.c")
    assert len(c_chunks) == 1
    assert c_chunks[0].symbol_name == "clamp_branch"
    assert "#include <stdint.h>" in c_chunks[0].scope_header
    assert "Clamp value to non-negative" in c_chunks[0].docstring
    assert c_chunks[0].cyclomatic_complexity >= 2

    py_src = '''
import math

class VectorMath:
    def norm(self, x):
        """Compute magnitude."""
        if x < 0:
            return -x
        return x
'''
    py_chunks = chunker.chunk_source(py_src, Language.PYTHON, file_path="vec.py")
    names = [c.symbol_name for c in py_chunks]
    assert "VectorMath" in names
    assert "norm" in names
    norm_chunk = next(c for c in py_chunks if c.symbol_name == "norm")
    assert "class_definition VectorMath" in norm_chunk.scope_header
    assert "import math" in norm_chunk.scope_header
    assert "Compute magnitude." in norm_chunk.docstring

    rs_src = """
use std::fmt;

fn add_one(val: i32) -> i32 {
    val + 1
}
"""
    rs_chunks = chunker.chunk_source(rs_src, Language.RUST, file_path="lib.rs")
    assert len(rs_chunks) == 1
    assert rs_chunks[0].symbol_name == "add_one"
    assert "use std::fmt;" in rs_chunks[0].scope_header


def test_ut01_error_tolerant_chunking() -> None:
    """UT-01: Tree-sitter isolates ERROR node and still chunks sibling functions cleanly."""
    chunker = ASTChunker()
    broken_c = """
int broken_func(int a) {
    return a + ;
}

int valid_sibling(int x) {
    return x * 2;
}
"""
    chunks = chunker.chunk_source(broken_c, Language.C, file_path="broken.c")
    names = [c.symbol_name for c in chunks]
    assert "valid_sibling" in names
    valid = next(c for c in chunks if c.symbol_name == "valid_sibling")
    assert valid.has_syntax_error is False


def test_ut02_cross_file_call_resolution_and_def_use() -> None:
    """UT-02: Emits exact CALLS edge linking call site in main.c to definition URI in math.c."""
    chunker = ASTChunker()
    math_chunks = chunker.chunk_source(
        "int scale_pos(int x) {\n    int y = x * 4;\n    return y;\n}\n",
        Language.C,
        file_path="src/math.c",
    )
    main_chunks = chunker.chunk_source(
        '#include "math.h"\nint main(void) {\n    return scale_pos(3);\n}\n',
        Language.C,
        file_path="src/main.c",
    )

    builder = SemanticCPGBuilder(package_name="demo", version="1.0.0")
    symbols = builder.build_from_ast_chunks([*math_chunks, *main_chunks])

    main_uri = next(u for u, s in symbols.items() if s.display_name == "main")
    math_uri = next(u for u, s in symbols.items() if s.display_name == "scale_pos")

    call_edges = [
        e
        for e in symbols[main_uri].edges
        if e.kind == CPGEdgeKind.CALLS and e.target_id == math_uri
    ]
    assert len(call_edges) == 1
    assert call_edges[0].line == 3

    # Verify intraprocedural REACHES edge for y = x * 4 -> return y
    reaches_edges = [
        (u, v, d)
        for u, v, d in builder.graph.edges(data=True)
        if d.get("kind") == CPGEdgeKind.REACHES.value
    ]
    assert any(d.get("metadata", {}).get("var") == "y" for _, _, d in reaches_edges)


def test_k_hop_slice_handles_cycles() -> None:
    builder = SemanticCPGBuilder()
    u1 = make_symbol_uri("pkg", "1.0", "a()")
    u2 = make_symbol_uri("pkg", "1.0", "b()")
    builder.ingest_scip_payload(
        {
            "documents": [
                {
                    "relative_path": "a.c",
                    "symbols": [
                        {"symbol": u1, "display_name": "a", "kind": "func"},
                        {"symbol": u2, "display_name": "b", "kind": "func"},
                    ],
                    "occurrences": [
                        {"enclosing_symbol": u1, "symbol": u2, "range": [2]},
                        {"enclosing_symbol": u2, "symbol": u1, "range": [5]},
                    ],
                }
            ]
        }
    )
    sliced = builder.k_hop_slice(u1, max_hops=4, direction="forward")
    assert len(sliced) >= 2
