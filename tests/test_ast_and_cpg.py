"""Phase 1 Checkpoint 1 tests: AST boundary chunking and cross-file SCIP/CPG linking."""

from __future__ import annotations

from indexer.ast_chunker import ASTChunker, split_identifier_tokens
from indexer.models import CPGEdgeKind, Language, LLVMIRGraph, ProGraMLNode, compute_content_sha256
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


def test_decorated_python_docstring_rust_impl_type_and_cpp_operators() -> None:
    """Decorated Python functions retain docstrings; Rust impl and C++ operators extract leaves."""
    chunker = ASTChunker()

    py_src = (
        "class Service:\n"
        "    @staticmethod\n"
        "    def run(x: int) -> int:\n"
        '        """Execute service task."""\n'
        "        return x + 1\n"
    )
    py_chunks = chunker.chunk_source(py_src, Language.PYTHON, file_path="svc.py")
    run_chunk = next(c for c in py_chunks if c.symbol_name == "run")
    assert "Execute service task." in run_chunk.docstring

    rs_src = (
        "pub struct Circle { pub r: f64 }\n"
        "pub trait Measurable { fn area(&self) -> f64; }\n"
        "impl Measurable for Circle {\n"
        "    fn area(&self) -> f64 { self.r * self.r }\n"
        "}\n"
    )
    rs_chunks = chunker.chunk_source(rs_src, Language.RUST, file_path="circle.rs")
    impl_chunk = next(c for c in rs_chunks if c.node_kind == "impl_item")
    assert impl_chunk.symbol_name == "Circle"

    cpp_src = (
        "class Foo {\n"
        "public:\n"
        "    ~Foo() {}\n"
        "    bool operator==(const Foo& other) const { return true; }\n"
        "};\n"
        "int Foo::bar(int x) { return x + 1; }\n"
    )
    cpp_chunks = chunker.chunk_source(cpp_src, Language.CPP, file_path="foo.cpp")
    cpp_syms = {c.symbol_name for c in cpp_chunks}
    assert "~Foo" in cpp_syms
    assert "operator==" in cpp_syms
    assert "bar" in cpp_syms


def test_cross_class_same_method_name_call_and_diamond_k_hop_slice() -> None:
    """Cross-class delegation to a same-named method is linked and diamond k_hop_slice works."""
    chunker = ASTChunker()
    py_src = (
        "class Inner:\n"
        "    def close(self) -> int:\n"
        "        return 1\n\n"
        "class Wrapper:\n"
        "    def close(self) -> int:\n"
        "        return self.inner.close()\n"
    )
    chunks = chunker.chunk_source(py_src, Language.PYTHON, file_path="wrap.py")
    cpg = SemanticCPGBuilder()
    symbols = cpg.build_from_ast_chunks(chunks)

    close_uris = [u for u, s in symbols.items() if s.display_name == "close"]
    assert len(close_uris) == 2
    inner_uri, wrapper_uri = sorted(close_uris, key=lambda u: symbols[u].line_start)
    call_targets = [
        e.target_id for e in symbols[wrapper_uri].edges if e.kind == CPGEdgeKind.CALLS
    ]
    assert inner_uri in call_targets

    # Diamond graph: A -> B -> D and A -> C -> D
    diamond = SemanticCPGBuilder()
    ua = make_symbol_uri("pkg", "1.0", "A")
    ub = make_symbol_uri("pkg", "1.0", "B")
    uc = make_symbol_uri("pkg", "1.0", "C")
    ud = make_symbol_uri("pkg", "1.0", "D")
    diamond.ingest_scip_payload(
        {
            "documents": [
                {
                    "relative_path": "d.py",
                    "symbols": [
                        {"symbol": ua, "display_name": "A", "kind": "func"},
                        {"symbol": ub, "display_name": "B", "kind": "func"},
                        {"symbol": uc, "display_name": "C", "kind": "func"},
                        {"symbol": ud, "display_name": "D", "kind": "func"},
                    ],
                    "occurrences": [
                        {"enclosing_symbol": ua, "symbol": ub, "range": [2]},
                        {"enclosing_symbol": ua, "symbol": uc, "range": [3]},
                        {"enclosing_symbol": ub, "symbol": ud, "range": [6]},
                        {"enclosing_symbol": uc, "symbol": ud, "range": [9]},
                    ],
                }
            ]
        }
    )
    sliced = diamond.k_hop_slice(ua, max_hops=2, direction="both")
    pairs = {(s["source"], s["target"]) for s in sliced}
    assert pairs == {(ua, ub), (ua, uc), (ub, ud), (uc, ud)}
    assert len(sliced) == 4


def test_validate_against_ir_calls_no_mangled_ir_only_duplicates() -> None:
    """validate_against_ir_calls normalizes mangled C++ IR symbols without polluting ir_only."""
    chunker = ASTChunker()
    cpp_src = (
        "int leaf(int x) { return x + 1; }\n"
        "int caller(int y) { return leaf(y); }\n"
    )
    chunks = chunker.chunk_source(cpp_src, Language.CPP, file_path="calls.cpp")
    cpg = SemanticCPGBuilder()
    cpg.build_from_ast_chunks(chunks)

    ir_graph = LLVMIRGraph(
        function_name="_Z6calleri",
        source_name="caller",
        source_file="calls.cpp",
        nodes=[
            ProGraMLNode(
                node_id=0,
                node_type="inst",
                opcode="call",
                text="%0 = call noundef i32 @_Z4leafi(i32 noundef %y)",
            )
        ],
    )
    validation = cpg.validate_against_ir_calls([ir_graph])
    assert validation["confirmed"] == [("caller", "leaf")]
    assert validation["ir_only"] == []


def test_def_use_self_referential_increment_reaches() -> None:
    """Self-referential assignments (y = y + 2 and y += 1) preserve incoming REACHES edges."""
    chunker = ASTChunker()
    py_src = (
        "def accumulate(x: int) -> int:\n"
        "    y = x + 1\n"
        "    y = y + 2\n"
        "    y += 3\n"
        "    return y\n"
    )
    chunks = chunker.chunk_source(py_src, Language.PYTHON, file_path="acc.py")
    cpg = SemanticCPGBuilder()
    cpg.build_from_ast_chunks(chunks)

    reaches = [
        (u.rsplit(":", 1)[-1], v.rsplit(":", 1)[-1])
        for u, v, d in cpg.graph.edges(data=True)
        if d.get("kind") == CPGEdgeKind.REACHES.value and d.get("metadata", {}).get("var") == "y"
    ]
    assert ("L2", "L3") in reaches
    assert ("L3", "L4") in reaches
    assert ("L4", "L5") in reaches
