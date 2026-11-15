"""Multi-language AST chunking & CPG corpus tests across C, C++, Rust, Python, and JavaScript."""

from __future__ import annotations

from dataclasses import dataclass

from indexer.ast_chunker import ASTChunker, detect_language
from indexer.models import ASTChunkNode, CPGEdgeKind, Language
from indexer.semantic_cpg import SemanticCPGBuilder


@dataclass(frozen=True)
class LabeledSource:
    """Labeled source sample with expected extracted symbol names."""

    file_path: str
    language: Language
    source: str
    expected_symbols: frozenset[str]


def test_detect_language_all_five_extensions() -> None:
    assert detect_language("src/main.c") == Language.C
    assert detect_language("include/vec.hpp") == Language.CPP
    assert detect_language("src/lib.rs") == Language.RUST
    assert detect_language("pkg/app.py") == Language.PYTHON
    assert detect_language("web/index.js") == Language.JAVASCRIPT
    assert detect_language("web/widget.jsx") == Language.JAVASCRIPT


def test_nested_functions_and_lambdas_python_js_cpp() -> None:
    """Verify nested functions, closures, and lambdas in Python, JS, and C++."""
    chunker = ASTChunker()

    py_src = """
def outer_pipeline(items: list[int]) -> list[int]:
    def inner_scale(val: int) -> int:
        if val < 0:
            return 0
        return val * 2

    mapper = lambda x: inner_scale(x) + 1
    return [mapper(item) for item in items]
"""
    py_chunks = chunker.chunk_source(py_src, Language.PYTHON, file_path="pipeline.py")
    assert len(py_chunks) == 1
    outer = py_chunks[0]
    assert outer.symbol_name == "outer_pipeline"
    assert outer.has_syntax_error is False
    assert "def inner_scale" in outer.source_text
    assert "lambda x:" in outer.source_text
    assert outer.cyclomatic_complexity >= 2
    assert "mapper" in outer.lexical_tokens

    js_src = """
class Transformer {
    apply(items) {
        const doubleIt = (x) => (x > 0 ? x * 2 : 0);
        return items.map(doubleIt);
    }
}

function makeAdder(base) {
    function add(delta) {
        return base + delta;
    }
    return add;
}
"""
    js_chunks = chunker.chunk_source(js_src, Language.JAVASCRIPT, file_path="transform.js")
    js_names = [c.symbol_name for c in js_chunks]
    assert "Transformer" in js_names
    assert "apply" in js_names
    assert "makeAdder" in js_names
    apply_chunk = next(c for c in js_chunks if c.symbol_name == "apply")
    assert "class_declaration Transformer" in apply_chunk.scope_header
    assert "=>" in apply_chunk.source_text
    assert apply_chunk.cyclomatic_complexity >= 2

    cpp_src = """
#include <algorithm>
#include <vector>

int count_positive(const std::vector<int>& nums) {
    int threshold = 0;
    auto is_pos = [threshold](int v) { return v > threshold; };
    return static_cast<int>(std::count_if(nums.begin(), nums.end(), is_pos));
}
"""
    cpp_chunks = chunker.chunk_source(cpp_src, Language.CPP, file_path="filter.cpp")
    assert len(cpp_chunks) == 1
    cpp_fn = cpp_chunks[0]
    assert cpp_fn.symbol_name == "count_positive"
    assert cpp_fn.has_syntax_error is False
    assert "[threshold](int v)" in cpp_fn.source_text
    assert "threshold" in cpp_fn.lexical_tokens
    assert "pos" in cpp_fn.lexical_tokens


def test_c_macros_and_cpp_templates_namespaces_inheritance() -> None:
    """Test C #define macros and C++ templates, namespaces, and inheritance."""
    chunker = ASTChunker()

    c_src = """
#define CLAMP_ZERO(x) ((x) > 0 ? (x) : 0)
#define MAX_LIMIT 1024

struct Bounds {
    int low;
    int high;
};

int clamp_to_limit(int val) {
    int positive = CLAMP_ZERO(val);
    if (positive > MAX_LIMIT) {
        return MAX_LIMIT;
    }
    return positive;
}
"""
    c_chunks = chunker.chunk_source(c_src, Language.C, file_path="bounds.c")
    c_names = [c.symbol_name for c in c_chunks]
    assert c_names == ["Bounds", "clamp_to_limit"]
    assert all(not c.has_syntax_error for c in c_chunks)
    clamp_fn = next(c for c in c_chunks if c.symbol_name == "clamp_to_limit")
    assert "clamp" in clamp_fn.lexical_tokens
    assert clamp_fn.cyclomatic_complexity >= 2

    cpp_src = """
namespace math {

template <typename T>
T clamp_generic(T value, T lo, T hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

class Shape {
public:
    virtual double area() const = 0;
};

class Rectangle : public Shape {
public:
    double area() const override {
        return 10.0;
    }
};

} // namespace math
"""
    cpp_chunks = chunker.chunk_source(cpp_src, Language.CPP, file_path="shapes.cpp")
    cpp_names = [c.symbol_name for c in cpp_chunks]
    assert "clamp_generic" in cpp_names
    assert "Shape" in cpp_names
    assert "Rectangle" in cpp_names
    assert "area" in cpp_names

    tmpl_chunk = next(c for c in cpp_chunks if c.symbol_name == "clamp_generic")
    assert tmpl_chunk.node_kind == "function_definition"
    assert "template <typename T>" in tmpl_chunk.source_text
    assert "namespace_definition math" in tmpl_chunk.scope_header

    rect_chunk = next(c for c in cpp_chunks if c.symbol_name == "Rectangle")
    assert "namespace_definition math" in rect_chunk.scope_header

    builder = SemanticCPGBuilder(package_name="geom", version="1.0.0")
    symbols = builder.build_from_ast_chunks(cpp_chunks)
    shape_uri = next(u for u, s in symbols.items() if s.display_name == "Shape")
    rect_uri = next(u for u, s in symbols.items() if s.display_name == "Rectangle")
    implements_edges = [
        e
        for e in symbols[rect_uri].edges
        if e.kind == CPGEdgeKind.IMPLEMENTS and e.target_id == shape_uri
    ]
    assert len(implements_edges) == 1


def test_rust_modules_structs_traits_and_impl_blocks() -> None:
    """Test Rust mod_item scoping, struct_item, trait_item, impl_item, and IMPLEMENTS edges."""
    chunker = ASTChunker()
    rs_src = """
use std::fmt;

mod geometry {
    pub struct Circle {
        pub radius: f64,
    }

    pub trait Measurable {
        fn area(&self) -> f64;
    }

    impl Measurable for Circle {
        fn area(&self) -> f64 {
            3.14159 * self.radius * self.radius
        }
    }
}
"""
    rs_chunks = chunker.chunk_source(rs_src, Language.RUST, file_path="geometry.rs")
    kinds_by_name = [(c.node_kind, c.symbol_name) for c in rs_chunks]
    assert ("struct_item", "Circle") in kinds_by_name
    assert ("trait_item", "Measurable") in kinds_by_name
    assert any(k == "impl_item" for k, _ in kinds_by_name)
    assert ("function_item", "area") in kinds_by_name

    circle_chunk = next(c for c in rs_chunks if c.symbol_name == "Circle")
    assert "use std::fmt;" in circle_chunk.scope_header
    assert "mod_item geometry" in circle_chunk.scope_header

    area_fn = next(
        c for c in rs_chunks if c.node_kind == "function_item" and c.symbol_name == "area"
    )
    assert "mod_item geometry" in area_fn.scope_header
    assert "impl_item" in area_fn.scope_header

    builder = SemanticCPGBuilder(package_name="rs_pkg", version="0.1.0")
    symbols = builder.build_from_ast_chunks(rs_chunks)
    trait_uri = next(
        u for u, s in symbols.items() if s.display_name == "Measurable" and s.kind == "trait_item"
    )
    impl_uris = [u for u, s in symbols.items() if s.kind == "impl_item"]
    assert any(
        e.kind == CPGEdgeKind.IMPLEMENTS and e.target_id == trait_uri
        for u in impl_uris
        for e in symbols[u].edges
    )


def test_cross_file_and_cross_module_references() -> None:
    """Verify cross-file CALLS and IMPLEMENTS edges across Python, JS, C++, and Rust."""
    chunker = ASTChunker()

    # Python cross-module CALLS + IMPLEMENTS
    py_base = chunker.chunk_source(
        "class BaseEngine:\n    def run(self):\n        return 1\n",
        Language.PYTHON,
        file_path="pkg/base.py",
    )
    py_sub = chunker.chunk_source(
        "from pkg.base import BaseEngine\n\n"
        "def helper_step(x):\n    return x + 1\n\n"
        "class FastEngine(BaseEngine):\n    def execute(self):\n        return helper_step(5)\n",
        Language.PYTHON,
        file_path="pkg/fast.py",
    )

    # JavaScript cross-file CALLS
    js_math = chunker.chunk_source(
        "export function computeHash(seed) {\n    return seed * 31;\n}\n",
        Language.JAVASCRIPT,
        file_path="web/hash.js",
    )
    js_app = chunker.chunk_source(
        "import { computeHash } from './hash.js';\n"
        "function buildToken(id) {\n    return computeHash(id);\n}\n",
        Language.JAVASCRIPT,
        file_path="web/app.js",
    )

    # C++ cross-file IMPLEMENTS + CALLS
    cpp_hdr = chunker.chunk_source(
        "class StorageBackend {\npublic:\n    virtual int flush() = 0;\n};\n",
        Language.CPP,
        file_path="include/storage.hpp",
    )
    cpp_impl = chunker.chunk_source(
        '#include "storage.hpp"\n'
        "class DiskBackend : public StorageBackend {\npublic:\n"
        "    int flush() override { return 0; }\n};\n",
        Language.CPP,
        file_path="src/disk.cpp",
    )

    all_chunks: list[ASTChunkNode] = [
        *py_base,
        *py_sub,
        *js_math,
        *js_app,
        *cpp_hdr,
        *cpp_impl,
    ]
    builder = SemanticCPGBuilder(package_name="polyglot", version="1.0.0")
    symbols = builder.build_from_ast_chunks(all_chunks)

    # Python cross-file inheritance: FastEngine -> BaseEngine
    base_uri = next(u for u, s in symbols.items() if s.display_name == "BaseEngine")
    fast_uri = next(u for u, s in symbols.items() if s.display_name == "FastEngine")
    assert any(
        e.kind == CPGEdgeKind.IMPLEMENTS and e.target_id == base_uri
        for e in symbols[fast_uri].edges
    )

    # JS cross-file call: buildToken -> computeHash
    build_uri = next(u for u, s in symbols.items() if s.display_name == "buildToken")
    hash_uri = next(u for u, s in symbols.items() if s.display_name == "computeHash")
    assert any(
        e.kind == CPGEdgeKind.CALLS and e.target_id == hash_uri for e in symbols[build_uri].edges
    )

    # C++ cross-file inheritance: DiskBackend -> StorageBackend
    storage_uri = next(u for u, s in symbols.items() if s.display_name == "StorageBackend")
    disk_uri = next(u for u, s in symbols.items() if s.display_name == "DiskBackend")
    assert any(
        e.kind == CPGEdgeKind.IMPLEMENTS and e.target_id == storage_uri
        for e in symbols[disk_uri].edges
    )

    # K-hop slice from buildToken reaches computeHash in 1 hop
    sliced = builder.k_hop_slice(build_uri, max_hops=2, direction="forward")
    assert any(item["target"] == hash_uri and item["hop"] == 1 for item in sliced)


def test_malformed_sources_preserve_valid_siblings_across_all_languages() -> None:
    """Malformed files set has_syntax_error=True while still extracting valid siblings."""
    chunker = ASTChunker()

    malformed_samples: list[tuple[Language, str, str, str]] = [
        (
            Language.C,
            "broken.c",
            "int bad_c(int a) {\n    return a + ;\n}\n\n"
            "int good_c(int b) {\n    return b + 1;\n}\n",
            "good_c",
        ),
        (
            Language.CPP,
            "broken.cpp",
            "int bad_cpp(int a) {\n    return (a + ;\n}\n\n"
            "int good_cpp(int b) {\n    return b * 3;\n}\n",
            "good_cpp",
        ),
        (
            Language.RUST,
            "broken.rs",
            "fn bad_rs(a: i32) -> i32 {\n    let x = ;\n    a\n}\n\n"
            "fn good_rs(b: i32) -> i32 {\n    b + 2\n}\n",
            "good_rs",
        ),
        (
            Language.PYTHON,
            "broken.py",
            "def bad_py(a):\n    return a +\n\ndef good_py(b):\n    return b + 4\n",
            "good_py",
        ),
        (
            Language.JAVASCRIPT,
            "broken.js",
            "function badJs(a) {\n    return a + ;\n}\n\n"
            "function goodJs(b) {\n    return b + 5;\n}\n",
            "goodJs",
        ),
    ]

    for lang, file_path, src, valid_name in malformed_samples:
        chunks = chunker.chunk_source(src, lang, file_path=file_path)
        assert chunks, f"Expected chunks for {lang.value}"
        assert any(c.has_syntax_error for c in chunks), (
            f"Expected at least one chunk with has_syntax_error=True for {lang.value}"
        )
        valid_chunks = [c for c in chunks if c.symbol_name == valid_name]
        assert len(valid_chunks) == 1, f"Missing valid sibling {valid_name} in {lang.value}"
        assert valid_chunks[0].has_syntax_error is False


def test_symbol_extraction_precision_and_recall_labeled_corpus() -> None:
    """Verify symbol extraction precision >= 0.98 and recall >= 0.95 on a labeled corpus."""
    corpus: list[LabeledSource] = [
        LabeledSource(
            file_path="c/codec.c",
            language=Language.C,
            source="""
struct PacketHeader {
    int magic;
    int length;
};

enum CodecStatus {
    CODEC_OK = 0,
    CODEC_ERR = 1
};

int encode_packet(int len) {
    return len + 4;
}

int decode_packet(int raw) {
    return raw - 4;
}
""",
            expected_symbols=frozenset(
                {"PacketHeader", "CodecStatus", "encode_packet", "decode_packet"}
            ),
        ),
        LabeledSource(
            file_path="cpp/vector_ops.cpp",
            language=Language.CPP,
            source="""
namespace linalg {

struct Vec2 {
    double x;
    double y;
};

class Matrix {
public:
    double determinant() const {
        return 1.0;
    }
};

template <typename T>
T dot_scale(T a, T b) {
    return a * b;
}

}
""",
            expected_symbols=frozenset({"Vec2", "Matrix", "determinant", "dot_scale"}),
        ),
        LabeledSource(
            file_path="rust/store.rs",
            language=Language.RUST,
            source="""
pub struct KeyValue {
    pub key: u64,
}

pub trait Store {
    fn get_key(&self) -> u64;
}

pub fn hash_key(k: u64) -> u64 {
    k ^ 0x5555
}

pub fn verify_key(k: u64) -> bool {
    k > 0
}
""",
            expected_symbols=frozenset({"KeyValue", "Store", "hash_key", "verify_key"}),
        ),
        LabeledSource(
            file_path="python/metrics.py",
            language=Language.PYTHON,
            source="""
class Gauge:
    def record(self, value: float) -> float:
        return value

    def reset(self) -> None:
        pass

def compute_mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))

def compute_max(values: list[float]) -> float:
    return max(values)
""",
            expected_symbols=frozenset({"Gauge", "record", "reset", "compute_mean", "compute_max"}),
        ),
        LabeledSource(
            file_path="js/router.js",
            language=Language.JAVASCRIPT,
            source="""
class Router {
    register(path) {
        return path;
    }

    dispatch(path) {
        return path.length;
    }
}

function normalizePath(raw) {
    return raw.trim();
}

function matchRoute(a, b) {
    return a === b;
}
""",
            expected_symbols=frozenset(
                {"Router", "register", "dispatch", "normalizePath", "matchRoute"}
            ),
        ),
    ]

    chunker = ASTChunker()
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for sample in corpus:
        chunks = chunker.chunk_source(sample.source, sample.language, file_path=sample.file_path)
        extracted = {c.symbol_name for c in chunks if c.symbol_name}
        expected = set(sample.expected_symbols)

        true_positives += len(extracted & expected)
        false_positives += len(extracted - expected)
        false_negatives += len(expected - extracted)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    assert precision >= 0.98, f"Precision {precision:.4f} < 0.98"
    assert recall >= 0.95, f"Recall {recall:.4f} < 0.95"
