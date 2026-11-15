"""Layer 1: Structure-aware Tree-sitter cAST chunker for C, C++, Rust, and Python."""

from __future__ import annotations

import re
from pathlib import Path

import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_rust
from tree_sitter import Language as TSLanguage
from tree_sitter import Node, Parser

from indexer.models import ASTChunkNode, Language

_EXTENSION_MAP: dict[str, Language] = {
    ".c": Language.C,
    ".h": Language.C,
    ".cc": Language.CPP,
    ".cpp": Language.CPP,
    ".cxx": Language.CPP,
    ".hpp": Language.CPP,
    ".hh": Language.CPP,
    ".rs": Language.RUST,
    ".py": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
}

_CHUNK_NODE_TYPES: dict[Language, frozenset[str]] = {
    Language.C: frozenset({"function_definition", "struct_specifier", "enum_specifier"}),
    Language.CPP: frozenset(
        {"function_definition", "class_specifier", "struct_specifier", "template_declaration"}
    ),
    Language.RUST: frozenset({"function_item", "impl_item", "struct_item", "trait_item"}),
    Language.PYTHON: frozenset({"function_definition", "class_definition", "decorated_definition"}),
    Language.JAVASCRIPT: frozenset(
        {
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "method_definition",
        }
    ),
}

_SCOPE_NODE_TYPES: frozenset[str] = frozenset(
    {
        "namespace_definition",
        "class_specifier",
        "struct_specifier",
        "impl_item",
        "mod_item",
        "class_definition",
        "class_declaration",
    }
)

_IMPORT_NODE_TYPES: frozenset[str] = frozenset(
    {
        "preproc_include",
        "using_declaration",
        "use_declaration",
        "import_statement",
        "import_from_statement",
    }
)

_BRANCH_NODE_TYPES: frozenset[str] = frozenset(
    {
        "if_statement",
        "elif_clause",
        "for_statement",
        "for_expression",
        "while_statement",
        "while_expression",
        "do_statement",
        "case_statement",
        "switch_case",
        "conditional_expression",
        "ternary_expression",
        "match_arm",
        "except_clause",
        "catch_clause",
        "boolean_operator",
    }
)

_IDENTIFIER_SPLIT_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+")


def split_identifier_tokens(text: str) -> list[str]:
    """Split camelCase and snake_case identifiers into normalized lowercase tokens."""
    tokens: list[str] = []
    seen: set[str] = set()
    for raw_word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        for segment in raw_word.split("_"):
            if not segment:
                continue
            parts = _IDENTIFIER_SPLIT_RE.findall(segment) or [segment]
            for part in parts:
                lower = part.lower()
                if lower not in seen:
                    seen.add(lower)
                    tokens.append(lower)
    return tokens


_HEADER_EXTENSIONS: frozenset[str] = frozenset({".h", ".hpp", ".hh"})
MAX_FILE_BYTES: int = 2_000_000


def is_header_file(file_path: str | Path) -> bool:
    """Return True if file_path is a C/C++ header rather than a compilation unit."""
    return Path(file_path).suffix.lower() in _HEADER_EXTENSIONS


def detect_language(file_path: str | Path) -> Language | None:
    """Infer supported Language from a file extension."""
    return _EXTENSION_MAP.get(Path(file_path).suffix.lower())


class ASTChunker:
    """Structure-aware Tree-sitter chunker with scope headers and error isolation."""

    def __init__(self) -> None:
        self._languages: dict[Language, TSLanguage] = {
            Language.C: TSLanguage(tree_sitter_c.language()),
            Language.CPP: TSLanguage(tree_sitter_cpp.language()),
            Language.RUST: TSLanguage(tree_sitter_rust.language()),
            Language.PYTHON: TSLanguage(tree_sitter_python.language()),
            Language.JAVASCRIPT: TSLanguage(tree_sitter_javascript.language()),
        }
        self._parsers: dict[Language, Parser] = {
            lang: Parser(ts_lang) for lang, ts_lang in self._languages.items()
        }

    def chunk_file(self, file_path: str | Path, source: str | None = None) -> list[ASTChunkNode]:
        """Parse a source file and return cohesive cAST chunks."""
        path_str = str(file_path)
        lang = detect_language(path_str)
        if lang is None:
            return []
        if source is None:
            path = Path(path_str)
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    return []
            except OSError:
                return []
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            if len(source.encode("utf-8", errors="ignore")) > MAX_FILE_BYTES:
                return []
            text = source
        return self.chunk_source(text, lang, file_path=path_str)

    def chunk_source(
        self, source: str, language: Language, file_path: str = "<memory>"
    ) -> list[ASTChunkNode]:
        """Parse source text in the given language into structure-aware ASTChunkNodes."""
        parser = self._parsers[language]
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)
        root = tree.root_node

        imports = self._collect_imports(root, source_bytes)
        chunks: list[ASTChunkNode] = []
        target_kinds = _CHUNK_NODE_TYPES[language]

        self._walk_chunks(
            node=root,
            source_bytes=source_bytes,
            language=language,
            file_path=file_path,
            target_kinds=target_kinds,
            imports=imports,
            scope_stack=[],
            chunks=chunks,
        )

        if not chunks and source.strip():
            chunks.append(
                self._build_chunk(
                    node=root,
                    source_bytes=source_bytes,
                    language=language,
                    file_path=file_path,
                    scope_header="\n".join(imports),
                    docstring="",
                )
            )
        return chunks

    def _walk_chunks(
        self,
        node: Node,
        source_bytes: bytes,
        language: Language,
        file_path: str,
        target_kinds: frozenset[str],
        imports: list[str],
        scope_stack: list[str],
        chunks: list[ASTChunkNode],
    ) -> None:
        if node.type in target_kinds:
            # Avoid duplicating wrapped decorated_definition or template_declaration nodes.
            if node.type == "decorated_definition" or node.type == "template_declaration":
                inner = self._find_inner_chunk(node, target_kinds)
                target_node = node
                symbol_node = inner if inner is not None else node
            else:
                target_node = node
                symbol_node = node

            scope_parts = [*imports, *scope_stack]
            docstring = self._extract_docstring(
                target_node, source_bytes, language, symbol_node=symbol_node
            )
            chunks.append(
                self._build_chunk(
                    node=target_node,
                    source_bytes=source_bytes,
                    language=language,
                    file_path=file_path,
                    scope_header="\n".join(scope_parts),
                    docstring=docstring,
                    symbol_node=symbol_node,
                )
            )
            # Recurse into classes/impls/structs to capture methods as separate chunks too.
            if symbol_node.type in _SCOPE_NODE_TYPES:
                scope_label = self._format_scope_label(symbol_node, source_bytes)
                next_stack = [*scope_stack, scope_label] if scope_label else scope_stack
                for child in symbol_node.children:
                    self._walk_chunks(
                        child,
                        source_bytes,
                        language,
                        file_path,
                        target_kinds,
                        imports,
                        next_stack,
                        chunks,
                    )
            return

        next_stack = scope_stack
        if node.type in _SCOPE_NODE_TYPES:
            scope_label = self._format_scope_label(node, source_bytes)
            if scope_label:
                next_stack = [*scope_stack, scope_label]

        for child in node.children:
            self._walk_chunks(
                child,
                source_bytes,
                language,
                file_path,
                target_kinds,
                imports,
                next_stack,
                chunks,
            )

    def _find_inner_chunk(self, wrapper: Node, target_kinds: frozenset[str]) -> Node | None:
        for child in wrapper.children:
            if child.type in target_kinds and child.type not in (
                "decorated_definition",
                "template_declaration",
            ):
                return child
        return None

    def _collect_imports(self, root: Node, source_bytes: bytes) -> list[str]:
        imports: list[str] = []
        for child in root.children:
            if child.type in _IMPORT_NODE_TYPES:
                imports.append(self._slice(child, source_bytes).strip())
        return imports[:20]

    def _format_scope_label(self, node: Node, source_bytes: bytes) -> str:
        name = self._extract_symbol_name(node, source_bytes)
        return f"{node.type} {name}".strip() if name else node.type

    def _build_chunk(
        self,
        node: Node,
        source_bytes: bytes,
        language: Language,
        file_path: str,
        scope_header: str,
        docstring: str,
        symbol_node: Node | None = None,
    ) -> ASTChunkNode:
        text = self._slice(node, source_bytes)
        sym_node = symbol_node or node
        symbol_name = self._extract_symbol_name(sym_node, source_bytes)
        complexity, has_error = self._analyze_subtree(node)
        return ASTChunkNode(
            file_path=file_path,
            language=language,
            node_kind=sym_node.type,
            symbol_name=symbol_name,
            byte_start=node.start_byte,
            byte_end=node.end_byte,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            scope_header=scope_header,
            docstring=docstring,
            source_text=text,
            cyclomatic_complexity=complexity,
            lexical_tokens=split_identifier_tokens(text),
            has_syntax_error=has_error,
        )

    def _extract_symbol_name(self, node: Node, source_bytes: bytes) -> str:
        if node.type == "impl_item":
            type_node = node.child_by_field_name("type")
            if type_node is not None:
                return (
                    self._slice(type_node, source_bytes).split("<", 1)[0].split("::")[-1].strip()
                )
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return self._slice(name_node, source_bytes).strip()
        decl_leaf_types = (
            "identifier",
            "field_identifier",
            "qualified_identifier",
            "destructor_name",
            "operator_name",
        )
        declarator = node.child_by_field_name("declarator")
        while declarator is not None:
            if declarator.type in decl_leaf_types:
                return self._slice(declarator, source_bytes).split("::")[-1].strip()
            inner = declarator.child_by_field_name("declarator")
            if inner is None:
                for child in declarator.children:
                    if child.type in decl_leaf_types:
                        return self._slice(child, source_bytes).split("::")[-1].strip()
                break
            declarator = inner
        for child in node.children:
            if child.type in ("type_identifier", "identifier", "property_identifier"):
                return self._slice(child, source_bytes).strip()
        return ""

    def _extract_docstring(
        self,
        node: Node,
        source_bytes: bytes,
        language: Language,
        symbol_node: Node | None = None,
    ) -> str:
        comments: list[str] = []
        prev = node.prev_named_sibling
        while prev is not None and prev.type in ("comment", "line_comment", "block_comment"):
            comments.append(self._slice(prev, source_bytes).strip())
            prev = prev.prev_named_sibling
        if comments:
            comments.reverse()
            return "\n".join(comments)

        if language == Language.PYTHON:
            body = (symbol_node or node).child_by_field_name("body")
            if body is not None and body.named_child_count > 0:
                first_stmt = body.named_children[0]
                if first_stmt.type == "expression_statement" and first_stmt.named_child_count > 0:
                    expr = first_stmt.named_children[0]
                    if expr.type == "string":
                        return self._slice(expr, source_bytes).strip()
        return ""

    def _analyze_subtree(self, root: Node) -> tuple[int, bool]:
        complexity = 1
        has_error = False
        stack: list[Node] = [root]
        while stack:
            curr = stack.pop()
            if curr.type == "ERROR" or curr.is_missing:
                has_error = True
            if curr.type in _BRANCH_NODE_TYPES:
                complexity += 1
            stack.extend(curr.children)
        return complexity, has_error

    @staticmethod
    def _slice(node: Node, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
