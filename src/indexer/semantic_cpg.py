"""Layer 2: SCIP symbol ingestion and cross-file Code Property Graph (CPG) builder."""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx
import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_rust
from tree_sitter import Language as TSLanguage
from tree_sitter import Node, Parser

from indexer.llvm_ir_pass import extract_leaf_symbol_name
from indexer.models import (
    ASTChunkNode,
    CPGEdgeKind,
    Language,
    LLVMIRGraph,
    SCIPEdge,
    SCIPSymbolNode,
)

_CALL_NODE_TYPES: frozenset[str] = frozenset(
    {"call_expression", "call", "macro_invocation", "new_expression"}
)
_COMMENT_OR_STRING_TYPES: frozenset[str] = frozenset(
    {
        "comment",
        "line_comment",
        "block_comment",
        "string",
        "string_literal",
        "raw_string_literal",
        "char_literal",
        "template_string",
    }
)
_ASSIGN_RE = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_:*&<>]*\s+)*([A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?::\s*[A-Za-z0-9_\[\]., |]+)?\s*(?:([+\-*/%&|^]|<<|>>))?=(?!=)"
)
_CONTAINER_CHUNK_KINDS: frozenset[str] = frozenset(
    {
        "class_definition",
        "class_declaration",
        "class_specifier",
        "struct_specifier",
        "impl_item",
        "trait_item",
        "enum_specifier",
    }
)
_MAX_CFG_LINES_PER_CHUNK = 500
_INHERIT_RE = re.compile(
    r"(?:class|struct)\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:\(([^)]+)\)|:\s*(?:public|private|protected)?\s*([A-Za-z_][A-Za-z0-9_:]*))"
)
_RUST_IMPL_RE = re.compile(r"impl\s+([A-Za-z_][A-Za-z0-9_:]*)\s+for\s+([A-Za-z_][A-Za-z0-9_:]*)")
_KEYWORDS = frozenset(
    {"if", "else", "for", "while", "switch", "return", "sizeof", "catch", "def", "class", "fn"}
)
_COMPATIBLE_LANGUAGES: dict[Language, frozenset[Language]] = {
    Language.C: frozenset({Language.C, Language.CPP}),
    Language.CPP: frozenset({Language.C, Language.CPP}),
    Language.RUST: frozenset({Language.RUST}),
    Language.PYTHON: frozenset({Language.PYTHON}),
    Language.JAVASCRIPT: frozenset({Language.JAVASCRIPT}),
}


def make_symbol_uri(
    package: str,
    version: str,
    descriptor_chain: str,
    scheme: str = "scip-local",
) -> str:
    """Format a canonical SCIP symbol URI: <scheme>:<package>/<version>/<descriptor_chain>."""
    clean_pkg = package.strip("/") or "workspace"
    clean_ver = version.strip("/") or "0.0.0"
    clean_desc = descriptor_chain.lstrip("/")
    return f"{scheme}:{clean_pkg}/{clean_ver}/{clean_desc}"


class SemanticCPGBuilder:
    """Builds and queries a unified SCIP + Code Property Graph over AST chunks."""

    def __init__(self, package_name: str = "workspace", version: str = "0.1.0") -> None:
        self.package_name = package_name
        self.version = version
        self.graph: nx.MultiDiGraph[str] = nx.MultiDiGraph()
        self.symbols: dict[str, SCIPSymbolNode] = {}
        self._uri_languages: dict[str, Language] = {}
        self._uri_scopes: dict[str, str] = {}
        self._parsers: dict[Language, Parser] = {
            Language.C: Parser(TSLanguage(tree_sitter_c.language())),
            Language.CPP: Parser(TSLanguage(tree_sitter_cpp.language())),
            Language.RUST: Parser(TSLanguage(tree_sitter_rust.language())),
            Language.PYTHON: Parser(TSLanguage(tree_sitter_python.language())),
            Language.JAVASCRIPT: Parser(TSLanguage(tree_sitter_javascript.language())),
        }

    def ingest_scip_json(self, scip_path: str | Path) -> list[SCIPSymbolNode]:
        """Ingest a JSON-encoded SCIP index payload into the CPG."""
        payload = json.loads(Path(scip_path).read_text(encoding="utf-8"))
        return self.ingest_scip_payload(payload)

    def ingest_scip_payload(self, payload: dict[str, Any]) -> list[SCIPSymbolNode]:
        """Ingest a decoded SCIP index dictionary (documents, occurrences, symbols)."""
        ingested: list[SCIPSymbolNode] = []
        for doc in payload.get("documents", []):
            rel_path = str(doc.get("relative_path", ""))
            for sym_data in doc.get("symbols", []):
                uri = str(sym_data.get("symbol", ""))
                if not uri:
                    continue
                node = SCIPSymbolNode(
                    symbol_uri=uri,
                    display_name=str(sym_data.get("display_name", uri.rsplit("/", 1)[-1])),
                    kind=str(sym_data.get("kind", "symbol")),
                    file_path=rel_path,
                    line_start=max(1, int(sym_data.get("line_start", 1))),
                    line_end=max(1, int(sym_data.get("line_end", 1))),
                    documentation=str(sym_data.get("documentation", "")),
                )
                self._register_symbol(node)
                for rel in sym_data.get("relationships", []):
                    target = str(rel.get("symbol", ""))
                    if not target:
                        continue
                    kind = (
                        CPGEdgeKind.IMPLEMENTS
                        if rel.get("is_implementation") or rel.get("is_type_definition")
                        else CPGEdgeKind.CALLS
                    )
                    self._add_edge(uri, target, kind)
                ingested.append(node)

            for occ in doc.get("occurrences", []):
                caller = str(occ.get("enclosing_symbol", ""))
                callee = str(occ.get("symbol", ""))
                if caller and callee and caller != callee:
                    line = int(occ.get("range", [1])[0])
                    self._add_edge(caller, callee, CPGEdgeKind.CALLS, line=max(1, line))
        return ingested

    def build_from_ast_chunks(self, chunks: list[ASTChunkNode]) -> dict[str, SCIPSymbolNode]:
        """Construct symbol nodes and cross-file CALLS/IMPLEMENTS/REACHES edges from AST chunks."""
        name_to_uris: dict[str, list[str]] = {}
        chunk_uris: list[tuple[ASTChunkNode, str]] = []

        for chunk in chunks:
            name = chunk.symbol_name or f"anon_{chunk.chunk_id[:8]}"
            norm_path = chunk.file_path.replace("\\", "/").lstrip("/") or "<memory>"
            desc = f"{norm_path}/{name}#{chunk.line_start}"
            uri = make_symbol_uri(self.package_name, self.version, desc)
            node = SCIPSymbolNode(
                symbol_uri=uri,
                display_name=name,
                kind=chunk.node_kind,
                file_path=chunk.file_path,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                chunk_id=chunk.chunk_id,
                documentation=chunk.docstring,
            )
            self._register_symbol(node)
            self._uri_languages[uri] = chunk.language
            self._uri_scopes[uri] = chunk.scope_header
            name_to_uris.setdefault(name, []).append(uri)
            chunk_uris.append((chunk, uri))

        for chunk, src_uri in chunk_uris:
            self._link_inheritance(chunk, src_uri, name_to_uris)
            if chunk.node_kind not in _CONTAINER_CHUNK_KINDS:
                calls, sanitized_source = self._extract_calls_and_sanitized_source(chunk)
                self._link_calls(chunk, src_uri, name_to_uris, calls=calls)
                self._link_intraprocedural_def_use(
                    chunk, src_uri, sanitized_source=sanitized_source
                )

        return self.symbols

    def k_hop_slice(
        self,
        start_uri: str,
        max_hops: int = 2,
        edge_kinds: frozenset[CPGEdgeKind] | None = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Perform cycle-safe BFS neighborhood expansion tracking (node_id, hop_depth)."""
        if start_uri not in self.graph:
            return []

        node_min_hop: dict[str, int] = {start_uri: 0}
        emitted_edges: set[tuple[str, str, str, int | None]] = set()
        queue: deque[tuple[str, int]] = deque([(start_uri, 0)])
        results: list[dict[str, Any]] = []
        allowed_kinds = {k.value for k in edge_kinds} if edge_kinds is not None else None

        while queue:
            curr_id, depth = queue.popleft()
            if depth >= max_hops:
                continue

            neighbors: list[tuple[str, str, dict[str, Any]]] = []
            if direction in ("forward", "both"):
                for _, dst, data in self.graph.out_edges(curr_id, data=True):
                    neighbors.append((dst, "out", data))
            if direction in ("backward", "both"):
                for src, _, data in self.graph.in_edges(curr_id, data=True):
                    neighbors.append((src, "in", data))

            next_depth = depth + 1
            for nxt_id, edge_dir, data in neighbors:
                kind_str = str(data.get("kind", ""))
                if allowed_kinds is not None and kind_str not in allowed_kinds:
                    continue
                if nxt_id == curr_id and depth > 0:
                    continue
                src_edge = curr_id if edge_dir == "out" else nxt_id
                dst_edge = nxt_id if edge_dir == "out" else curr_id
                line_val = data.get("line")
                edge_key = (src_edge, dst_edge, kind_str, line_val)
                if edge_key in emitted_edges:
                    continue
                emitted_edges.add(edge_key)
                results.append(
                    {
                        "source": src_edge,
                        "target": dst_edge,
                        "kind": kind_str,
                        "hop": next_depth,
                        "line": line_val,
                    }
                )
                if nxt_id not in node_min_hop:
                    node_min_hop[nxt_id] = next_depth
                    queue.append((nxt_id, next_depth))

        return results

    def _register_symbol(self, node: SCIPSymbolNode) -> None:
        self.symbols[node.symbol_uri] = node
        self.graph.add_node(
            node.symbol_uri,
            display_name=node.display_name,
            kind=node.kind,
            file_path=node.file_path,
            chunk_id=node.chunk_id,
        )

    def _add_edge(
        self,
        source_id: str,
        target_id: str,
        kind: CPGEdgeKind,
        line: int | None = None,
        metadata: dict[str, str] | None = None,
        confidence: str = "heuristic",
        evidence: str = "",
    ) -> None:
        self.graph.add_edge(
            source_id,
            target_id,
            kind=kind.value,
            line=line,
            metadata=metadata or {},
            confidence=confidence,
            evidence=evidence,
        )
        if source_id in self.symbols:
            self.symbols[source_id].edges.append(
                SCIPEdge(
                    source_id=source_id,
                    target_id=target_id,
                    kind=kind,
                    line=line,
                    metadata=metadata or {},
                    confidence=confidence,
                    evidence=evidence,
                )
            )

    def _extract_calls_and_sanitized_source(
        self, chunk: ASTChunkNode
    ) -> tuple[list[tuple[str, str, int]], str]:
        """Parse chunk once to extract AST call sites and mask comments/strings."""
        parser = self._parsers.get(chunk.language)
        if parser is None:
            return [], chunk.source_text
        source_bytes = chunk.source_text.encode("utf-8")
        masked_bytes = bytearray(source_bytes)
        tree = parser.parse(source_bytes)
        calls: list[tuple[str, str, int]] = []

        def _leaf_qualifier(raw: str) -> str:
            return raw.split("::")[-1].split(".")[-1].strip()

        def _callee_identifier(node: Node) -> tuple[str, str]:
            fn_node = node.child_by_field_name("function") or node.child_by_field_name("macro")
            if fn_node is None and node.named_children:
                fn_node = node.named_children[0]
            if fn_node is None:
                return ("", "")
            if fn_node.type in ("attribute", "member_expression", "field_expression"):
                attr = (
                    fn_node.child_by_field_name("attribute")
                    or fn_node.child_by_field_name("property")
                    or fn_node.child_by_field_name("field")
                )
                if attr is not None:
                    name_text = source_bytes[attr.start_byte : attr.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    obj = (
                        fn_node.child_by_field_name("object")
                        or fn_node.child_by_field_name("argument")
                        or fn_node.child_by_field_name("value")
                    )
                    qual_text = (
                        source_bytes[obj.start_byte : obj.end_byte].decode(
                            "utf-8", errors="replace"
                        )
                        if obj is not None
                        else ""
                    )
                    return (name_text, _leaf_qualifier(qual_text))
            if fn_node.type in ("scoped_identifier", "qualified_identifier"):
                name_child = fn_node.child_by_field_name("name")
                if name_child is not None:
                    name_text = source_bytes[name_child.start_byte : name_child.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    scope_child = fn_node.child_by_field_name(
                        "scope"
                    ) or fn_node.child_by_field_name("path")
                    qual_text = (
                        source_bytes[scope_child.start_byte : scope_child.end_byte].decode(
                            "utf-8", errors="replace"
                        )
                        if scope_child is not None
                        else ""
                    )
                    return (name_text, _leaf_qualifier(qual_text))
            text = source_bytes[fn_node.start_byte : fn_node.end_byte].decode(
                "utf-8", errors="replace"
            )
            if "::" in text:
                prefix, leaf = text.rsplit("::", 1)
                return (leaf.strip(), _leaf_qualifier(prefix))
            if "." in text:
                prefix, leaf = text.rsplit(".", 1)
                return (leaf.strip(), _leaf_qualifier(prefix))
            return (text.strip(), "")

        def _walk(node: Node) -> None:
            if node.type in _COMMENT_OR_STRING_TYPES:
                for i in range(node.start_byte, min(node.end_byte, len(masked_bytes))):
                    if masked_bytes[i] != ord("\n"):
                        masked_bytes[i] = ord(" ")
                return
            if node.type in _CALL_NODE_TYPES:
                name, qualifier = _callee_identifier(node)
                if name and name not in _KEYWORDS and (
                    name != chunk.symbol_name or bool(qualifier)
                ):
                    line_no = chunk.line_start + node.start_point[0]
                    calls.append((name, qualifier, line_no))
            for child in node.children:
                _walk(child)

        _walk(tree.root_node)
        return calls, bytes(masked_bytes).decode("utf-8", errors="replace")

    def _extract_calls_ast(self, chunk: ASTChunkNode) -> list[tuple[str, str, int]]:
        """Extract (callee_name, qualifier, 1-based line) from AST call nodes."""
        calls, _ = self._extract_calls_and_sanitized_source(chunk)
        return calls

    def _strip_comments_and_strings(self, chunk: ASTChunkNode) -> str:
        """Blank out comment and string spans using Tree-sitter so def-use ignores them."""
        _, sanitized = self._extract_calls_and_sanitized_source(chunk)
        return sanitized

    def _link_calls(
        self,
        chunk: ASTChunkNode,
        src_uri: str,
        name_to_uris: dict[str, list[str]],
        calls: list[tuple[str, str, int]] | None = None,
    ) -> None:
        compat_langs = _COMPATIBLE_LANGUAGES.get(chunk.language, frozenset({chunk.language}))
        call_list = calls if calls is not None else self._extract_calls_ast(chunk)
        for callee_name, qualifier, line_no in call_list:
            candidates = [
                u
                for u in name_to_uris.get(callee_name, [])
                if u != src_uri
                and (u not in self._uri_languages or self._uri_languages[u] in compat_langs)
            ]
            if callee_name == chunk.symbol_name and (
                not qualifier or qualifier in ("self", "this")
            ):
                if not candidates:
                    self._add_edge(
                        src_uri,
                        src_uri,
                        CPGEdgeKind.CALLS,
                        line=line_no,
                        confidence="heuristic",
                        evidence="ast_call_self_recursive",
                    )
                continue
            if not candidates:
                continue
            targets: list[str] = []
            qualifier_matched = False
            if qualifier and qualifier not in ("self", "this"):
                targets = [
                    u
                    for u in candidates
                    if qualifier in self._uri_scopes.get(u, "")
                    or (u in self.symbols and qualifier in self.symbols[u].file_path)
                ]
                qualifier_matched = bool(targets)
            if not targets:
                same_file = [
                    u
                    for u in candidates
                    if u in self.symbols and self.symbols[u].file_path == chunk.file_path
                ]
                targets = same_file if same_file else candidates
            if len(targets) > 1:
                confidence = "ambiguous"
                evidence = "ast_call_multiple_candidates"
            else:
                confidence = "heuristic"
                evidence = (
                    "ast_call_qualified" if qualifier_matched else "ast_call_single_candidate"
                )
            for target_uri in targets:
                self._add_edge(
                    src_uri,
                    target_uri,
                    CPGEdgeKind.CALLS,
                    line=line_no,
                    confidence=confidence,
                    evidence=evidence,
                )

    def _link_inheritance(
        self, chunk: ASTChunkNode, src_uri: str, name_to_uris: dict[str, list[str]]
    ) -> None:
        for match in _INHERIT_RE.finditer(chunk.source_text):
            raw_bases = match.group(1) or match.group(2) or ""
            for base in raw_bases.split(","):
                base_name = base.strip().split("::")[-1].split(".")[-1]
                for target_uri in name_to_uris.get(base_name, []):
                    if target_uri != src_uri:
                        self._add_edge(
                            src_uri, target_uri, CPGEdgeKind.IMPLEMENTS, line=chunk.line_start
                        )
        for match in _RUST_IMPL_RE.finditer(chunk.source_text):
            trait_name = match.group(1).split("::")[-1]
            for target_uri in name_to_uris.get(trait_name, []):
                if target_uri != src_uri:
                    self._add_edge(
                        src_uri, target_uri, CPGEdgeKind.IMPLEMENTS, line=chunk.line_start
                    )

    def _link_intraprocedural_def_use(
        self,
        chunk: ASTChunkNode,
        src_uri: str,
        sanitized_source: str | None = None,
    ) -> None:
        last_def_node: dict[str, str] = {}
        prev_line_node: str | None = None
        clean_src = (
            sanitized_source
            if sanitized_source is not None
            else self._strip_comments_and_strings(chunk)
        )
        lines = clean_src.splitlines()[:_MAX_CFG_LINES_PER_CHUNK]
        for offset, line_text in enumerate(lines):
            stripped = line_text.strip()
            if not stripped:
                continue
            line_no = chunk.line_start + offset
            line_node_id = f"{src_uri}:L{line_no}"
            self.graph.add_node(line_node_id, kind="cfg_stmt", file_path=chunk.file_path)
            if prev_line_node is not None:
                self._add_edge(prev_line_node, line_node_id, CPGEdgeKind.CFG_NEXT, line=line_no)
            prev_line_node = line_node_id

            assign_m = _ASSIGN_RE.match(stripped)
            if assign_m is not None:
                lhs_var = assign_m.group(1)
                compound_op = assign_m.group(2)
                assigned_vars = {lhs_var}
                rhs_text = stripped[assign_m.end() :]
                used_vars = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", rhs_text))
                if compound_op:
                    used_vars.add(lhs_var)
            else:
                assigned_vars = set()
                used_vars = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", stripped))
            for var in used_vars:
                if var in last_def_node and last_def_node[var] != line_node_id:
                    self._add_edge(
                        last_def_node[var],
                        line_node_id,
                        CPGEdgeKind.REACHES,
                        line=line_no,
                        metadata={"var": var},
                    )
            for var in assigned_vars:
                last_def_node[var] = line_node_id

    def validate_against_ir_calls(
        self, ir_graphs: list[LLVMIRGraph]
    ) -> dict[str, list[tuple[str, str]]]:
        """Validate CPG CALLS edges against compiler-generated LLVM IR call instructions."""
        ir_call_re = re.compile(r'\bcall\b[^@]*@(?:"([^"]+)"|([A-Za-z0-9_.$-]+))\s*\(')
        ir_calls: set[tuple[str, str]] = set()
        for graph in ir_graphs:
            caller_norm = (
                extract_leaf_symbol_name(graph.source_name or graph.function_name)
                or graph.source_name
                or graph.function_name
            )
            for node in graph.nodes:
                if node.opcode == "call":
                    for match in ir_call_re.finditer(node.text):
                        raw_callee = match.group(1) or match.group(2) or ""
                        callee = raw_callee.split(".")[0]
                        if callee and not callee.startswith("llvm"):
                            callee_norm = extract_leaf_symbol_name(raw_callee) or callee
                            ir_calls.add((caller_norm, callee_norm))

        cpg_calls: set[tuple[str, str]] = set()
        ambiguous_calls: set[tuple[str, str]] = set()
        for src_uri, dst_uri, data in self.graph.edges(data=True):
            if data.get("kind") == CPGEdgeKind.CALLS.value:
                src_sym = self.symbols.get(src_uri)
                dst_sym = self.symbols.get(dst_uri)
                if src_sym and dst_sym:
                    pair = (src_sym.display_name, dst_sym.display_name)
                    cpg_calls.add(pair)
                    if data.get("confidence") == "ambiguous":
                        ambiguous_calls.add(pair)
                    elif pair in ir_calls:
                        data["confidence"] = "confirmed"
                        data["evidence"] = "llvm_ir_call"

        for sym in self.symbols.values():
            for edge in sym.edges:
                if edge.kind == CPGEdgeKind.CALLS and edge.confidence != "ambiguous":
                    dst_sym = self.symbols.get(edge.target_id)
                    if dst_sym and (sym.display_name, dst_sym.display_name) in ir_calls:
                        edge.confidence = "confirmed"
                        edge.evidence = "llvm_ir_call"

        return {
            "confirmed": sorted(cpg_calls & ir_calls),
            "cpg_only": sorted(cpg_calls - ir_calls),
            "ir_only": sorted(ir_calls - cpg_calls),
            "ambiguous": sorted(ambiguous_calls),
        }
