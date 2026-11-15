"""Layer 2: SCIP symbol ingestion and cross-file Code Property Graph (CPG) builder."""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx

from indexer.models import ASTChunkNode, CPGEdgeKind, SCIPEdge, SCIPSymbolNode

_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_ASSIGN_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)")
_INHERIT_RE = re.compile(
    r"(?:class|struct)\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:\(([^)]+)\)|:\s*(?:public|private|protected)?\s*([A-Za-z_][A-Za-z0-9_:]*))"
)
_RUST_IMPL_RE = re.compile(r"impl\s+([A-Za-z_][A-Za-z0-9_:]*)\s+for\s+([A-Za-z_][A-Za-z0-9_:]*)")
_KEYWORDS = frozenset(
    {"if", "else", "for", "while", "switch", "return", "sizeof", "catch", "def", "class", "fn"}
)


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
            desc = f"{Path(chunk.file_path).name}/{name}#{chunk.line_start}"
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
            name_to_uris.setdefault(name, []).append(uri)
            chunk_uris.append((chunk, uri))

        for chunk, src_uri in chunk_uris:
            self._link_calls(chunk, src_uri, name_to_uris)
            self._link_inheritance(chunk, src_uri, name_to_uris)
            self._link_intraprocedural_def_use(chunk, src_uri)

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

        visited_states: set[tuple[str, int]] = {(start_uri, 0)}
        seen_nodes: set[str] = {start_uri}
        queue: deque[tuple[str, int]] = deque([(start_uri, 0)])
        results: list[dict[str, Any]] = []

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

            for nxt_id, edge_dir, data in neighbors:
                kind_str = str(data.get("kind", ""))
                if edge_kinds is not None and kind_str not in {k.value for k in edge_kinds}:
                    continue
                next_depth = depth + 1
                state = (nxt_id, next_depth)
                if state in visited_states:
                    continue
                visited_states.add(state)
                results.append(
                    {
                        "source": curr_id if edge_dir == "out" else nxt_id,
                        "target": nxt_id if edge_dir == "out" else curr_id,
                        "kind": kind_str,
                        "hop": next_depth,
                        "line": data.get("line"),
                    }
                )
                if nxt_id not in seen_nodes:
                    seen_nodes.add(nxt_id)
                queue.append(state)

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
    ) -> None:
        self.graph.add_edge(
            source_id,
            target_id,
            kind=kind.value,
            line=line,
            metadata=metadata or {},
        )
        if source_id in self.symbols:
            self.symbols[source_id].edges.append(
                SCIPEdge(
                    source_id=source_id,
                    target_id=target_id,
                    kind=kind,
                    line=line,
                    metadata=metadata or {},
                )
            )

    def _link_calls(
        self, chunk: ASTChunkNode, src_uri: str, name_to_uris: dict[str, list[str]]
    ) -> None:
        for offset, line_text in enumerate(chunk.source_text.splitlines()):
            line_no = chunk.line_start + offset
            for match in _CALL_RE.finditer(line_text):
                callee_name = match.group(1)
                if callee_name in _KEYWORDS or callee_name == chunk.symbol_name:
                    continue
                for target_uri in name_to_uris.get(callee_name, []):
                    if target_uri != src_uri:
                        self._add_edge(src_uri, target_uri, CPGEdgeKind.CALLS, line=line_no)

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

    def _link_intraprocedural_def_use(self, chunk: ASTChunkNode, src_uri: str) -> None:
        last_def_node: dict[str, str] = {}
        prev_line_node: str | None = None
        for offset, line_text in enumerate(chunk.source_text.splitlines()):
            stripped = line_text.strip()
            if not stripped:
                continue
            line_no = chunk.line_start + offset
            line_node_id = f"{src_uri}:L{line_no}"
            self.graph.add_node(line_node_id, kind="cfg_stmt", file_path=chunk.file_path)
            if prev_line_node is not None:
                self._add_edge(prev_line_node, line_node_id, CPGEdgeKind.CFG_NEXT, line=line_no)
            prev_line_node = line_node_id

            assigned_vars = {m.group(1) for m in _ASSIGN_RE.finditer(stripped)}
            used_vars = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", stripped)) - assigned_vars
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
