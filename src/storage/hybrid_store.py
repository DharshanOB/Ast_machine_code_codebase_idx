"""Layer 5: Tri-modal hybrid store (Tantivy + Qdrant + Property Graph) with RRF retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

import networkx as nx
import tantivy
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from indexer.ast_chunker import split_identifier_tokens
from indexer.models import BinaryBasicBlock, LLVMIRGraph, UnifiedCodeUnit

VECTOR_DIM_AST = 128
VECTOR_DIM_IR = 128
VECTOR_DIM_BIN = 128
RRF_ETA = 60.0


class SearchHit(BaseModel):
    """Ranked hybrid retrieval result across lexical, vector, and graph modalities."""

    unit_id: str
    symbol_name: str
    file_path: str
    line_start: int
    line_end: int
    score: float
    modalities: dict[str, int] = Field(default_factory=dict)
    ir_status: str = "DEGRADED"


def _hash_feature_vector(features: list[str], dim: int) -> list[float]:
    """Deterministically project weighted symbolic features into a unit-normalized vector."""
    vec = [0.0] * dim
    if not features:
        return vec
    for feat in features:
        digest = hashlib.sha256(feat.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def compute_wl_ir_vector(
    ir_graph: LLVMIRGraph | None, dim: int = VECTOR_DIM_IR, iterations: int = 2
) -> list[float]:
    """Compute Weisfeiler-Lehman subtree kernel feature vector over ProGraML IR graph."""
    if ir_graph is None or not ir_graph.nodes:
        return [0.0] * dim

    labels: dict[int, str] = {
        n.node_id: (n.opcode or n.node_type or "node") for n in ir_graph.nodes
    }
    adj: dict[int, list[str]] = {n.node_id: [] for n in ir_graph.nodes}
    for edge in ir_graph.edges:
        if edge.source in adj and edge.target in labels:
            adj[edge.source].append(f"{edge.flow_type}:{edge.position}:{labels[edge.target]}")

    features: list[str] = list(labels.values())
    for _ in range(iterations):
        next_labels: dict[int, str] = {}
        for nid, cur_label in labels.items():
            # Sort neighbor labels lexicographically for deterministic hashing.
            nbrs = sorted(adj.get(nid, []))
            combined = f"{cur_label}|{'|'.join(nbrs)}"
            wl_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:12]
            next_labels[nid] = wl_hash
            features.append(wl_hash)
        labels = next_labels
        # Rebuild neighbor labels using updated WL hashes.
        new_adj: dict[int, list[str]] = {nid: [] for nid in labels}
        for edge in ir_graph.edges:
            if edge.source in new_adj and edge.target in labels:
                new_adj[edge.source].append(f"{edge.flow_type}:{edge.position}:{labels[edge.target]}")
        adj = new_adj

    return _hash_feature_vector(features, dim)


def compute_binary_cfg_vector(
    blocks: list[BinaryBasicBlock], dim: int = VECTOR_DIM_BIN
) -> list[float]:
    """Compute structural + instruction-bigram feature vector over normalized binary BB-CFGs."""
    if not blocks:
        return [0.0] * dim
    features: list[str] = []
    for bb in blocks:
        features.append(f"deg:{len(bb.successor_vas)}")
        insns = bb.normalized_instructions
        for idx, insn in enumerate(insns):
            features.append(f"op:{insn}")
            if idx + 1 < len(insns):
                features.append(f"bi:{insn}->{insns[idx + 1]}")
    return _hash_feature_vector(features, dim)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class HybridCodeStore:
    """Coordinates Tantivy lexical search, Qdrant multi-vector search, and SQLite/NX graph."""

    COLLECTION_NAME = "code_units"

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir is not None else None
        self.units: dict[str, UnifiedCodeUnit] = {}
        self.unit_vectors: dict[str, dict[str, list[float]]] = {}
        self.graph: nx.MultiDiGraph[str] = nx.MultiDiGraph()
        self._point_ids: dict[str, int] = {}

        if self.storage_dir is not None:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            tantivy_dir = self.storage_dir / "tantivy"
            tantivy_dir.mkdir(parents=True, exist_ok=True)
            self._qdrant = QdrantClient(path=str(self.storage_dir / "qdrant"))
            self._db_path: Path | None = self.storage_dir / "graph_snapshots.sqlite"
            self._init_sqlite(self._db_path)
        else:
            tantivy_dir = None
            self._qdrant = QdrantClient(":memory:")
            self._db_path = None

        self._tantivy_schema = self._build_tantivy_schema()
        self._tantivy_index = (
            tantivy.Index(self._tantivy_schema, path=str(tantivy_dir))
            if tantivy_dir is not None
            else tantivy.Index(self._tantivy_schema)
        )
        self._ensure_qdrant_collection()

    def upsert_units(
        self, units: list[UnifiedCodeUnit], commit_sha: str = "working-tree"
    ) -> int:
        """Atomically index a batch of UnifiedCodeUnits across all three modalities."""
        if not units:
            return 0

        writer = self._tantivy_index.writer()
        qdrant_points: list[PointStruct] = []

        for unit in units:
            self.units[unit.unit_id] = unit
            ast = unit.ast_chunk
            ir_tokens = unit.ir_graph.canonical_tokens if unit.ir_graph else []
            asm_tokens = [
                insn for bb in unit.binary_blocks for insn in bb.normalized_instructions
            ]

            writer.add_document(
                tantivy.Document(
                    unit_id=unit.unit_id,
                    symbol_name=ast.symbol_name or "anonymous",
                    file_path=ast.file_path,
                    source_tokens=" ".join(ast.lexical_tokens),
                    ir_tokens=" ".join(ir_tokens),
                    asm_tokens=" ".join(asm_tokens),
                )
            )

            ast_vec = _hash_feature_vector(
                [*ast.lexical_tokens, *split_identifier_tokens(ast.scope_header)],
                VECTOR_DIM_AST,
            )
            ir_vec = compute_wl_ir_vector(unit.ir_graph)
            bin_vec = compute_binary_cfg_vector(unit.binary_blocks)
            self.unit_vectors[unit.unit_id] = {
                "ast_dense": ast_vec,
                "ir_structural": ir_vec,
                "bin_cfg": bin_vec,
            }

            pid = self._point_ids.setdefault(unit.unit_id, len(self._point_ids) + 1)
            qdrant_points.append(
                PointStruct(
                    id=pid,
                    vector={
                        "ast_dense": ast_vec,
                        "ir_structural": ir_vec,
                        "bin_cfg": bin_vec,
                    },
                    payload={
                        "unit_id": unit.unit_id,
                        "symbol_name": ast.symbol_name,
                        "file_path": ast.file_path,
                    },
                )
            )

            self.graph.add_node(
                unit.unit_id,
                symbol_name=ast.symbol_name,
                file_path=ast.file_path,
                commit_sha=commit_sha,
            )
            if unit.scip_symbol is not None:
                for edge in unit.scip_symbol.edges:
                    self.graph.add_edge(
                        edge.source_id,
                        edge.target_id,
                        kind=edge.kind.value,
                        line=edge.line,
                    )

        writer.commit()
        self._tantivy_index.reload()
        self._qdrant.upsert(collection_name=self.COLLECTION_NAME, points=qdrant_points)
        if self._db_path is not None:
            self._save_snapshot(commit_sha, units)
        return len(units)

    def rrf_search(
        self,
        query: str,
        layers: list[str] | None = None,
        top_k: int = 10,
        weights: dict[str, float] | None = None,
    ) -> list[SearchHit]:
        """Execute hybrid search with Weighted Reciprocal Rank Fusion across modalities."""
        active = set(layers or ["lex", "ast_vec", "ir_gnn", "bin_cfg"])
        w = weights or {"lex": 1.0, "ast_vec": 1.0, "ir_gnn": 0.9, "bin_cfg": 0.8}
        ranks_by_unit: dict[str, dict[str, int]] = {}
        scores: dict[str, float] = {}

        if "lex" in active:
            for rank, uid in enumerate(self._search_tantivy(query, limit=top_k * 3), start=1):
                ranks_by_unit.setdefault(uid, {})["lex"] = rank
                scores[uid] = scores.get(uid, 0.0) + w.get("lex", 1.0) / (RRF_ETA + rank)

        q_tokens = split_identifier_tokens(query)
        q_ast = _hash_feature_vector(q_tokens, VECTOR_DIM_AST)
        if "ast_vec" in active and any(v != 0.0 for v in q_ast):
            for rank, uid in enumerate(
                self._search_qdrant("ast_dense", q_ast, limit=top_k * 3), start=1
            ):
                ranks_by_unit.setdefault(uid, {})["ast_vec"] = rank
                scores[uid] = scores.get(uid, 0.0) + w.get("ast_vec", 1.0) / (RRF_ETA + rank)

        if "ir_gnn" in active and q_tokens:
            q_ir = _hash_feature_vector(q_tokens, VECTOR_DIM_IR)
            for rank, uid in enumerate(
                self._search_qdrant("ir_structural", q_ir, limit=top_k * 3), start=1
            ):
                ranks_by_unit.setdefault(uid, {})["ir_gnn"] = rank
                scores[uid] = scores.get(uid, 0.0) + w.get("ir_gnn", 0.9) / (RRF_ETA + rank)

        if "bin_cfg" in active and q_tokens:
            q_bin = _hash_feature_vector([f"op:{t.upper()}" for t in q_tokens], VECTOR_DIM_BIN)
            for rank, uid in enumerate(
                self._search_qdrant("bin_cfg", q_bin, limit=top_k * 3), start=1
            ):
                ranks_by_unit.setdefault(uid, {})["bin_cfg"] = rank
                scores[uid] = scores.get(uid, 0.0) + w.get("bin_cfg", 0.8) / (RRF_ETA + rank)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        hits: list[SearchHit] = []
        for uid, score in ordered:
            unit = self.units.get(uid)
            if unit is None:
                continue
            hits.append(
                SearchHit(
                    unit_id=uid,
                    symbol_name=unit.ast_chunk.symbol_name,
                    file_path=unit.ast_chunk.file_path,
                    line_start=unit.ast_chunk.line_start,
                    line_end=unit.ast_chunk.line_end,
                    score=round(score, 6),
                    modalities=ranks_by_unit.get(uid, {}),
                    ir_status=unit.ir_status.value,
                )
            )
        return hits

    def find_clones(
        self,
        unit_id: str,
        modality: str = "ir_structural",
        threshold: float = 0.75,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Find cross-layer semantic clones via cosine similarity in Qdrant."""
        vecs = self.unit_vectors.get(unit_id)
        if vecs is None or modality not in vecs:
            return []
        target_vec = vecs[modality]
        if not any(v != 0.0 for v in target_vec):
            return []

        res = self._qdrant.query_points(
            collection_name=self.COLLECTION_NAME,
            query=target_vec,
            using=modality,
            limit=top_k + 1,
        )
        clones: list[tuple[str, float]] = []
        for pt in res.points:
            payload = pt.payload or {}
            cand_id = str(payload.get("unit_id", ""))
            if not cand_id or cand_id == unit_id:
                continue
            if float(pt.score) >= threshold:
                clones.append((cand_id, round(float(pt.score), 4)))
        return clones[:top_k]

    def _search_tantivy(self, query: str, limit: int) -> list[str]:
        searcher = self._tantivy_index.searcher()
        cleaned = " ".join(split_identifier_tokens(query)) or query.strip()
        if not cleaned:
            return []
        try:
            parsed = self._tantivy_index.parse_query(
                cleaned, ["symbol_name", "source_tokens", "ir_tokens", "asm_tokens"]
            )
        except Exception:
            return []
        top_docs = searcher.search(parsed, limit).hits
        out: list[str] = []
        for _, doc_addr in top_docs:
            doc = searcher.doc(doc_addr)
            vals = doc.get_all("unit_id")
            if vals:
                out.append(str(vals[0]))
        return out

    def _search_qdrant(self, vector_name: str, query_vec: list[float], limit: int) -> list[str]:
        if not self.units:
            return []
        res = self._qdrant.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vec,
            using=vector_name,
            limit=limit,
        )
        out: list[str] = []
        for pt in res.points:
            if float(pt.score) <= 0.0:
                continue
            payload = pt.payload or {}
            uid = str(payload.get("unit_id", ""))
            if uid:
                out.append(uid)
        return out

    @staticmethod
    def _build_tantivy_schema() -> tantivy.Schema:
        builder = tantivy.SchemaBuilder()
        builder.add_text_field("unit_id", stored=True)
        builder.add_text_field("symbol_name", stored=True)
        builder.add_text_field("file_path", stored=True)
        builder.add_text_field("source_tokens", stored=False)
        builder.add_text_field("ir_tokens", stored=False)
        builder.add_text_field("asm_tokens", stored=False)
        return builder.build()

    def _ensure_qdrant_collection(self) -> None:
        if self._qdrant.collection_exists(self.COLLECTION_NAME):
            return
        self._qdrant.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config={
                "ast_dense": VectorParams(size=VECTOR_DIM_AST, distance=Distance.COSINE),
                "ir_structural": VectorParams(size=VECTOR_DIM_IR, distance=Distance.COSINE),
                "bin_cfg": VectorParams(size=VECTOR_DIM_BIN, distance=Distance.COSINE),
            },
        )

    @staticmethod
    def _init_sqlite(db_path: Path) -> None:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    commit_sha TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _save_snapshot(self, commit_sha: str, units: list[UnifiedCodeUnit]) -> None:
        if self._db_path is None:
            return
        payload = json.dumps([u.model_dump(mode="json") for u in units])
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO snapshots (commit_sha, payload_json) VALUES (?, ?)",
                (commit_sha, payload),
            )
            conn.commit()
