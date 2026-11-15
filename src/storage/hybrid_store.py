"""Layer 5: Tri-modal hybrid store (Tantivy + Qdrant + Property Graph) with RRF retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_OID, UUID, uuid5

import networkx as nx
import tantivy
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

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
        self._point_ids: dict[str, str] = {}
        self._lock = threading.RLock()

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
        if self._db_path is not None:
            self.recover_from_sqlite()

    def _connect_sqlite(self) -> sqlite3.Connection:
        """Open a SQLite connection configured with WAL mode and busy timeout."""
        assert self._db_path is not None
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def close(self) -> None:
        """Close underlying storage clients (releasing local Qdrant file locks)."""
        with self._lock:
            self._qdrant.close()

    @staticmethod
    def _unit_point_id(unit_id: str) -> str:
        """Derive a deterministic 128-bit UUID Qdrant point ID from a unit_id."""
        if len(unit_id) >= 32:
            try:
                return str(UUID(unit_id[:32]))
            except ValueError:
                pass
        return str(uuid5(NAMESPACE_OID, unit_id))

    def _prepare_unit_index_data(
        self, unit: UnifiedCodeUnit
    ) -> tuple[tantivy.Document, dict[str, list[float]], str, PointStruct]:
        """Build Tantivy document, modality vectors, point ID, and Qdrant point for a unit."""
        ast = unit.ast_chunk
        ir_tokens = unit.ir_graph.canonical_tokens if unit.ir_graph else []
        asm_tokens = [
            insn for bb in unit.binary_blocks for insn in bb.normalized_instructions
        ]
        doc = tantivy.Document(
            unit_id=unit.unit_id,
            symbol_name=ast.symbol_name or "anonymous",
            file_path=ast.file_path,
            source_tokens=" ".join(ast.lexical_tokens),
            ir_tokens=" ".join(ir_tokens),
            asm_tokens=" ".join(asm_tokens),
        )
        ast_vec = _hash_feature_vector(
            [*ast.lexical_tokens, *split_identifier_tokens(ast.scope_header)],
            VECTOR_DIM_AST,
        )
        ir_vec = compute_wl_ir_vector(unit.ir_graph)
        bin_vec = compute_binary_cfg_vector(unit.binary_blocks)
        vectors = {
            "ast_dense": ast_vec,
            "ir_structural": ir_vec,
            "bin_cfg": bin_vec,
        }
        pid = self._unit_point_id(unit.unit_id)
        point = PointStruct(
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
        return doc, vectors, pid, point

    def _add_units_to_memory(
        self,
        staged_units: dict[str, UnifiedCodeUnit],
        staged_vectors: dict[str, dict[str, list[float]]],
        staged_point_ids: dict[str, str],
        commit_sha: str,
    ) -> None:
        """Update in-memory unit maps and property graph for upserted units."""
        for uid, unit in staged_units.items():
            self.units[uid] = unit
            self.unit_vectors[uid] = staged_vectors[uid]
            self._point_ids[uid] = staged_point_ids[uid]
            ast = unit.ast_chunk
            self.graph.add_node(
                uid,
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

    def upsert_units(
        self, units: list[UnifiedCodeUnit], commit_sha: str = "working-tree"
    ) -> int:
        """Atomically index a batch of UnifiedCodeUnits across all three modalities."""
        with self._lock:
            return self._upsert_units_locked(units, commit_sha=commit_sha)

    def _upsert_units_locked(
        self, units: list[UnifiedCodeUnit], commit_sha: str = "working-tree"
    ) -> int:
        if not units:
            return 0

        writer = self._tantivy_index.writer()
        qdrant_points: list[PointStruct] = []
        staged_units: dict[str, UnifiedCodeUnit] = {}
        staged_vectors: dict[str, dict[str, list[float]]] = {}
        staged_point_ids: dict[str, str] = {}

        for unit in units:
            staged_units[unit.unit_id] = unit
            doc, vectors, pid, point = self._prepare_unit_index_data(unit)
            writer.delete_documents_by_term("unit_id", unit.unit_id)
            writer.add_document(doc)
            staged_vectors[unit.unit_id] = vectors
            staged_point_ids[unit.unit_id] = pid
            qdrant_points.append(point)

        writer.commit()
        self._tantivy_index.reload()
        self._qdrant.upsert(collection_name=self.COLLECTION_NAME, points=qdrant_points)
        if self._db_path is not None:
            self._save_snapshot(commit_sha, units)
        self._add_units_to_memory(staged_units, staged_vectors, staged_point_ids, commit_sha)
        return len(units)

    def delete_units(self, unit_ids: list[str]) -> int:
        """Delete stale/removed units across SQLite, Tantivy, Qdrant, and in-memory state."""
        with self._lock:
            return self._delete_units_locked(unit_ids)

    def _delete_units_locked(self, unit_ids: list[str]) -> int:
        if not unit_ids:
            return 0

        if self._db_path is not None:
            with self._connect_sqlite() as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO tombstones (unit_id) VALUES (?)",
                    [(u,) for u in unit_ids],
                )
                conn.executemany(
                    "DELETE FROM units WHERE unit_id = ?",
                    [(u,) for u in unit_ids],
                )

        writer = self._tantivy_index.writer()
        point_ids: list[int | str | UUID] = []
        for uid in unit_ids:
            writer.delete_documents_by_term("unit_id", uid)
            point_ids.append(self._point_ids.get(uid, self._unit_point_id(uid)))
        writer.commit()
        self._tantivy_index.reload()
        if point_ids:
            self._qdrant.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=PointIdsList(points=point_ids),
            )

        removed = 0
        for uid in unit_ids:
            self._point_ids.pop(uid, None)
            if uid in self.units:
                self.units.pop(uid, None)
                self.unit_vectors.pop(uid, None)
                if uid in self.graph:
                    self.graph.remove_node(uid)
                removed += 1

        if self._db_path is not None:
            with self._connect_sqlite() as conn:
                conn.executemany(
                    "DELETE FROM tombstones WHERE unit_id = ?",
                    [(u,) for u in unit_ids],
                )
        return removed

    def apply_batch(
        self,
        upsert_units: list[UnifiedCodeUnit],
        delete_unit_ids: list[str],
        save_file_states: dict[str, tuple[str, list[str]]] | None = None,
        delete_file_paths: list[str] | None = None,
        commit_sha: str = "working-tree",
    ) -> tuple[int, int]:
        """Atomically apply upserts, deletes, and file_state updates across all stores."""
        with self._lock:
            return self._apply_batch_locked(
                upsert_units=upsert_units,
                delete_unit_ids=delete_unit_ids,
                save_file_states=save_file_states,
                delete_file_paths=delete_file_paths,
                commit_sha=commit_sha,
            )

    def _apply_batch_locked(
        self,
        upsert_units: list[UnifiedCodeUnit],
        delete_unit_ids: list[str],
        save_file_states: dict[str, tuple[str, list[str]]] | None = None,
        delete_file_paths: list[str] | None = None,
        commit_sha: str = "working-tree",
    ) -> tuple[int, int]:
        if (
            not upsert_units
            and not delete_unit_ids
            and not save_file_states
            and not delete_file_paths
        ):
            return 0, 0

        # Phase 1: persist tombstones before mutating secondary search stores.
        if self._db_path is not None and delete_unit_ids:
            with self._connect_sqlite() as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO tombstones (unit_id) VALUES (?)",
                    [(u,) for u in delete_unit_ids],
                )

        staged_units: dict[str, UnifiedCodeUnit] = {}
        staged_vectors: dict[str, dict[str, list[float]]] = {}
        staged_point_ids: dict[str, str] = {}
        qdrant_points: list[PointStruct] = []
        delete_point_ids: list[int | str | UUID] = []

        # Phase 2: update Tantivy and Qdrant; if this raises, SQLite units/file_state remain clean.
        if delete_unit_ids or upsert_units:
            writer = self._tantivy_index.writer()
            for uid in delete_unit_ids:
                writer.delete_documents_by_term("unit_id", uid)
                delete_point_ids.append(self._point_ids.get(uid, self._unit_point_id(uid)))
            for unit in upsert_units:
                staged_units[unit.unit_id] = unit
                doc, vectors, pid, point = self._prepare_unit_index_data(unit)
                writer.delete_documents_by_term("unit_id", unit.unit_id)
                writer.add_document(doc)
                staged_vectors[unit.unit_id] = vectors
                staged_point_ids[unit.unit_id] = pid
                qdrant_points.append(point)
            writer.commit()
            self._tantivy_index.reload()

            if delete_point_ids:
                self._qdrant.delete(
                    collection_name=self.COLLECTION_NAME,
                    points_selector=PointIdsList(points=delete_point_ids),
                )
            if qdrant_points:
                self._qdrant.upsert(collection_name=self.COLLECTION_NAME, points=qdrant_points)

        # Phase 3: commit SQLite units, snapshot summary, file_state, and tombstones atomically.
        if self._db_path is not None:
            with self._connect_sqlite() as conn:
                if delete_unit_ids:
                    conn.executemany(
                        "DELETE FROM units WHERE unit_id = ?",
                        [(u,) for u in delete_unit_ids],
                    )
                    for idx in range(0, len(delete_unit_ids), 500):
                        batch = delete_unit_ids[idx : idx + 500]
                        placeholders = ",".join("?" for _ in batch)
                        conn.execute(
                            f"DELETE FROM tombstones WHERE unit_id IN ({placeholders})",
                            batch,
                        )
                if upsert_units:
                    upserted_ids = [u.unit_id for u in upsert_units]
                    payload = json.dumps(
                        {"unit_count": len(upsert_units), "unit_ids": upserted_ids}
                    )
                    unit_rows = [(u.unit_id, commit_sha, u.model_dump_json()) for u in upsert_units]
                    conn.execute(
                        "INSERT OR REPLACE INTO snapshots (commit_sha, payload_json) VALUES (?, ?)",
                        (commit_sha, payload),
                    )
                    conn.executemany(
                        "INSERT OR REPLACE INTO units (unit_id, commit_sha, unit_json)"
                        " VALUES (?, ?, ?)",
                        unit_rows,
                    )
                    for idx in range(0, len(upserted_ids), 500):
                        batch = upserted_ids[idx : idx + 500]
                        placeholders = ",".join("?" for _ in batch)
                        conn.execute(
                            f"DELETE FROM tombstones WHERE unit_id IN ({placeholders})",
                            batch,
                        )
                if save_file_states:
                    file_rows = [
                        (fp, content_hash, json.dumps(unit_ids))
                        for fp, (content_hash, unit_ids) in save_file_states.items()
                    ]
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO file_state (file_path, content_hash, unit_ids_json)
                        VALUES (?, ?, ?)
                        """,
                        file_rows,
                    )
                if delete_file_paths:
                    conn.executemany(
                        "DELETE FROM file_state WHERE file_path = ?",
                        [(fp,) for fp in delete_file_paths],
                    )

        # Phase 4: update in-memory maps and property graph.
        removed_count = 0
        for uid in delete_unit_ids:
            self._point_ids.pop(uid, None)
            if uid in self.units:
                self.units.pop(uid, None)
                self.unit_vectors.pop(uid, None)
                if uid in self.graph:
                    self.graph.remove_node(uid)
                removed_count += 1

        if staged_units:
            self._add_units_to_memory(staged_units, staged_vectors, staged_point_ids, commit_sha)

        return len(upsert_units), removed_count

    def recover_from_sqlite(self) -> int:
        """Reload units from SQLite, replay tombstones, and reconcile Tantivy/Qdrant."""
        with self._lock:
            return self._recover_from_sqlite_locked()

    def _recover_from_sqlite_locked(self) -> int:
        if self._db_path is None or not self._db_path.exists():
            return 0
        with self._connect_sqlite() as conn:
            tombstone_rows = conn.execute("SELECT unit_id FROM tombstones").fetchall()
            unit_rows = conn.execute("SELECT unit_id, commit_sha, unit_json FROM units").fetchall()

        tombstone_ids = [str(r[0]) for r in tombstone_rows]
        recovered = [UnifiedCodeUnit.model_validate_json(r[2]) for r in unit_rows]
        active_ids = {u.unit_id for u in recovered}

        if tombstone_ids:
            writer = self._tantivy_index.writer()
            for uid in tombstone_ids:
                writer.delete_documents_by_term("unit_id", uid)
            writer.commit()
            self._tantivy_index.reload()
            self._qdrant.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=PointIdsList(
                    points=[self._unit_point_id(uid) for uid in tombstone_ids]
                ),
            )
            with self._connect_sqlite() as conn:
                conn.executemany(
                    "DELETE FROM tombstones WHERE unit_id = ?",
                    [(uid,) for uid in tombstone_ids],
                )

        self._reconcile_stray_ids(active_ids)

        for stale_uid in [uid for uid in self.units if uid not in active_ids]:
            self.units.pop(stale_uid, None)
            self.unit_vectors.pop(stale_uid, None)
            self._point_ids.pop(stale_uid, None)
            if stale_uid in self.graph:
                self.graph.remove_node(stale_uid)

        if not recovered:
            return 0

        tantivy_count = self._tantivy_index.searcher().num_docs
        qdrant_count = int(self._qdrant.get_collection(self.COLLECTION_NAME).points_count or 0)
        needs_secondary_sync = (
            bool(tombstone_ids)
            or tantivy_count != len(recovered)
            or qdrant_count != len(recovered)
        )

        staged_units: dict[str, UnifiedCodeUnit] = {}
        staged_vectors: dict[str, dict[str, list[float]]] = {}
        staged_point_ids: dict[str, str] = {}
        qdrant_points: list[PointStruct] = []
        maybe_writer: tantivy.IndexWriter | None = (
            self._tantivy_index.writer() if needs_secondary_sync else None
        )

        for unit in recovered:
            staged_units[unit.unit_id] = unit
            doc, vectors, pid, point = self._prepare_unit_index_data(unit)
            staged_vectors[unit.unit_id] = vectors
            staged_point_ids[unit.unit_id] = pid
            if maybe_writer is not None:
                maybe_writer.delete_documents_by_term("unit_id", unit.unit_id)
                maybe_writer.add_document(doc)
                qdrant_points.append(point)

        if maybe_writer is not None:
            maybe_writer.commit()
            self._tantivy_index.reload()
            self._qdrant.upsert(collection_name=self.COLLECTION_NAME, points=qdrant_points)

        commit_sha = str(unit_rows[0][1]) if unit_rows else "working-tree"
        self._add_units_to_memory(staged_units, staged_vectors, staged_point_ids, commit_sha)
        return len(recovered)

    def _reconcile_stray_ids(self, active_ids: set[str]) -> None:
        """Delete any Tantivy documents or Qdrant points not present in SQLite active_ids."""
        searcher = self._tantivy_index.searcher()
        if searcher.num_docs > 0:
            all_hits = searcher.search(tantivy.Query.all_query(), searcher.num_docs).hits
            stray_tantivy_ids: set[str] = set()
            for _, doc_addr in all_hits:
                vals = searcher.doc(doc_addr).get_all("unit_id")
                if vals:
                    uid = str(vals[0])
                    if uid not in active_ids:
                        stray_tantivy_ids.add(uid)
            if stray_tantivy_ids:
                writer = self._tantivy_index.writer()
                for uid in stray_tantivy_ids:
                    writer.delete_documents_by_term("unit_id", uid)
                writer.commit()
                self._tantivy_index.reload()

        stray_qdrant_ids: list[int | str | UUID] = []
        offset: Any = None
        while True:
            points, next_offset = self._qdrant.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                payload = pt.payload or {}
                uid = str(payload.get("unit_id", ""))
                if not uid or uid not in active_ids:
                    stray_qdrant_ids.append(pt.id)
            if next_offset is None:
                break
            offset = next_offset
        if stray_qdrant_ids:
            self._qdrant.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=PointIdsList(points=stray_qdrant_ids),
            )

    def save_file_state(self, updates: dict[str, tuple[str, list[str]]]) -> None:
        """Persist per-file content hashes and indexed unit IDs in SQLite."""
        with self._lock:
            if self._db_path is None or not updates:
                return
            rows = [
                (file_path, content_hash, json.dumps(unit_ids))
                for file_path, (content_hash, unit_ids) in updates.items()
            ]
            with self._connect_sqlite() as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO file_state (file_path, content_hash, unit_ids_json)
                    VALUES (?, ?, ?)
                    """,
                    rows,
                )

    def delete_file_state(self, file_paths: list[str]) -> None:
        """Remove persisted file state entries for deleted files."""
        with self._lock:
            if self._db_path is None or not file_paths:
                return
            with self._connect_sqlite() as conn:
                conn.executemany(
                    "DELETE FROM file_state WHERE file_path = ?",
                    [(fp,) for fp in file_paths],
                )

    def load_file_state(self) -> dict[str, tuple[str, list[str]]]:
        """Load persisted per-file content hashes and unit IDs from SQLite."""
        with self._lock:
            if self._db_path is None or not self._db_path.exists():
                return {}
            with self._connect_sqlite() as conn:
                rows = conn.execute(
                    "SELECT file_path, content_hash, unit_ids_json FROM file_state"
                ).fetchall()
            state: dict[str, tuple[str, list[str]]] = {}
            for file_path, content_hash, unit_ids_json in rows:
                raw_ids = json.loads(str(unit_ids_json))
                unit_ids = [str(uid) for uid in raw_ids] if isinstance(raw_ids, list) else []
                state[str(file_path)] = (str(content_hash), unit_ids)
            return state

    def verify_consistency(self) -> dict[str, Any]:
        """Check document/point/row counts across in-memory, Tantivy, Qdrant, and SQLite."""
        with self._lock:
            return self._verify_consistency_locked()

    def _verify_consistency_locked(self) -> dict[str, Any]:
        tantivy_count = self._tantivy_index.searcher().num_docs
        qdrant_info = self._qdrant.get_collection(self.COLLECTION_NAME)
        qdrant_count = int(qdrant_info.points_count or 0)
        sqlite_count = len(self.units)
        if self._db_path is not None and self._db_path.exists():
            with self._connect_sqlite() as conn:
                row = conn.execute("SELECT COUNT(*) FROM units").fetchone()
                sqlite_count = int(row[0]) if row else 0
        mem_count = len(self.units)
        consistent = mem_count == tantivy_count == qdrant_count == sqlite_count
        return {
            "consistent": consistent,
            "memory": mem_count,
            "tantivy": tantivy_count,
            "qdrant": qdrant_count,
            "sqlite": sqlite_count,
        }

    def health_check(self) -> dict[str, Any]:
        """Return consistency counts, pending tombstone count, and SQLite health status."""
        with self._lock:
            pending_tombstones = 0
            sqlite_ok = True
            if self._db_path is not None:
                if not self._db_path.exists():
                    sqlite_ok = False
                else:
                    try:
                        with self._connect_sqlite() as conn:
                            tomb_row = conn.execute("SELECT COUNT(*) FROM tombstones").fetchone()
                            pending_tombstones = int(tomb_row[0]) if tomb_row else 0
                            check_row = conn.execute("PRAGMA quick_check").fetchone()
                            sqlite_ok = bool(check_row and str(check_row[0]) == "ok")
                    except sqlite3.Error:
                        sqlite_ok = False
            try:
                report = self._verify_consistency_locked()
            except sqlite3.Error:
                sqlite_ok = False
                qdrant_info = self._qdrant.get_collection(self.COLLECTION_NAME)
                report = {
                    "consistent": False,
                    "memory": len(self.units),
                    "tantivy": self._tantivy_index.searcher().num_docs,
                    "qdrant": int(qdrant_info.points_count or 0),
                    "sqlite": 0,
                }
            report["pending_tombstones"] = pending_tombstones
            report["sqlite_ok"] = sqlite_ok
            return report

    def backup_sqlite(self, dest_path: str | Path) -> Path:
        """Back up the SQLite snapshot database to dest_path using sqlite3.Connection.backup."""
        with self._lock:
            if self._db_path is None:
                raise RuntimeError("Cannot backup SQLite: store has no storage_dir")
            if not self._db_path.exists():
                raise FileNotFoundError(f"SQLite database not found: {self._db_path}")
            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            src_conn = self._connect_sqlite()
            dst_conn = sqlite3.connect(dest, timeout=30.0)
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
                src_conn.close()
            return dest

    def restore_sqlite(self, src_path: str | Path) -> int:
        """Restore the SQLite database from src_path and recover all store modalities."""
        with self._lock:
            if self._db_path is None:
                raise RuntimeError("Cannot restore SQLite: store has no storage_dir")
            src = Path(src_path)
            if not src.exists():
                raise FileNotFoundError(f"Backup SQLite file not found: {src}")
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            src_conn = sqlite3.connect(src, timeout=30.0)
            dst_conn = self._connect_sqlite()
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
                src_conn.close()
            return self._recover_from_sqlite_locked()

    def rrf_search(
        self,
        query: str,
        layers: list[str] | None = None,
        top_k: int = 10,
        weights: dict[str, float] | None = None,
    ) -> list[SearchHit]:
        """Execute hybrid search with Weighted Reciprocal Rank Fusion across modalities."""
        with self._lock:
            return self._rrf_search_locked(
                query=query, layers=layers, top_k=top_k, weights=weights
            )

    def _rrf_search_locked(
        self,
        query: str,
        layers: list[str] | None = None,
        top_k: int = 10,
        weights: dict[str, float] | None = None,
    ) -> list[SearchHit]:
        # Default text queries to lexical + AST token vectors; ir_gnn/bin_cfg use structural
        # IR/ASM vocabularies and are only queried when explicitly requested in `layers`.
        active = set(layers or ["lex", "ast_vec"])
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
        with self._lock:
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
        builder.add_text_field("unit_id", stored=True, tokenizer_name="raw")
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
        with sqlite3.connect(db_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    commit_sha TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS units (
                    unit_id TEXT PRIMARY KEY,
                    commit_sha TEXT NOT NULL,
                    unit_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tombstones (
                    unit_id TEXT PRIMARY KEY
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_state (
                    file_path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    unit_ids_json TEXT NOT NULL
                )
                """
            )

    def _save_snapshot(self, commit_sha: str, units: list[UnifiedCodeUnit]) -> None:
        if self._db_path is None:
            return
        unit_ids = [u.unit_id for u in units]
        payload = json.dumps({"unit_count": len(units), "unit_ids": unit_ids})
        unit_rows = [(u.unit_id, commit_sha, u.model_dump_json()) for u in units]
        with self._connect_sqlite() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO snapshots (commit_sha, payload_json) VALUES (?, ?)",
                (commit_sha, payload),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO units (unit_id, commit_sha, unit_json) VALUES (?, ?, ?)",
                unit_rows,
            )
            for idx in range(0, len(unit_ids), 500):
                batch = unit_ids[idx : idx + 500]
                placeholders = ",".join("?" for _ in batch)
                conn.execute(
                    f"DELETE FROM tombstones WHERE unit_id IN ({placeholders})",
                    batch,
                )
