from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from .audit import (
    AUDIT_EVENT_SAMPLE_ADDED,
    AUDIT_EVENT_SAMPLE_STATUS_CHANGED,
    AUDIT_EVENT_SKU_CREATED,
)
from .config import StorageConfig
from .schemas import (
    AuditRecord,
    FeatureBundle,
    FeatureRecord,
    QualityResult,
    SKU,
    Sample,
)


class LocalRepository:
    def __init__(self, config: StorageConfig) -> None:
        self.config = config
        self.config.root.mkdir(parents=True, exist_ok=True)
        self.config.metadata_root.mkdir(parents=True, exist_ok=True)
        self.config.feature_root.mkdir(parents=True, exist_ok=True)
        self.config.patch_token_root.mkdir(parents=True, exist_ok=True)
        self._db_path = self.config.database_path
        self._initialize_database()

    def create_sku(self, sku_name: str, created_by: str = "system", status: str = "active") -> SKU:
        sku = SKU(
            sku_id=self.generate_sku_id(),
            sku_name=sku_name,
            status=status,
            created_by=created_by,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO skus (sku_id, sku_name, status, created_by, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (sku.sku_id, sku.sku_name, sku.status, sku.created_by, sku.created_at, sku.updated_at),
            )
            self._insert_audit(
                conn,
                event_type=AUDIT_EVENT_SKU_CREATED,
                entity_type="sku",
                entity_id=sku.sku_id,
                actor=created_by,
                payload={"sku_name": sku_name, "status": status},
            )
        return sku

    def get_sku(self, sku_id: str) -> SKU | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM skus WHERE sku_id = ?", (sku_id,)).fetchone()
        return self._sku_from_row(row) if row else None

    def add_sample(
        self,
        sku_id: str,
        image_path: str,
        roi_points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        quality: QualityResult,
        sample_type: str = "register",
        created_by: str = "system",
        status: str = "active",
        source_task_id: str | None = None,
    ) -> Sample:
        sample = Sample(
            sample_id=self.generate_sample_id(),
            sku_id=sku_id,
            image_path=image_path,
            roi_points=roi_points,
            quality_score=quality.score,
            quality_status=quality.status,
            sample_type=sample_type,
            status=status,
            source_task_id=source_task_id,
            created_by=created_by,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO samples (
                    sample_id, sku_id, image_path, roi_box, roi_version, quality_score, quality_status,
                    sample_type, status, source_task_id, created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample.sample_id,
                    sample.sku_id,
                    sample.image_path,
                    json.dumps([list(point) for point in sample.roi_points]),
                    sample.roi_version,
                    sample.quality_score,
                    sample.quality_status,
                    sample.sample_type,
                    sample.status,
                    sample.source_task_id,
                    sample.created_by,
                    sample.created_at,
                    sample.updated_at,
                ),
            )
            self._insert_audit(
                conn,
                event_type=AUDIT_EVENT_SAMPLE_ADDED,
                entity_type="sample",
                entity_id=sample.sample_id,
                actor=created_by,
                payload={"sku_id": sku_id, "sample_type": sample_type, "status": status},
            )
        return sample

    def update_sample_status(self, sample_id: str, status: str, actor: str = "system") -> Sample:
        with self._connect() as conn:
            conn.execute(
                "UPDATE samples SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE sample_id = ?",
                (status, sample_id),
            )
            self._insert_audit(
                conn,
                event_type=AUDIT_EVENT_SAMPLE_STATUS_CHANGED,
                entity_type="sample",
                entity_id=sample_id,
                actor=actor,
                payload={"status": status},
            )
            row = conn.execute("SELECT * FROM samples WHERE sample_id = ?", (sample_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown sample_id: {sample_id}")
        return self._sample_from_row(row)

    def get_sample(self, sample_id: str) -> Sample | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM samples WHERE sample_id = ?", (sample_id,)).fetchone()
        return self._sample_from_row(row) if row else None

    def list_samples_by_sku(self, sku_id: str, include_inactive: bool = False) -> list[Sample]:
        query = "SELECT * FROM samples WHERE sku_id = ?"
        params: list[object] = [sku_id]
        if not include_inactive:
            query += " AND status = ?"
            params.append("active")
        query += " ORDER BY sample_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._sample_from_row(row) for row in rows]

    def list_samples(self, include_inactive: bool = False) -> list[Sample]:
        query = "SELECT * FROM samples"
        params: list[object] = []
        if not include_inactive:
            query += " WHERE status = ?"
            params.append("active")
        query += " ORDER BY sample_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._sample_from_row(row) for row in rows]

    def save_feature_bundle(
        self,
        sample: Sample,
        bundle: FeatureBundle,
        feature_version: str,
    ) -> FeatureRecord:
        global_path = self.config.feature_root / f"{sample.sample_id}.npy"
        patch_path = self.config.patch_token_root / f"{sample.sample_id}.npy"
        np.save(global_path, bundle.global_embedding.astype(np.float32))
        np.save(patch_path, bundle.patch_tokens.astype(np.float32))
        record = FeatureRecord(
            sample_id=sample.sample_id,
            sku_id=sample.sku_id,
            feature_version=feature_version,
            global_embedding_path=str(global_path),
            patch_token_path=str(patch_path),
            backend=bundle.backend,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO features (
                    sample_id, sku_id, feature_version, global_embedding_path, patch_token_path, backend
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.sample_id,
                    record.sku_id,
                    record.feature_version,
                    record.global_embedding_path,
                    record.patch_token_path,
                    record.backend,
                ),
            )
        return record

    def get_feature_record(self, sample_id: str) -> FeatureRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM features WHERE sample_id = ?", (sample_id,)).fetchone()
        return FeatureRecord(**dict(row)) if row else None

    def list_feature_records(self, active_only: bool = True) -> list[FeatureRecord]:
        query = """
            SELECT f.* FROM features f
            JOIN samples s ON s.sample_id = f.sample_id
        """
        params: list[object] = []
        if active_only:
            query += " WHERE s.status = ?"
            params.append("active")
        query += " ORDER BY f.sample_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [FeatureRecord(**dict(row)) for row in rows]

    def load_feature_bundle(self, record: FeatureRecord) -> FeatureBundle:
        return FeatureBundle(
            global_embedding=np.load(record.global_embedding_path).astype(np.float32),
            patch_tokens=np.load(record.patch_token_path).astype(np.float32),
            backend=record.backend,
        )

    def load_feature_bundle_by_sample(self, sample_id: str) -> FeatureBundle:
        record = self.get_feature_record(sample_id)
        if record is None:
            raise KeyError(f"Unknown sample_id: {sample_id}")
        return self.load_feature_bundle(record)

    def list_audit_records(self) -> list[AuditRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM audits ORDER BY audit_id").fetchall()
        return [self._audit_from_row(row) for row in rows]

    def generate_sku_id(self) -> str:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM skus").fetchone()[0] + 1
        return f"sku-{count:06d}"

    def generate_sample_id(self) -> str:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] + 1
        return f"sample-{count:06d}"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_database(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS skus (
                    sku_id TEXT PRIMARY KEY,
                    sku_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS samples (
                    sample_id TEXT PRIMARY KEY,
                    sku_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    roi_box TEXT NOT NULL,
                    roi_version TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    quality_status TEXT NOT NULL,
                    sample_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_task_id TEXT,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS features (
                    sample_id TEXT PRIMARY KEY,
                    sku_id TEXT NOT NULL,
                    feature_version TEXT NOT NULL,
                    global_embedding_path TEXT NOT NULL,
                    patch_token_path TEXT NOT NULL,
                    backend TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audits (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def _insert_audit(
        self,
        conn: sqlite3.Connection,
        *,
        event_type: str,
        entity_type: str,
        entity_id: str,
        actor: str,
        payload: dict,
    ) -> None:
        conn.execute(
            """
            INSERT INTO audits (event_type, entity_type, entity_id, actor, payload)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_type, entity_type, entity_id, actor, json.dumps(payload)),
        )

    def _sku_from_row(self, row: sqlite3.Row) -> SKU:
        return SKU(**dict(row))

    def _sample_from_row(self, row: sqlite3.Row) -> Sample:
        payload = dict(row)
        payload["roi_points"] = tuple(tuple(point) for point in json.loads(payload["roi_box"]))
        payload.pop("roi_box")
        return Sample(**payload)

    def _audit_from_row(self, row: sqlite3.Row) -> AuditRecord:
        payload = dict(row)
        return AuditRecord(
            audit_id=payload["audit_id"],
            event_type=payload["event_type"],
            entity_type=payload["entity_type"],
            entity_id=payload["entity_id"],
            actor=payload["actor"],
            payload=json.loads(payload["payload"]),
            created_at=payload["created_at"],
        )
