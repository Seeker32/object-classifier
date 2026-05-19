from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import StorageConfig
from .schemas import FeatureBundle, FeatureRecord, QualityResult, SKU, Sample


class LocalRepository:
    def __init__(self, config: StorageConfig) -> None:
        self.config = config
        self.config.root.mkdir(parents=True, exist_ok=True)
        self.config.metadata_root.mkdir(parents=True, exist_ok=True)
        self.config.feature_root.mkdir(parents=True, exist_ok=True)
        self.config.patch_token_root.mkdir(parents=True, exist_ok=True)
        self._skus_file = self.config.metadata_root / "skus.json"
        self._samples_file = self.config.metadata_root / "samples.json"
        self._features_file = self.config.metadata_root / "features.json"
        for path in (self._skus_file, self._samples_file, self._features_file):
            if not path.exists():
                path.write_text("[]", encoding="utf-8")

    def create_sku(self, sku_name: str, created_by: str = "system") -> SKU:
        sku = SKU(
            sku_id=self.generate_sku_id(),
            sku_name=sku_name,
            created_by=created_by,
        )
        payload = self._load_json(self._skus_file)
        payload.append(asdict(sku))
        self._write_json(self._skus_file, payload)
        return sku

    def get_sku(self, sku_id: str) -> SKU | None:
        for row in self._load_json(self._skus_file):
            if row["sku_id"] == sku_id:
                return SKU(**row)
        return None

    def add_sample(
        self,
        sku_id: str,
        image_path: str,
        roi_box: tuple[int, int, int, int],
        quality: QualityResult,
        sample_type: str = "register",
        created_by: str = "system",
    ) -> Sample:
        sample = Sample(
            sample_id=self.generate_sample_id(),
            sku_id=sku_id,
            image_path=image_path,
            roi_box=roi_box,
            quality_score=quality.score,
            quality_status=quality.status,
            sample_type=sample_type,
            created_by=created_by,
        )
        payload = self._load_json(self._samples_file)
        payload.append(asdict(sample))
        self._write_json(self._samples_file, payload)
        return sample

    def get_sample(self, sample_id: str) -> Sample | None:
        for row in self._load_json(self._samples_file):
            if row["sample_id"] == sample_id:
                return self._sample_from_row(row)
        return None

    def list_samples_by_sku(self, sku_id: str) -> list[Sample]:
        return [
            self._sample_from_row(row)
            for row in self._load_json(self._samples_file)
            if row["sku_id"] == sku_id
        ]

    def list_samples(self) -> list[Sample]:
        return [self._sample_from_row(row) for row in self._load_json(self._samples_file)]

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
        payload = self._load_json(self._features_file)
        payload.append(asdict(record))
        self._write_json(self._features_file, payload)
        return record

    def get_feature_record(self, sample_id: str) -> FeatureRecord | None:
        for row in self._load_json(self._features_file):
            if row["sample_id"] == sample_id:
                return FeatureRecord(**row)
        return None

    def list_feature_records(self) -> list[FeatureRecord]:
        return [FeatureRecord(**row) for row in self._load_json(self._features_file)]

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

    def generate_sku_id(self) -> str:
        count = len(self._load_json(self._skus_file)) + 1
        return f"sku-{count:06d}"

    def generate_sample_id(self) -> str:
        count = len(self._load_json(self._samples_file)) + 1
        return f"sample-{count:06d}"

    def _load_json(self, path: Path) -> list[dict]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: list[dict]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _sample_from_row(self, row: dict) -> Sample:
        payload = dict(row)
        payload["roi_box"] = tuple(payload["roi_box"])
        return Sample(**payload)
