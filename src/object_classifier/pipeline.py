from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .config import PipelineConfig
from .decision import aggregate_candidates, decide_top_candidate
from .features import BaseFeatureBackend, load_feature_cache, save_feature_cache
from .index import SampleIndex
from .quality import assess_quality
from .repository import LocalRepository
from .rerank import rerank_candidates
from .roi import normalize_roi
from .schemas import DecisionResult, ManualReviewPayload, RegistrationResult


class ObjectClassifierPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        backend: BaseFeatureBackend,
        repository: LocalRepository | None = None,
        index: SampleIndex | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.repository = repository or LocalRepository(config.storage)
        self.index = index or SampleIndex()
        self._rebuild_index()

    def register(self, sku_name: str, image_paths: list[Path]) -> RegistrationResult:
        sku = self.repository.create_sku(sku_name)
        samples = []
        warnings: list[str] = []
        for image_path in image_paths:
            roi = self._load_normalized_roi(Path(image_path))
            quality = assess_quality(roi.image, self.config.quality)
            if quality.status == "hard_fail":
                raise ValueError(f"Registration image failed quality checks: {quality.reasons}")
            if quality.status == "soft_fail":
                warnings.extend(quality.reasons)
            bundle = self._extract_features(roi)
            sample = self.repository.add_sample(
                sku_id=sku.sku_id,
                image_path=str(image_path),
                roi_box=roi.roi_box,
                quality=quality,
            )
            self.repository.save_feature_bundle(sample, bundle, feature_version="p0")
            samples.append(sample)
        self._rebuild_index()
        return RegistrationResult(sku=sku, samples=samples, warnings=warnings)

    def identify(self, image_path: Path) -> DecisionResult:
        roi = self._load_normalized_roi(Path(image_path))
        quality = assess_quality(roi.image, self.config.quality)
        if quality.status == "hard_fail":
            return self._manual_review_result(str(image_path), [], quality, ["quality_hard_fail"])

        bundle = self._extract_features(roi)
        recalled = self.index.search_topk(bundle.global_embedding, k=self.config.topk)
        reranked = rerank_candidates(
            bundle.patch_tokens,
            recalled,
            self.repository.load_feature_bundle_by_sample,
        )
        aggregated = aggregate_candidates(reranked)
        result = decide_top_candidate(aggregated, self.config.decision)

        if quality.status == "soft_fail":
            reasons = result.reasons + ["quality_soft_fail"]
            return self._manual_review_result(str(image_path), result.candidates, quality, reasons)
        if result.decision == "manual_review":
            return self._manual_review_result(str(image_path), result.candidates, quality, result.reasons)
        return result

    def _extract_features(self, roi):
        cache_path = self._cache_path(roi)
        if self.config.cache.enabled and cache_path.exists():
            return load_feature_cache(cache_path)
        bundle = self.backend.extract(roi)
        if self.config.cache.enabled:
            save_feature_cache(cache_path, bundle)
        return bundle

    def _load_normalized_roi(self, image_path: Path):
        with Image.open(image_path) as image:
            return normalize_roi(
                image=image,
                roi_box=self.config.model.roi_box,
                output_size=self.config.model.input_size,
                min_size=(self.config.quality.min_width, self.config.quality.min_height),
                source_path=str(image_path),
            )

    def _cache_path(self, roi) -> Path:
        return self.config.cache.cache_dir / f"{self.backend.cache_key(roi)}.npz"

    def _manual_review_result(self, image_path: str, candidates, quality, reasons) -> DecisionResult:
        payload = ManualReviewPayload(
            image_path=image_path,
            candidates=list(candidates),
            quality=quality,
            query_metadata={"backend": self.backend.backend_name},
        )
        return DecisionResult(
            decision="manual_review",
            status="manual_review",
            sku_id=None,
            top_candidate=candidates[0] if candidates else None,
            candidates=list(candidates),
            reasons=list(reasons),
            metadata={"manual_review": json.loads(json.dumps(payload, default=lambda item: item.__dict__))},
        )

    def _rebuild_index(self) -> None:
        records = self.repository.list_feature_records()
        vectors = [self.repository.load_feature_bundle(record).global_embedding for record in records]
        self.index.rebuild(records, vectors)
