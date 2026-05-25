from __future__ import annotations
from pathlib import Path

from PIL import Image

from .config import PipelineConfig
from .decision import aggregate_candidates, decide_registration_candidate, decide_top_candidate
from .features import BaseFeatureBackend, load_feature_cache, save_feature_cache
from .index import SampleIndex
from .quality import assess_quality
from .repository import LocalRepository
from .rerank import rerank_candidates
from .roi import normalize_roi
from .schemas import DecisionResult, RegistrationResult


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

    def register(self, sku_name: str, image_paths: list[Path], created_by: str = "system") -> RegistrationResult:
        if not image_paths:
            raise ValueError("Registration requires at least one image")

        contexts, warnings = self._prepare_contexts(image_paths)
        first_context = contexts[0]
        candidates = self._search_candidates(first_context["bundle"])
        reasons = decide_registration_candidate(candidates, self.config.decision)

        sku = self.repository.create_sku(sku_name, created_by=created_by)
        samples = self._persist_contexts(sku.sku_id, contexts, created_by=created_by, sample_type="register")
        return RegistrationResult(
            decision="safe_create",
            sku=sku,
            samples=samples,
            warnings=warnings,
            candidates=candidates,
            reasons=["created_new_sku", *reasons],
            metadata={"backend": self.backend.backend_name},
        )

    def identify(self, image_path: Path, created_by: str = "system") -> DecisionResult:
        roi = self._load_normalized_roi(Path(image_path))
        quality = assess_quality(roi.image, self.config.quality)
        if quality.status == "hard_fail":
            return DecisionResult(
                decision="best_effort",
                status="quality_rejected",
                sku_id=None,
                top_candidate=None,
                candidates=[],
                reasons=["quality_hard_fail"],
                metadata={"quality": quality, "backend": self.backend.backend_name},
            )

        bundle = self._extract_features(roi)
        aggregated = self._search_candidates(bundle)
        result = decide_top_candidate(aggregated, self.config.decision)

        if quality.status == "soft_fail":
            return DecisionResult(
                decision="best_effort",
                status="quality_warning",
                sku_id=None,
                top_candidate=result.top_candidate,
                candidates=result.candidates,
                reasons=[*result.reasons, "quality_soft_fail"],
                metadata={"quality": quality, "backend": self.backend.backend_name},
            )
        return result

    def _prepare_contexts(self, image_paths: list[Path]):
        contexts = []
        warnings: list[str] = []
        for image_path in image_paths:
            roi = self._load_normalized_roi(Path(image_path))
            quality = assess_quality(roi.image, self.config.quality)
            if quality.status == "hard_fail":
                raise ValueError(f"Registration image failed quality checks: {quality.reasons}")
            if quality.status == "soft_fail":
                warnings.extend(quality.reasons)
            bundle = self._extract_features(roi)
            contexts.append({"image_path": image_path, "roi": roi, "quality": quality, "bundle": bundle})
        return contexts, warnings

    def _persist_contexts(
        self,
        sku_id: str,
        contexts: list[dict],
        *,
        created_by: str,
        sample_type: str = "register",
    ):
        samples = []
        for context in contexts:
            sample = self.repository.add_sample(
                sku_id=sku_id,
                image_path=str(context["image_path"]),
                roi_points=context["roi"].roi_points,
                quality=context["quality"],
                sample_type=sample_type,
                created_by=created_by,
            )
            record = self.repository.save_feature_bundle(sample, context["bundle"], feature_version="p1")
            samples.append(sample)
            self._append_index(record, context["bundle"].global_embedding)
        return samples

    def _search_candidates(self, bundle):
        recalled = self.index.search_topk(bundle.global_embedding, k=self.config.topk)
        reranked = rerank_candidates(
            bundle.patch_tokens,
            recalled,
            self.repository.load_feature_bundle_by_sample,
        )
        return aggregate_candidates(reranked)

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
                min_size=(self.config.quality.min_width, self.config.quality.min_height),
                source_path=str(image_path),
            )

    def _cache_path(self, roi) -> Path:
        return self.config.cache.cache_dir / f"{self.backend.cache_key(roi)}.npz"

    def _append_index(self, record, vector) -> None:
        try:
            self.index.append(record, vector)
        except Exception:
            self.index.mark_stale()
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        records = self.repository.list_feature_records(active_only=True)
        vectors = [self.repository.load_feature_bundle(record).global_embedding for record in records]
        self.index.rebuild(records, vectors)
