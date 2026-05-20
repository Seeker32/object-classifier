from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .config import PipelineConfig
from .decision import aggregate_candidates, decide_registration_candidate, decide_top_candidate
from .features import BaseFeatureBackend, load_feature_cache, save_feature_cache
from .index import SampleIndex
from .quality import assess_quality
from .repository import LocalRepository
from .rerank import rerank_candidates
from .review import (
    IDENTIFICATION_REVIEW_ACTIONS,
    REGISTRATION_REVIEW_ACTIONS,
    REVIEW_ACTION_ADD_HARD_CASE_SAMPLE,
    REVIEW_ACTION_BIND_EXISTING_SKU,
    REVIEW_ACTION_CREATE_NEW_SKU,
)
from .roi import normalize_roi
from .schemas import (
    DecisionResult,
    ManualReviewPayload,
    RegistrationResult,
    ReviewConfirmationResult,
)


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
        registration_decision, reasons = decide_registration_candidate(candidates, self.config.decision)

        if registration_decision != "safe_create":
            review = self.repository.create_review(
                review_type="registration",
                requested_actions=REGISTRATION_REVIEW_ACTIONS,
                image_paths=[str(path) for path in image_paths],
                candidates=candidates,
                quality=first_context["quality"],
                target_sku_name=sku_name,
                metadata={"warnings": warnings, "backend": self.backend.backend_name},
                created_by=created_by,
            )
            return RegistrationResult(
                decision=registration_decision,
                sku=None,
                samples=[],
                warnings=warnings,
                review_id=review.review_id,
                candidates=candidates,
                reasons=reasons,
                metadata={"review_type": "registration"},
            )

        sku = self.repository.create_sku(sku_name, created_by=created_by)
        samples = self._persist_contexts(sku.sku_id, contexts, created_by=created_by, sample_type="register")
        return RegistrationResult(
            decision="safe_create",
            sku=sku,
            samples=samples,
            warnings=warnings,
            reasons=["created_new_sku"],
        )

    def identify(self, image_path: Path, created_by: str = "system") -> DecisionResult:
        roi = self._load_normalized_roi(Path(image_path))
        quality = assess_quality(roi.image, self.config.quality)
        if quality.status == "hard_fail":
            return self._manual_review_result(
                review_type="identify",
                image_paths=[str(image_path)],
                candidates=[],
                quality=quality,
                reasons=["quality_hard_fail"],
                created_by=created_by,
            )

        bundle = self._extract_features(roi)
        aggregated = self._search_candidates(bundle)
        result = decide_top_candidate(aggregated, self.config.decision)

        if quality.status == "soft_fail":
            reasons = result.reasons + ["quality_soft_fail"]
            return self._manual_review_result(
                review_type="identify",
                image_paths=[str(image_path)],
                candidates=result.candidates,
                quality=quality,
                reasons=reasons,
                created_by=created_by,
            )
        if result.decision == "manual_review":
            return self._manual_review_result(
                review_type="identify",
                image_paths=[str(image_path)],
                candidates=result.candidates,
                quality=quality,
                reasons=result.reasons,
                created_by=created_by,
            )
        return result

    def confirm_review(
        self,
        review_id: str,
        action: str,
        reviewer: str,
        *,
        target_sku_id: str | None = None,
        new_sku_name: str | None = None,
    ) -> ReviewConfirmationResult:
        review = self.repository.get_review(review_id)
        if review is None:
            raise KeyError(f"Unknown review_id: {review_id}")

        sample_ids: list[str] = []
        sku_id = target_sku_id
        if review.review_type == "registration":
            if action == REVIEW_ACTION_CREATE_NEW_SKU:
                sku = self.repository.create_sku(new_sku_name or review.target_sku_name or "reviewed-sku", created_by=reviewer)
                sku_id = sku.sku_id
                contexts, _ = self._prepare_contexts([Path(path) for path in review.image_paths])
                sample_ids = [sample.sample_id for sample in self._persist_contexts(sku_id, contexts, created_by=reviewer)]
            elif action == REVIEW_ACTION_BIND_EXISTING_SKU:
                if target_sku_id is None:
                    raise ValueError("target_sku_id is required for bind_existing_sku")
                contexts, _ = self._prepare_contexts([Path(path) for path in review.image_paths])
                sample_ids = [sample.sample_id for sample in self._persist_contexts(target_sku_id, contexts, created_by=reviewer)]
            else:
                sku_id = None
        elif review.review_type == "identify":
            if target_sku_id is None:
                raise ValueError("target_sku_id is required for identification review confirmation")
            if action == REVIEW_ACTION_ADD_HARD_CASE_SAMPLE:
                contexts, _ = self._prepare_contexts([Path(path) for path in review.image_paths])
                sample_ids = [
                    sample.sample_id
                    for sample in self._persist_contexts(
                        target_sku_id,
                        contexts,
                        created_by=reviewer,
                        sample_type="hard_case",
                        source_task_id=review_id,
                    )
                ]
            elif action == REVIEW_ACTION_BIND_EXISTING_SKU:
                sku_id = target_sku_id
            else:
                sku_id = None

        resolved = self.repository.confirm_review(
            review_id,
            actor=reviewer,
            action=action,
            resolution_payload={"target_sku_id": sku_id, "sample_ids": sample_ids},
        )
        return ReviewConfirmationResult(
            review_id=resolved.review_id,
            status=resolved.status,
            action=action,
            sku_id=sku_id,
            sample_ids=sample_ids,
            metadata=resolved.resolution_payload,
        )

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
        source_task_id: str | None = None,
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
                source_task_id=source_task_id,
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

    def _manual_review_result(
        self,
        *,
        review_type: str,
        image_paths: list[str],
        candidates,
        quality,
        reasons,
        created_by: str,
    ) -> DecisionResult:
        review = self.repository.create_review(
            review_type=review_type,
            requested_actions=IDENTIFICATION_REVIEW_ACTIONS,
            image_paths=image_paths,
            candidates=list(candidates),
            quality=quality,
            created_by=created_by,
            metadata={"backend": self.backend.backend_name, "reasons": list(reasons)},
        )
        payload = ManualReviewPayload(
            image_path=image_paths[0],
            candidates=list(candidates),
            quality=quality,
            review_type=review_type,
            requested_actions=IDENTIFICATION_REVIEW_ACTIONS,
            query_metadata={"backend": self.backend.backend_name},
        )
        return DecisionResult(
            decision="manual_review",
            status="manual_review",
            sku_id=None,
            top_candidate=candidates[0] if candidates else None,
            candidates=list(candidates),
            reasons=list(reasons),
            review_id=review.review_id,
            metadata={"manual_review": json.loads(json.dumps(payload, default=lambda item: item.__dict__))},
        )

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
