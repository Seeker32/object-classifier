from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from object_classifier.config import (
    DecisionThresholds,
    FeatureCacheConfig,
    ModelConfig,
    PipelineConfig,
    QualityThresholds,
    ROIBox,
    StorageConfig,
)
from object_classifier.features import PyTorchFeatureBackend
from object_classifier.pipeline import ObjectClassifierPipeline


class ColorSession:
    def infer(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        red = float(batch[0, 0].mean())
        green = float(batch[0, 1].mean())
        blue = float(batch[0, 2].mean())
        embedding = np.array([red, green, blue], dtype=np.float32)
        patch_tokens = np.array(
            [[red, green], [green, blue], [red, blue]],
            dtype=np.float32,
        )
        return embedding, patch_tokens


def write_pattern_image(path: Path, rgb: tuple[int, int, int]) -> None:
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    image[..., 0] = rgb[0]
    image[..., 1] = rgb[1]
    image[..., 2] = rgb[2]
    image[::2, ::2] = np.clip(image[::2, ::2] + 20, 0, 255)
    Image.fromarray(image).save(path)


def build_pipeline(root: Path) -> ObjectClassifierPipeline:
    config = PipelineConfig(
        model=ModelConfig(
            backend="pytorch",
            input_size=(32, 32),
            roi_box=ROIBox(0, 0, 48, 48),
            embedding_dim=3,
        ),
        quality=QualityThresholds(
            min_width=24,
            min_height=24,
            blur_soft_limit=0.0,
            blur_hard_limit=0.0,
            dark_ratio_limit=1.0,
            bright_ratio_limit=1.0,
        ),
        decision=DecisionThresholds(
            absolute_score=0.8,
            margin_score=0.04,
            registration_duplicate_score=0.88,
            registration_ambiguous_margin=0.04,
        ),
        storage=StorageConfig(root=root),
        cache=FeatureCacheConfig(enabled=True, cache_dir=root / "cache"),
        topk=20,
    )
    backend = PyTorchFeatureBackend(config=config.model, session=ColorSession())
    return ObjectClassifierPipeline(config=config, backend=backend)


def test_confirm_review_can_add_hard_case_sample_and_write_audit_record(tmp_path) -> None:
    pipeline = build_pipeline(tmp_path)
    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    ambiguous = tmp_path / "purple.png"
    write_pattern_image(red, (220, 40, 30))
    write_pattern_image(blue, (30, 40, 220))
    write_pattern_image(ambiguous, (140, 40, 140))

    registered = pipeline.register("Red Widget", [red])
    pipeline.register("Blue Widget", [blue])
    review_result = pipeline.identify(ambiguous)

    confirmation = pipeline.confirm_review(
        review_id=review_result.review_id,
        action="add_hard_case_sample",
        reviewer="qa-user",
        target_sku_id=registered.sku.sku_id,
    )

    samples = pipeline.repository.list_samples_by_sku(registered.sku.sku_id, include_inactive=True)
    assert confirmation.status == "approved"
    assert any(sample.sample_type == "hard_case" for sample in samples)
    assert pipeline.repository.list_audit_records()


def test_confirm_registration_review_can_bind_existing_sku(tmp_path) -> None:
    pipeline = build_pipeline(tmp_path)
    red = tmp_path / "red.png"
    near_red = tmp_path / "near-red.png"
    write_pattern_image(red, (220, 40, 30))
    write_pattern_image(near_red, (210, 50, 35))

    registered = pipeline.register("Red Widget", [red])
    duplicate_attempt = pipeline.register("Duplicate Red Widget", [near_red])

    confirmation = pipeline.confirm_review(
        review_id=duplicate_attempt.review_id,
        action="bind_existing_sku",
        reviewer="qa-user",
        target_sku_id=registered.sku.sku_id,
    )

    assert confirmation.status == "approved"
    samples = pipeline.repository.list_samples_by_sku(registered.sku.sku_id, include_inactive=True)
    assert len(samples) == 2
