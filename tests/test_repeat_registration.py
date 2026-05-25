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
            absolute_score=0.75,
            margin_score=0.05,
            registration_duplicate_score=0.88,
            registration_ambiguous_margin=0.04,
        ),
        storage=StorageConfig(root=root),
        cache=FeatureCacheConfig(enabled=True, cache_dir=root / "cache"),
        topk=20,
    )
    backend = PyTorchFeatureBackend(config=config.model, session=ColorSession())
    return ObjectClassifierPipeline(config=config, backend=backend)


def test_register_returns_safe_create_for_new_sku(tmp_path) -> None:
    pipeline = build_pipeline(tmp_path)
    red = tmp_path / "red.png"
    write_pattern_image(red, (220, 40, 30))

    result = pipeline.register("Red Widget", [red])

    assert result.decision == "safe_create"
    assert result.sku is not None


def test_register_still_creates_sku_when_query_is_close_to_existing_sku(tmp_path) -> None:
    pipeline = build_pipeline(tmp_path)
    red = tmp_path / "red.png"
    near_red = tmp_path / "near-red.png"
    write_pattern_image(red, (220, 40, 30))
    write_pattern_image(near_red, (210, 50, 35))

    first = pipeline.register("Red Widget", [red])
    duplicate_attempt = pipeline.register("New Red Widget", [near_red])

    assert first.sku is not None
    assert duplicate_attempt.decision == "safe_create"
    assert duplicate_attempt.sku is not None
    assert duplicate_attempt.candidates
    assert duplicate_attempt.candidates[0].sku_id == first.sku.sku_id


def test_register_still_creates_sku_when_query_is_close_to_multiple_skus(tmp_path) -> None:
    pipeline = build_pipeline(tmp_path)
    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    ambiguous = tmp_path / "purple.png"
    write_pattern_image(red, (220, 40, 30))
    write_pattern_image(blue, (30, 40, 220))
    write_pattern_image(ambiguous, (140, 40, 140))

    pipeline.register("Red Widget", [red])
    pipeline.register("Blue Widget", [blue])
    result = pipeline.register("Unknown Widget", [ambiguous])

    assert result.decision == "safe_create"
    assert result.sku is not None
    assert len(result.candidates) >= 2
