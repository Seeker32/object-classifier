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
    ROIPolygon,
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
            roi_box=ROIPolygon(((0, 0), (0, 47), (47, 47), (47, 0))),
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
        decision=DecisionThresholds(absolute_score=0.75, margin_score=0.05),
        storage=StorageConfig(root=root),
        cache=FeatureCacheConfig(enabled=True, cache_dir=root / "cache"),
        topk=20,
    )
    backend = PyTorchFeatureBackend(config=config.model, session=ColorSession())
    return ObjectClassifierPipeline(config=config, backend=backend)


def test_pipeline_registers_and_identifies_auto_accept(tmp_path) -> None:
    pipeline = build_pipeline(tmp_path)
    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    query = tmp_path / "query-red.png"
    write_pattern_image(red, (220, 40, 30))
    write_pattern_image(blue, (30, 40, 220))
    write_pattern_image(query, (210, 50, 40))

    red_registration = pipeline.register("Red Widget", [red])
    pipeline.register("Blue Widget", [blue])
    result = pipeline.identify(query)

    assert red_registration.sku.sku_id.startswith("sku-")
    assert result.decision == "auto_accept"
    assert result.sku_id == red_registration.sku.sku_id


def test_pipeline_routes_ambiguous_queries_to_manual_review(tmp_path) -> None:
    pipeline = build_pipeline(tmp_path)
    red = tmp_path / "red.png"
    green = tmp_path / "green.png"
    query = tmp_path / "query-ambiguous.png"
    write_pattern_image(red, (200, 40, 40))
    write_pattern_image(green, (40, 200, 40))
    write_pattern_image(query, (140, 140, 40))

    pipeline.register("Red Widget", [red])
    pipeline.register("Green Widget", [green])
    result = pipeline.identify(query)

    assert result.decision == "manual_review"
    assert len(result.candidates) >= 2
    assert "manual_review" in result.metadata
