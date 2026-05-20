from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from object_classifier.config import QualityThresholds, ROIPolygon
from object_classifier.quality import assess_quality
from object_classifier.roi import normalize_roi


def build_image(size: tuple[int, int] = (96, 96)) -> Image.Image:
    height, width = size
    y, x = np.indices((height, width))
    base = ((x * 7 + y * 11) % 255).astype(np.uint8)
    image = np.stack([base, np.flipud(base), np.fliplr(base)], axis=-1)
    return Image.fromarray(image)


def test_normalize_roi_preserves_raw_crop_and_relative_points() -> None:
    image = build_image((512, 512))
    roi = normalize_roi(
        image,
        ROIPolygon(((102, 98), (102, 439), (471, 433), (479, 94))),
        min_size=(24, 24),
    )

    assert roi.image.shape == (346, 378, 3)
    assert roi.crop_box == (102, 94, 480, 440)
    assert roi.roi_points == ((102, 98), (102, 439), (471, 433), (479, 94))
    assert roi.relative_points == ((0, 4), (0, 345), (369, 339), (377, 0))
    assert roi.original_size == (512, 512)


def test_normalize_roi_rejects_out_of_range_polygon() -> None:
    image = build_image()

    with pytest.raises(ValueError, match="outside"):
        normalize_roi(
            image,
            ROIPolygon(((10, 10), (10, 80), (120, 80), (120, 10))),
            min_size=(24, 24),
        )


def test_quality_hard_fails_obviously_invalid_input() -> None:
    dark = np.zeros((32, 32, 3), dtype=np.uint8)
    result = assess_quality(
        dark,
        QualityThresholds(
            min_width=24,
            min_height=24,
            blur_soft_limit=5.0,
            blur_hard_limit=2.5,
            dark_ratio_limit=0.5,
            bright_ratio_limit=0.5,
        ),
    )

    assert result.status == "hard_fail"
    assert "underexposed" in result.reasons
