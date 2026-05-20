from __future__ import annotations

import numpy as np

from object_classifier.config import ModelConfig, ROIPolygon
from object_classifier.features import prepare_roi_image, preprocess_roi
from object_classifier.schemas import NormalizedROI


def test_prepare_roi_image_warps_polygon_before_resize() -> None:
    image = np.zeros((346, 378, 3), dtype=np.uint8)
    image[:20, :20] = (255, 0, 0)
    image[:20, -20:] = (0, 255, 0)
    image[-20:, -20:] = (0, 0, 255)
    image[-20:, :20] = (255, 255, 0)
    roi = NormalizedROI(
        image=image,
        source_path=None,
        roi_points=((102, 98), (102, 439), (471, 433), (479, 94)),
        relative_points=((0, 4), (0, 345), (369, 339), (377, 0)),
        crop_box=(102, 94, 480, 440),
        original_size=(512, 512),
    )

    prepared = prepare_roi_image(roi, (224, 224))
    batch = preprocess_roi(prepared, (224, 224))

    assert prepared.shape == (224, 224, 3)
    assert batch.shape == (1, 3, 224, 224)
    assert prepared[10, 10, 0] > 150
    assert prepared[10, -10, 1] > 150
    assert prepared[-10, 10, 0] > 150
    assert prepared[-10, 10, 1] > 150


def test_model_config_uses_new_default_polygon() -> None:
    config = ModelConfig()

    assert config.roi_box == ROIPolygon(((102, 98), (102, 439), (471, 433), (479, 94)))
