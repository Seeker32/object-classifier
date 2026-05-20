from __future__ import annotations

import numpy as np
from PIL import Image

from .config import ROIBox, ROIPolygon
from .schemas import NormalizedROI


def normalize_roi(
    image: Image.Image,
    roi_box: ROIPolygon | ROIBox,
    min_size: tuple[int, int],
    source_path: str | None = None,
) -> NormalizedROI:
    polygon = roi_box if isinstance(roi_box, ROIPolygon) else ROIPolygon(roi_box.points)
    width, height = image.size
    if (
        polygon.left < 0
        or polygon.top < 0
        or polygon.right >= width
        or polygon.bottom >= height
    ):
        raise ValueError("ROI box outside image bounds")
    if polygon.width < min_size[0] or polygon.height < min_size[1]:
        raise ValueError("ROI box below minimum size")
    if polygon.width <= 0 or polygon.height <= 0:
        raise ValueError("ROI box must have positive area")

    crop_right = polygon.right + 1
    crop_bottom = polygon.bottom + 1
    cropped = image.crop((polygon.left, polygon.top, crop_right, crop_bottom)).convert("RGB")
    relative_points = tuple(
        (point[0] - polygon.left, point[1] - polygon.top)
        for point in polygon.points
    )
    return NormalizedROI(
        image=np.array(cropped),
        source_path=source_path,
        roi_points=polygon.points,
        relative_points=relative_points,
        crop_box=(polygon.left, polygon.top, crop_right, crop_bottom),
        original_size=(width, height),
    )
