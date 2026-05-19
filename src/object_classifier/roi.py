from __future__ import annotations

from PIL import Image

from .config import ROIBox
from .schemas import NormalizedROI


def normalize_roi(
    image: Image.Image,
    roi_box: ROIBox,
    output_size: tuple[int, int],
    min_size: tuple[int, int],
    source_path: str | None = None,
) -> NormalizedROI:
    width, height = image.size
    if roi_box.left < 0 or roi_box.top < 0 or roi_box.right > width or roi_box.bottom > height:
        raise ValueError("ROI box outside image bounds")
    if roi_box.width < min_size[0] or roi_box.height < min_size[1]:
        raise ValueError("ROI box below minimum size")
    if roi_box.width <= 0 or roi_box.height <= 0:
        raise ValueError("ROI box must have positive area")

    cropped = image.crop((roi_box.left, roi_box.top, roi_box.right, roi_box.bottom)).convert("RGB")
    resized = cropped.resize(output_size, Image.Resampling.BILINEAR)
    return NormalizedROI(
        image=__import__("numpy").array(resized),
        source_path=source_path,
        roi_box=(roi_box.left, roi_box.top, roi_box.right, roi_box.bottom),
        original_size=(width, height),
    )
