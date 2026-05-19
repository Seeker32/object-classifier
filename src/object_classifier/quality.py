from __future__ import annotations

import numpy as np

from .config import QualityThresholds
from .schemas import QualityResult


def assess_quality(image: np.ndarray, thresholds: QualityThresholds) -> QualityResult:
    height, width = image.shape[:2]
    reasons: list[str] = []

    if width < thresholds.min_width or height < thresholds.min_height:
        reasons.append("size_below_minimum")

    grayscale = image.astype(np.float32).mean(axis=2)
    blur_score = _blur_score(grayscale)
    dark_ratio = float(np.mean(grayscale < 30.0))
    bright_ratio = float(np.mean(grayscale > 225.0))

    if dark_ratio > thresholds.dark_ratio_limit:
        reasons.append("underexposed")
    if bright_ratio > thresholds.bright_ratio_limit:
        reasons.append("overexposed")
    if blur_score < thresholds.blur_hard_limit:
        reasons.append("too_blurry")

    status = "pass"
    if reasons:
        status = "hard_fail"
    elif blur_score < thresholds.blur_soft_limit:
        reasons.append("soft_blur_warning")
        status = "soft_fail"

    penalty = min(1.0, len(reasons) * 0.25)
    score = max(0.0, 1.0 - penalty)
    return QualityResult(
        status=status,
        score=score,
        blur_score=blur_score,
        dark_ratio=dark_ratio,
        bright_ratio=bright_ratio,
        reasons=reasons,
    )


def _blur_score(grayscale: np.ndarray) -> float:
    grad_x = np.diff(grayscale, axis=1)
    grad_y = np.diff(grayscale, axis=0)
    return float(np.var(grad_x) + np.var(grad_y))
