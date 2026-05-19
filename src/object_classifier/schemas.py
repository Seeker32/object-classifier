from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class SKU:
    sku_id: str
    sku_name: str
    status: str = "active"
    created_by: str = "system"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    sku_id: str
    image_path: str
    roi_box: tuple[int, int, int, int]
    roi_version: str = "p0-fixed"
    quality_score: float = 0.0
    quality_status: str = "pass"
    sample_type: str = "register"
    source_task_id: str | None = None
    created_by: str = "system"
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class FeatureBundle:
    global_embedding: np.ndarray
    patch_tokens: np.ndarray
    backend: str


@dataclass(frozen=True)
class FeatureRecord:
    sample_id: str
    sku_id: str
    feature_version: str
    global_embedding_path: str
    patch_token_path: str
    backend: str


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    sku_id: str
    global_score: float
    rerank_score: float | None = None
    best_sample_id: str | None = None
    hit_count: int = 1


@dataclass(frozen=True)
class DecisionResult:
    decision: str
    status: str
    sku_id: str | None
    top_candidate: Candidate | None
    candidates: list[Candidate]
    reasons: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityResult:
    status: str
    score: float
    blur_score: float
    dark_ratio: float
    bright_ratio: float
    reasons: list[str]


@dataclass(frozen=True)
class NormalizedROI:
    image: np.ndarray
    source_path: str | None
    roi_box: tuple[int, int, int, int]
    original_size: tuple[int, int]


@dataclass(frozen=True)
class ManualReviewPayload:
    image_path: str
    candidates: list[Candidate]
    quality: QualityResult
    query_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegistrationResult:
    sku: SKU
    samples: list[Sample]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExportArtifacts:
    status: str
    report_path: Path
    embedding_onnx: Path | None
    patch_tokens_onnx: Path | None
    rknn_log: Path | None = None
    notes: list[str] = field(default_factory=list)
