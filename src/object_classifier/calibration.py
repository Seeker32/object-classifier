from __future__ import annotations

import json
from pathlib import Path

from .config import DecisionThresholds


def recommend_thresholds(runtime_payloads: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
    recommendations: dict[str, dict[str, float]] = {}
    for runtime_name, payload in runtime_payloads.items():
        same_scores = sorted(payload.get("same_sku_scores", []))
        negative_scores = sorted(payload.get("hard_negative_scores", []))
        margins = sorted(payload.get("margins", []))
        if not same_scores:
            raise ValueError(f"same_sku_scores is required for runtime {runtime_name}")
        min_positive = same_scores[0]
        max_negative = negative_scores[-1] if negative_scores else 0.0
        absolute_score = round((min_positive + max_negative) / 2.0, 4)
        margin_score = round(max(margins[0] if margins else 0.05, 0.01), 4)
        recommendations[runtime_name] = {
            "absolute_score": absolute_score,
            "margin_score": margin_score,
            "negative_score_ceiling": round(max_negative, 4),
            "positive_score_floor": round(min_positive, 4),
        }
    return recommendations


def load_thresholds(path: Path, runtime_name: str = "default") -> DecisionThresholds:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runtime_payload = payload.get(runtime_name) or payload.get("default") or {}
    return DecisionThresholds(
        absolute_score=float(runtime_payload.get("absolute_score", 0.78)),
        margin_score=float(runtime_payload.get("margin_score", 0.05)),
        registration_duplicate_score=float(runtime_payload.get("registration_duplicate_score", 0.9)),
        registration_global_score=float(runtime_payload.get("registration_global_score", 0.8)),
        registration_ambiguous_margin=float(runtime_payload.get("registration_ambiguous_margin", 0.03)),
    )
