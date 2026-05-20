from __future__ import annotations

import json

from object_classifier.calibration import load_thresholds, recommend_thresholds


def test_recommend_thresholds_returns_runtime_specific_thresholds() -> None:
    payload = recommend_thresholds(
        {
            "pytorch": {
                "same_sku_scores": [0.96, 0.94, 0.91],
                "hard_negative_scores": [0.41, 0.45, 0.52],
                "margins": [0.18, 0.21, 0.14],
            },
            "rknn": {
                "same_sku_scores": [0.94, 0.92, 0.89],
                "hard_negative_scores": [0.48, 0.49, 0.55],
                "margins": [0.12, 0.10, 0.11],
            },
        }
    )

    assert set(payload) == {"pytorch", "rknn"}
    assert payload["pytorch"]["absolute_score"] > payload["pytorch"]["negative_score_ceiling"]
    assert payload["rknn"]["margin_score"] > 0.0


def test_load_thresholds_prefers_runtime_specific_values(tmp_path) -> None:
    config_path = tmp_path / "thresholds.json"
    config_path.write_text(
        json.dumps(
            {
                "default": {"absolute_score": 0.8, "margin_score": 0.05},
                "rknn": {"absolute_score": 0.82, "margin_score": 0.08},
            }
        ),
        encoding="utf-8",
    )

    thresholds = load_thresholds(config_path, runtime_name="rknn")

    assert thresholds.absolute_score == 0.82
    assert thresholds.margin_score == 0.08
