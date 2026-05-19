from __future__ import annotations

from object_classifier.config import DecisionThresholds
from object_classifier.decision import aggregate_candidates, decide_top_candidate
from object_classifier.schemas import Candidate


def test_aggregate_candidates_groups_samples_by_sku() -> None:
    candidates = [
        Candidate(sample_id="s1", sku_id="sku-a", global_score=0.9, rerank_score=0.81),
        Candidate(sample_id="s2", sku_id="sku-a", global_score=0.88, rerank_score=0.84),
        Candidate(sample_id="s3", sku_id="sku-b", global_score=0.87, rerank_score=0.79),
    ]

    aggregated = aggregate_candidates(candidates)

    assert [candidate.sku_id for candidate in aggregated] == ["sku-a", "sku-b"]
    assert aggregated[0].best_sample_id == "s2"
    assert aggregated[0].hit_count == 2
    assert aggregated[0].rerank_score == 0.84


def test_decide_top_candidate_accepts_clear_winner() -> None:
    candidates = [
        Candidate(sample_id="s1", sku_id="sku-a", global_score=0.9, rerank_score=0.92, best_sample_id="s1"),
        Candidate(sample_id="s2", sku_id="sku-b", global_score=0.85, rerank_score=0.78, best_sample_id="s2"),
    ]

    result = decide_top_candidate(
        candidates,
        DecisionThresholds(absolute_score=0.8, margin_score=0.08),
    )

    assert result.decision == "auto_accept"
    assert result.sku_id == "sku-a"


def test_decide_top_candidate_rejects_on_low_score() -> None:
    candidates = [
        Candidate(sample_id="s1", sku_id="sku-a", global_score=0.9, rerank_score=0.62, best_sample_id="s1"),
    ]

    result = decide_top_candidate(
        candidates,
        DecisionThresholds(absolute_score=0.8, margin_score=0.05),
    )

    assert result.decision == "manual_review"
    assert "below_absolute_threshold" in result.reasons


def test_decide_top_candidate_rejects_on_small_margin() -> None:
    candidates = [
        Candidate(sample_id="s1", sku_id="sku-a", global_score=0.9, rerank_score=0.85, best_sample_id="s1"),
        Candidate(sample_id="s2", sku_id="sku-b", global_score=0.89, rerank_score=0.83, best_sample_id="s2"),
    ]

    result = decide_top_candidate(
        candidates,
        DecisionThresholds(absolute_score=0.8, margin_score=0.03),
    )

    assert result.decision == "manual_review"
    assert "below_margin_threshold" in result.reasons
