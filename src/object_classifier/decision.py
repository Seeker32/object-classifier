from __future__ import annotations

from collections import defaultdict

from .config import DecisionThresholds
from .schemas import Candidate, DecisionResult


def aggregate_candidates(candidates: list[Candidate]) -> list[Candidate]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.sku_id].append(candidate)

    aggregated: list[Candidate] = []
    for sku_id, sku_candidates in grouped.items():
        best = max(sku_candidates, key=_candidate_score)
        aggregated.append(
            Candidate(
                sample_id=best.sample_id,
                sku_id=sku_id,
                global_score=best.global_score,
                rerank_score=best.rerank_score,
                best_sample_id=best.sample_id,
                hit_count=len(sku_candidates),
            )
        )
    return sorted(aggregated, key=_candidate_score, reverse=True)


def decide_top_candidate(
    candidates: list[Candidate],
    thresholds: DecisionThresholds,
) -> DecisionResult:
    ordered = sorted(candidates, key=_candidate_score, reverse=True)
    if not ordered:
        return DecisionResult(
            decision="manual_review",
            status="manual_review",
            sku_id=None,
            top_candidate=None,
            candidates=[],
            reasons=["no_candidates"],
        )

    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else None
    top1_score = _candidate_score(top1)
    top2_score = _candidate_score(top2) if top2 else 0.0
    margin = top1_score - top2_score if top2 else 1.0
    reasons: list[str] = []
    decision = "auto_accept"

    if top1_score < thresholds.absolute_score:
        decision = "manual_review"
        reasons.append("below_absolute_threshold")
    elif top2 and margin < thresholds.margin_score:
        decision = "manual_review"
        reasons.append("below_margin_threshold")

    return DecisionResult(
        decision=decision,
        status=decision,
        sku_id=top1.sku_id if decision == "auto_accept" else None,
        top_candidate=top1,
        candidates=ordered,
        reasons=reasons,
        metadata={
            "top1_score": top1_score,
            "top2_score": top2_score,
            "margin": margin,
        },
    )


def _candidate_score(candidate: Candidate | None) -> float:
    if candidate is None:
        return 0.0
    if candidate.rerank_score is not None:
        return candidate.rerank_score
    return candidate.global_score
