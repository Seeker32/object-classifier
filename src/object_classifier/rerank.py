from __future__ import annotations

import numpy as np

from .schemas import Candidate, FeatureBundle


def compute_patch_similarity(query_tokens: np.ndarray, candidate_tokens: np.ndarray) -> float:
    query = _normalize_rows(query_tokens)
    candidate = _normalize_rows(candidate_tokens)
    similarity = query @ candidate.T
    forward = similarity.max(axis=1).mean()
    backward = similarity.max(axis=0).mean()
    return float((forward + backward) / 2.0)


def rerank_candidates(
    query_tokens: np.ndarray,
    candidates: list[Candidate],
    feature_loader,
) -> list[Candidate]:
    reranked: list[Candidate] = []
    for candidate in candidates:
        bundle: FeatureBundle = feature_loader(candidate.sample_id)
        rerank_score = compute_patch_similarity(query_tokens, bundle.patch_tokens)
        reranked.append(
            Candidate(
                sample_id=candidate.sample_id,
                sku_id=candidate.sku_id,
                global_score=candidate.global_score,
                rerank_score=rerank_score,
            )
        )
    return sorted(reranked, key=lambda item: item.rerank_score or item.global_score, reverse=True)


def _normalize_rows(tokens: np.ndarray) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.float32)
    norms = np.linalg.norm(tokens, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return tokens / norms
