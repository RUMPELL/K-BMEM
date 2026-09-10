"""Small model-agnostic dense retrieval helper."""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("embedding must have a finite, non-zero norm")
    return [float(value) / norm for value in vector]


def cosine_ranking(query: Sequence[float], documents: Iterable[tuple[str, Sequence[float]]], top_k: int = 5):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    query_norm = l2_normalize(query)
    scored = []
    for document_id, vector in documents:
        document_norm = l2_normalize(vector)
        if len(document_norm) != len(query_norm):
            raise ValueError("embedding dimensions must match")
        score = sum(left * right for left, right in zip(query_norm, document_norm))
        scored.append((document_id, score))
    return sorted(scored, key=lambda row: (-row[1], row[0]))[:top_k]
