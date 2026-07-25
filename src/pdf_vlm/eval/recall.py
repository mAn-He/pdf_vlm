"""Retrieval recall@k."""

from __future__ import annotations

from typing import Sequence


def recall_at_k(retrieved_pages: Sequence[int], evidence_pages: Sequence[int], k: int | None = None) -> float:
    if not evidence_pages:
        return 0.0
    pages = list(retrieved_pages[:k] if k is not None else retrieved_pages)
    evidence = set(int(p) for p in evidence_pages)
    hit = evidence.intersection(int(p) for p in pages)
    return len(hit) / len(evidence)


def page_hit_at_k(retrieved_pages: Sequence[int], evidence_pages: Sequence[int], k: int = 1) -> float:
    """Binary: any evidence page in top-k (useful for MP-DocVQA single-page evidence)."""
    if not evidence_pages:
        return 0.0
    pages = set(int(p) for p in retrieved_pages[:k])
    return 1.0 if pages.intersection(int(p) for p in evidence_pages) else 0.0
