"""Common scoring interface for page-level and hierarchical retrieval."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from pdf_vlm.index.text_embedder import TextEmbedder


class Scorer(Protocol):
    """Score a query against candidate texts; higher is better."""

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        ...


class EmbeddingScorer:
    """Dense embedding cosine / IP scorer (shared by page & hierarchical)."""

    def __init__(self, embedder: TextEmbedder | None = None, device: str = "cpu"):
        self.embedder = embedder or TextEmbedder(device=device)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        q = self.embedder.embed([query])[0]
        mat = self.embedder.embed(list(texts))
        # assume L2-normalized -> IP == cosine
        sims = mat @ q
        return [float(x) for x in np.asarray(sims).reshape(-1)]

    def rank(
        self, query: str, texts: Sequence[str], top_k: int
    ) -> list[tuple[int, float]]:
        scores = self.score(query, texts)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in order[:top_k]]
