"""FAISS helpers with numpy fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pdf_vlm.utils.io import ensure_dir, load_json, save_json
from pdf_vlm.utils.logging import get_logger

logger = get_logger("index.faiss_store")


class VectorStore:
    def __init__(self, dim: int, metric: str = "ip"):
        self.dim = dim
        self.metric = metric
        self.vectors: np.ndarray | None = None
        self.ids: list[str] = []
        self._index = None
        self._backend = "numpy"

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected (*, {self.dim}) vectors, got {vectors.shape}")
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.ids.extend(ids)
        self._index = None

    def _build(self) -> None:
        if self.vectors is None:
            raise RuntimeError("Empty store")
        try:
            import faiss

            index = faiss.IndexFlatIP(self.dim) if self.metric == "ip" else faiss.IndexFlatL2(self.dim)
            index.add(self.vectors)
            self._index = index
            self._backend = "faiss"
        except Exception:
            self._backend = "numpy"
            self._index = "numpy"

    def search(self, query: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        if self.vectors is None or not self.ids:
            return []
        if self._index is None:
            self._build()
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        k = min(top_k, len(self.ids))
        if self._backend == "faiss":
            scores, idxs = self._index.search(q, k)
            return [(self.ids[i], float(scores[0][j])) for j, i in enumerate(idxs[0]) if i >= 0]

        # numpy cosine / dot
        mat = self.vectors
        sims = (mat @ q[0]) if self.metric == "ip" else -np.linalg.norm(mat - q[0], axis=1)
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in order]

    def save(self, directory: str | Path) -> None:
        directory = ensure_dir(Path(directory))
        if self.vectors is not None:
            np.save(directory / "vectors.npy", self.vectors)
        save_json(
            directory / "meta.json",
            {"dim": self.dim, "metric": self.metric, "ids": self.ids, "backend": self._backend},
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        meta = load_json(directory / "meta.json")
        store = cls(dim=int(meta["dim"]), metric=meta.get("metric", "ip"))
        store.ids = list(meta["ids"])
        store.vectors = np.load(directory / "vectors.npy")
        return store
