"""Text embedders (BGE-M3) with a hashing fallback for offline tests."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from pdf_vlm.utils.logging import get_logger

logger = get_logger("index.text_embedder")


class TextEmbedder:
    """Dense text embedder. Use ``name='hash'`` for a lightweight offline fallback (Colab/CPU)."""

    def __init__(
        self,
        name: str = "BAAI/bge-m3",
        device: str = "cpu",
        normalize: bool = True,
        dim: int = 1024,
        backend: str | None = None,
    ):
        self.name = name
        self.device = device
        self.normalize = normalize
        self.dim = dim
        self._model = None
        forced = (backend or "").lower() in {"hash", "hashing", "offline"} or name.lower() in {
            "hash",
            "hashing",
            "offline",
        }
        self._backend = "hash_forced" if forced else "unset"

    def _load(self) -> None:
        if self._model is not None or self._backend == "hash_forced":
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device=self.device)
            self._backend = "st"
            logger.info("Loaded text embedder %s on %s", self.name, self.device)
        except Exception as exc:
            logger.warning("sentence-transformers unavailable (%s); using hashing embedder", exc)
            self._backend = "hash"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        if self._backend == "st" and self._model is not None:
            vecs = self._model.encode(
                list(texts),
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return np.asarray(vecs, dtype=np.float32)

        # Deterministic hashing bag-of-features fallback
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            if self.normalize:
                n = np.linalg.norm(out[i])
                if n > 0:
                    out[i] /= n
        return out
