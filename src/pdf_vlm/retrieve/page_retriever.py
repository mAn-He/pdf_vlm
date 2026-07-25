"""Page-level retrieval: each page is an independent search unit."""

from __future__ import annotations

import time
from pathlib import Path

from pdf_vlm.index.store import VectorStore
from pdf_vlm.index.text_embedder import TextEmbedder
from pdf_vlm.index.vision_embedder import VisionEmbedder
from pdf_vlm.retrieve.result import RetrievalResult, StageHit
from pdf_vlm.retrieve.scoring import EmbeddingScorer
from pdf_vlm.schemas import Chunk, RetrievalHit
from pdf_vlm.utils.io import load_json
from pdf_vlm.utils.logging import get_logger

logger = get_logger("retrieve.page")


class PageRetriever:
    """Page-level retriever with shared EmbeddingScorer interface."""

    mode = "page"

    def __init__(
        self,
        index_dir: str | Path,
        modality: str = "text",
        device: str = "cpu",
        scorer: EmbeddingScorer | None = None,
    ):
        self.index_dir = Path(index_dir)
        self.modality = modality
        self.store = VectorStore.load(self.index_dir)
        self.chunks = {
            c["chunk_id"]: Chunk.model_validate(c) for c in load_json(self.index_dir / "chunks.json")
        }
        self.chunk_ids = list(self.chunks.keys())
        self.texts = [self.chunks[cid].text or "" for cid in self.chunk_ids]
        manifest = load_json(self.index_dir / "manifest.json")
        self.modality = manifest.get("modality", modality)
        if scorer is not None:
            self.scorer = scorer
        elif self.modality == "text":
            self.scorer = EmbeddingScorer(TextEmbedder(device=device))
            self.vision_embedder = None
        else:
            self.scorer = None
            self.vision_embedder = VisionEmbedder(device=device)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        return self.retrieve_detailed(query, top_k=top_k).final_hits

    def retrieve_detailed(self, query: str, top_k: int = 3) -> RetrievalResult:
        t0 = time.perf_counter()
        if self.scorer is not None:
            ranked = self.scorer.rank(query, self.texts, top_k=top_k)
            pairs = [(self.chunk_ids[i], score) for i, score in ranked]
        else:
            assert self.vision_embedder is not None
            q = self.vision_embedder.embed_texts([query])[0]
            pairs = self.store.search(q, top_k=top_k)

        hits: list[RetrievalHit] = []
        stages: list[StageHit] = []
        for chunk_id, score in pairs:
            chunk = self.chunks[chunk_id]
            hit = RetrievalHit(
                chunk_id=chunk.chunk_id,
                score=float(score),
                doc_id=chunk.doc_id,
                page_ids=list(chunk.page_ids),
                section_id=chunk.section_id,
                level="page",
                text=chunk.text,
                image_path=None if self.modality == "text" else chunk.image_path,
                meta={"modality": self.modality, "retrieval": "page"},
            )
            hits.append(hit)
            stages.append(
                StageHit(
                    stage="page",
                    chunk_id=chunk.chunk_id,
                    score=float(score),
                    page_ids=list(chunk.page_ids),
                    text_preview=(chunk.text or "")[:160],
                    kept=True,
                )
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = RetrievalResult(
            query=query,
            mode="page",
            final_hits=hits,
            path=stages,
            fine_hits=stages,
            latency_ms=latency_ms,
            meta={"n_candidates": len(self.chunk_ids)},
        )
        logger.debug("\n%s", result.format_trace())
        return result
