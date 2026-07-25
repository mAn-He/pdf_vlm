"""Hierarchical coarse-to-fine retrieval: document/section -> page/paragraph."""

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

logger = get_logger("retrieve.hierarchical")


class HierarchicalRetriever:
    """Coarse section retrieval, then fine page/paragraph within kept sections."""

    mode = "hierarchical"

    def __init__(
        self,
        index_dir: str | Path,
        modality: str = "text",
        device: str = "cpu",
        coarse_k: int = 5,
        fine_k: int = 3,
        scorer: EmbeddingScorer | None = None,
    ):
        self.index_dir = Path(index_dir)
        self.modality = modality
        self.coarse_k = coarse_k
        self.fine_k = fine_k
        self.section_store = (
            VectorStore.load(self.index_dir / "section")
            if (self.index_dir / "section").exists()
            else None
        )
        self.fine_store = VectorStore.load(self.index_dir / "fine")
        self.section_chunks = {
            c["chunk_id"]: Chunk.model_validate(c)
            for c in (
                load_json(self.index_dir / "section_chunks.json")
                if (self.index_dir / "section_chunks.json").exists()
                else []
            )
        }
        self.fine_chunks = {
            c["chunk_id"]: Chunk.model_validate(c)
            for c in load_json(self.index_dir / "fine_chunks.json")
        }
        self.section_ids = list(self.section_chunks.keys())
        self.section_texts = [self.section_chunks[cid].text or "" for cid in self.section_ids]
        self.fine_ids = list(self.fine_chunks.keys())
        self.fine_texts = [self.fine_chunks[cid].text or "" for cid in self.fine_ids]
        self.scorer = scorer or EmbeddingScorer(TextEmbedder(device=device))
        self.vision_embedder = VisionEmbedder(device=device) if modality == "multimodal" else None

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        return self.retrieve_detailed(query, top_k=top_k).final_hits

    def retrieve_detailed(self, query: str, top_k: int = 3) -> RetrievalResult:
        t0 = time.perf_counter()
        coarse_stage: list[StageHit] = []
        fine_stage: list[StageHit] = []
        path: list[StageHit] = []

        # ---- COARSE: section ----
        allowed_pages: set[int] | None = None
        allowed_sections: set[str] = set()
        if self.section_ids:
            ranked = self.scorer.rank(query, self.section_texts, top_k=self.coarse_k)
            allowed_pages = set()
            for idx, score in ranked:
                chunk = self.section_chunks[self.section_ids[idx]]
                kept = True
                stage = StageHit(
                    stage="section",
                    chunk_id=chunk.chunk_id,
                    score=float(score),
                    page_ids=list(chunk.page_ids),
                    section_id=chunk.section_id,
                    title=(chunk.meta or {}).get("title") or (chunk.text or "")[:80],
                    text_preview=(chunk.text or "")[:160],
                    kept=kept,
                )
                coarse_stage.append(stage)
                path.append(stage)
                allowed_pages.update(chunk.page_ids)
                if chunk.section_id:
                    allowed_sections.add(chunk.section_id)
                else:
                    allowed_sections.add(chunk.chunk_id)

        # ---- FINE: page / paragraph within coarse pages ----
        fine_k = top_k if top_k is not None else self.fine_k
        # Score all fine texts, then filter — same scorer interface as page-level
        all_ranked = self.scorer.rank(query, self.fine_texts, top_k=max(len(self.fine_texts), fine_k))
        hits: list[RetrievalHit] = []
        for idx, score in all_ranked:
            chunk = self.fine_chunks[self.fine_ids[idx]]
            in_scope = True
            if allowed_pages is not None:
                in_scope = bool(set(chunk.page_ids).intersection(allowed_pages))
            stage = StageHit(
                stage=chunk.level or "page",
                chunk_id=chunk.chunk_id,
                score=float(score),
                page_ids=list(chunk.page_ids),
                section_id=chunk.section_id,
                text_preview=(chunk.text or "")[:160],
                kept=in_scope and len(hits) < fine_k,
            )
            fine_stage.append(stage)
            if not in_scope:
                continue
            if len(hits) >= fine_k:
                # still log a few dropped in-scope for debug (already marked kept=False above once full)
                continue
            stage.kept = True
            path.append(stage)
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    score=float(score),
                    doc_id=chunk.doc_id,
                    page_ids=list(chunk.page_ids),
                    section_id=chunk.section_id,
                    level=chunk.level or "page",
                    text=chunk.text,
                    image_path=chunk.image_path if self.modality != "text" else None,
                    meta={
                        "modality": self.modality,
                        "retrieval": "hierarchical",
                        "coarse_pages": sorted(allowed_pages) if allowed_pages is not None else None,
                    },
                )
            )

        # Keep fine_stage manageable in logs: top scored + all kept
        fine_stage_sorted = sorted(fine_stage, key=lambda s: s.score, reverse=True)
        fine_log = fine_stage_sorted[: max(fine_k * 3, 10)]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = RetrievalResult(
            query=query,
            mode="hierarchical",
            final_hits=hits,
            path=path,
            coarse_hits=coarse_stage,
            fine_hits=fine_log,
            latency_ms=latency_ms,
            meta={
                "coarse_k": self.coarse_k,
                "fine_k": fine_k,
                "n_sections": len(self.section_ids),
                "n_fine": len(self.fine_ids),
                "allowed_pages": sorted(allowed_pages) if allowed_pages is not None else None,
            },
        )
        logger.info("\n%s", result.format_trace())
        return result
