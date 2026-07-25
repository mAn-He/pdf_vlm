"""Page image loading utilities for multimodal RAG (top-k only)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pdf_vlm.schemas import DocumentArtifact, PageArtifact, RetrievalHit
from pdf_vlm.utils.logging import get_logger

logger = get_logger("rag.page_images")


def page_image_map(artifact: DocumentArtifact) -> dict[int, str]:
    """Map page_id -> existing image path."""
    out: dict[int, str] = {}
    for page in artifact.pages:
        if page.image_path and Path(page.image_path).exists():
            out[int(page.page_id)] = str(page.image_path)
    return out


def page_text_map(artifact: DocumentArtifact) -> dict[int, str]:
    out: dict[int, str] = {}
    for page in artifact.pages:
        text = (page.markdown or "").strip()
        if not text and page.blocks:
            text = "\n".join(b.text for b in page.blocks if b.text.strip())
        if page.tables:
            extras = [t.html or t.text for t in page.tables if (t.html or t.text)]
            if extras:
                text = (text + "\n" + "\n".join(extras)).strip()
        out[int(page.page_id)] = text
    return out


def ordered_page_ids_from_hits(hits: list[RetrievalHit], max_pages: int | None = None) -> list[int]:
    """Deduplicate page ids preserving retrieval rank order."""
    pages: list[int] = []
    for hit in hits:
        for pid in hit.page_ids:
            pid = int(pid)
            if pid not in pages:
                pages.append(pid)
            if max_pages is not None and len(pages) >= max_pages:
                return pages
    return pages


def resolve_multimodal_pages(
    artifact: DocumentArtifact,
    hits: list[RetrievalHit],
    *,
    max_images: int = 3,
) -> tuple[list[dict[str, Any]], float]:
    """Build multimodal page payloads for top-ranked pages only.

    Returns (pages, image_load_latency_ms).
    Each page dict: page_id, ocr_text, image_path, score, exists
    """
    t0 = time.perf_counter()
    images = page_image_map(artifact)
    texts = page_text_map(artifact)

    # Best retrieval score per page
    best_score: dict[int, float] = {}
    for hit in hits:
        for pid in hit.page_ids:
            pid = int(pid)
            prev = best_score.get(pid)
            if prev is None or hit.score > prev:
                best_score[pid] = float(hit.score)

    page_ids = ordered_page_ids_from_hits(hits, max_pages=max_images)
    pages: list[dict[str, Any]] = []
    for pid in page_ids:
        path = images.get(pid)
        # Touch-read to include IO in latency when file exists
        exists = False
        if path:
            p = Path(path)
            exists = p.exists()
            if exists:
                _ = p.stat().st_size
        pages.append(
            {
                "page_id": pid,
                "ocr_text": texts.get(pid, ""),
                "image_path": path,
                "score": best_score.get(pid, 0.0),
                "exists": exists,
            }
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    missing = [p["page_id"] for p in pages if not p["exists"]]
    if missing:
        logger.warning("Missing page images for pages=%s (doc=%s)", missing, artifact.doc_id)
    return pages, latency_ms


def attach_hit_images(hits: list[RetrievalHit], artifact: DocumentArtifact) -> list[RetrievalHit]:
    """Fill image_path on hits from artifact (text index may have stripped images)."""
    images = page_image_map(artifact)
    enriched: list[RetrievalHit] = []
    for hit in hits:
        path = None
        for pid in hit.page_ids:
            if int(pid) in images:
                path = images[int(pid)]
                break
        enriched.append(hit.model_copy(update={"image_path": path}))
    return enriched
