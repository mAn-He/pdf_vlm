#!/usr/bin/env python
"""Minimal multimodal RAG sample (dry-run friendly)."""

from __future__ import annotations

from pdf_vlm.rag.multimodal import MultimodalRAGPipeline
from pdf_vlm.schemas import QAExample
from pdf_vlm.utils.io import resolve_path
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()

DOC_ID = "acme_demo_5pages_de31152fee"


def main() -> None:
    index_dir = resolve_path(f"indices/{DOC_ID}/page_text")
    pipe = MultimodalRAGPipeline.from_paths(
        index_dir,
        DOC_ID,
        retrieval_mode="page",
        top_k=2,
        max_images=2,
        dry_run=True,
    )
    pred = pipe.answer(
        QAExample(
            example_id="mm_sample_1",
            doc_id=DOC_ID,
            question="What is the flagship product?",
            answers=["VisionX-4"],
            evidence_pages=[1],
        )
    )
    print("retrieved_pages:", pred.retrieved_page_ids)
    print("used_images:", pred.meta.get("used_images"))
    print("image_load_ms:", pred.meta.get("image_load_latency_ms"))
    print("n_images_used:", pred.meta.get("n_images_used"))
    assert pred.meta.get("n_images_used", 0) <= 2
    assert len(pred.retrieved_page_ids) <= 5  # never all-page dump requirement


if __name__ == "__main__":
    main()
