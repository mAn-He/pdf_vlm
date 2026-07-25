#!/usr/bin/env python
"""Minimal sample for text-only RAG (dry-run friendly)."""

from __future__ import annotations

from pdf_vlm.rag.text_only import TextOnlyRAGPipeline
from pdf_vlm.schemas import QAExample
from pdf_vlm.utils.io import resolve_path
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()

DOC_ID = "acme_demo_5pages_de31152fee"


def main() -> None:
    index_dir = resolve_path(f"indices/{DOC_ID}/page_text")
    pipe = TextOnlyRAGPipeline.from_paths(
        index_dir, retrieval_mode="page", top_k=2, dry_run=True
    )
    example = QAExample(
        example_id="sample_1",
        doc_id=DOC_ID,
        question="When was Acme Corp founded?",
        answers=["1998"],
        evidence_pages=[0],
    )
    pred = pipe.answer(example)
    print("retrieved_pages:", pred.retrieved_page_ids)
    print("latency_ms:", round(pred.retrieval_latency_ms, 2))
    for h in pred.hits:
        print(f"  hit pages={h.page_ids} score={h.score:.3f} text={h.text[:80]!r}")


if __name__ == "__main__":
    main()
