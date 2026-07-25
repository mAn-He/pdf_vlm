"""Tests for multimodal RAG (dry-run, no Gemma weights)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from pdf_vlm.index.text_indexer import build_text_only_index
from pdf_vlm.rag.multimodal import MultimodalRAGPipeline, multimodal_log_record
from pdf_vlm.rag.prompt_builder import build_multimodal_prompt
from pdf_vlm.rag.page_images import ordered_page_ids_from_hits, resolve_multimodal_pages
from pdf_vlm.schemas import DocumentArtifact, PageArtifact, QAExample, RetrievalHit, SectionNode


def _artifact_with_images(tmp_path: Path) -> DocumentArtifact:
    pages = []
    for i, text in enumerate(
        [
            "Overview: Acme founded 1998",
            "Products: VisionX-4 chart peak at 80",
            "Warranty: 24 months",
        ]
    ):
        img = tmp_path / f"page_{i}.png"
        Image.new("RGB", (64, 64), color=(20 * i, 80, 120)).save(img)
        pages.append(PageArtifact(page_id=i, markdown=text, image_path=str(img)))
    sections = [
        SectionNode(section_id=f"s{i}", title=f"S{i}", page_ids=[i], summary_text=pages[i].markdown)
        for i in range(3)
    ]
    return DocumentArtifact(
        doc_id="mm_toy",
        source_path="toy.pdf",
        num_pages=3,
        pages=pages,
        sections=sections,
        full_markdown="\n".join(p.markdown for p in pages),
    )


def test_never_uses_all_pages_when_top_k_smaller(tmp_path: Path):
    art = _artifact_with_images(tmp_path)
    idx = tmp_path / "page_text"
    build_text_only_index([art], idx, mode="page", retrieval_cfg={"embedder": {"device": "cpu"}})

    # Fake 5-page artifact would be worse; with 3 pages and top_k=2 ensure <=2 images
    pipe = MultimodalRAGPipeline(
        llm=None,
        retriever=__import__("pdf_vlm.retrieve.page_retriever", fromlist=["PageRetriever"]).PageRetriever(
            idx, modality="text", device="cpu"
        ),
        artifact=art,
        top_k=2,
        max_images=2,
        dry_run=True,
    )
    pred = pipe.answer(
        QAExample(
            example_id="q",
            doc_id="mm_toy",
            question="What product is shown in the chart?",
            answers=["VisionX-4"],
            evidence_pages=[1],
        )
    )
    assert pred.meta["n_images_used"] <= 2
    assert len(pred.meta["used_images"]) <= 2
    assert len(pred.retrieved_page_ids) <= 3
    # Must not attach images for pages outside retrieved set
    used_ids = {p["page_id"] for p in pred.meta["mm_pages"]}
    assert used_ids.issubset(set(pred.retrieved_page_ids))


def test_resolve_multimodal_pages_latency_and_order(tmp_path: Path):
    art = _artifact_with_images(tmp_path)
    hits = [
        RetrievalHit(chunk_id="a", score=0.9, doc_id="mm_toy", page_ids=[2], text="w"),
        RetrievalHit(chunk_id="b", score=0.5, doc_id="mm_toy", page_ids=[0], text="o"),
    ]
    pages, latency = resolve_multimodal_pages(art, hits, max_images=2)
    assert ordered_page_ids_from_hits(hits) == [2, 0]
    assert [p["page_id"] for p in pages] == [2, 0]
    assert all(p["exists"] for p in pages)
    assert latency >= 0.0


def test_multimodal_prompt_requires_page_citation_and_images():
    pages = [
        {"page_id": 1, "ocr_text": "table total 42", "image_path": "a.png", "score": 0.8, "exists": True},
        {"page_id": 3, "ocr_text": "chart", "image_path": "b.png", "score": 0.7, "exists": True},
    ]
    prompt, images = build_multimodal_prompt("What is the table total?", pages)
    assert "OCR" in prompt
    assert "page" in prompt.lower()
    assert "Image 1: page_id=1" in prompt
    assert images == ["a.png", "b.png"]
    assert "all pages" not in prompt.lower()


def test_log_record_fields(tmp_path: Path):
    art = _artifact_with_images(tmp_path)
    idx = tmp_path / "page_text"
    build_text_only_index([art], idx, mode="page", retrieval_cfg={"embedder": {"device": "cpu"}})
    from pdf_vlm.retrieve.page_retriever import PageRetriever

    pipe = MultimodalRAGPipeline(
        None, PageRetriever(idx, modality="text", device="cpu"), art, top_k=1, dry_run=True
    )
    pred = pipe.answer(
        QAExample(example_id="1", doc_id="mm_toy", question="founded?", answers=["1998"], evidence_pages=[0])
    )
    rec = multimodal_log_record(pred)
    assert "retrieved_pages" in rec
    assert "used_images" in rec
    assert "final_answer" in rec
    assert "latency" in rec and "image_load_ms" in rec["latency"]
