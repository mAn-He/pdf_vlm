"""Tests for text-only RAG baseline (no LLM weights required)."""

from __future__ import annotations

from pathlib import Path

from pdf_vlm.index.text_indexer import build_text_only_index
from pdf_vlm.rag.prompt_builder import build_text_prompt
from pdf_vlm.rag.text_only import TextOnlyRAGPipeline, prediction_log_record
from pdf_vlm.schemas import DocumentArtifact, PageArtifact, QAExample, SectionNode
from pdf_vlm.retrieve.page_retriever import PageRetriever
from pdf_vlm.retrieve.section_retriever import SectionRetriever


def _toy_artifact() -> DocumentArtifact:
    pages = [
        PageArtifact(page_id=0, markdown="Acme Corp was founded in 1998 in Seoul."),
        PageArtifact(page_id=1, markdown="Flagship product is VisionX-4 industrial scanner."),
        PageArtifact(page_id=2, markdown="Warranty for VisionX-4 is 24 months."),
    ]
    sections = [
        SectionNode(section_id="sec0", title="Overview", page_ids=[0], summary_text=pages[0].markdown),
        SectionNode(section_id="sec1", title="Products", page_ids=[1], summary_text=pages[1].markdown),
        SectionNode(section_id="sec2", title="Support", page_ids=[2], summary_text=pages[2].markdown),
    ]
    return DocumentArtifact(
        doc_id="toy_rag",
        source_path="toy.pdf",
        num_pages=3,
        pages=pages,
        sections=sections,
        full_markdown="\n".join(p.markdown for p in pages),
    )


def test_build_page_and_section_indexes(tmp_path: Path):
    doc = _toy_artifact()
    page_dir = tmp_path / "page_text"
    sec_dir = tmp_path / "section_text"
    build_text_only_index([doc], page_dir, mode="page", retrieval_cfg={"embedder": {"device": "cpu"}})
    build_text_only_index([doc], sec_dir, mode="section", retrieval_cfg={"embedder": {"device": "cpu"}})
    assert (page_dir / "chunks.json").exists()
    assert (sec_dir / "manifest.json").exists()

    page_ret = PageRetriever(page_dir, modality="text", device="cpu")
    hits = page_ret.retrieve("When was Acme founded?", top_k=1)
    assert hits and hits[0].image_path is None
    assert 0 in hits[0].page_ids

    sec_ret = SectionRetriever(sec_dir, device="cpu")
    hits2 = sec_ret.retrieve("VisionX-4 warranty", top_k=1)
    assert hits2 and hits2[0].level == "section"
    assert 2 in hits2[0].page_ids


def test_text_only_pipeline_dry_run(tmp_path: Path):
    doc = _toy_artifact()
    idx = tmp_path / "page_text"
    build_text_only_index([doc], idx, mode="page", retrieval_cfg={"embedder": {"device": "cpu"}})
    pipe = TextOnlyRAGPipeline.from_paths(idx, retrieval_mode="page", top_k=2, dry_run=True)
    pred = pipe.answer(
        QAExample(
            example_id="q1",
            doc_id="toy_rag",
            question="What is the flagship product?",
            answers=["VisionX-4"],
            evidence_pages=[1],
        )
    )
    assert pred.meta["baseline"] == "text_only_rag"
    assert pred.meta["modality"] == "text"
    assert pred.retrieved_page_ids
    assert "prompt" in pred.meta
    assert "OCR text" in pred.meta["prompt"] or "Evidence" in pred.meta["prompt"]
    rec = prediction_log_record(pred)
    assert "answer" in rec and "retrieved_pages" in rec and "latency" in rec


def test_prompt_has_no_image_markers():
    from pdf_vlm.schemas import RetrievalHit

    hits = [
        RetrievalHit(
            chunk_id="c1",
            score=0.9,
            doc_id="d",
            page_ids=[0],
            level="page",
            text="hello",
            image_path=None,
        )
    ]
    prompt = build_text_prompt("Q?", hits)
    assert "image" not in prompt.lower() or "OCR text only" in prompt
    assert "<start_of_image>" not in prompt
