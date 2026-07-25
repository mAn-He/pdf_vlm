"""Tests for page vs hierarchical retrieval."""

from __future__ import annotations

from pathlib import Path

from pdf_vlm.index.hierarchy import apply_hierarchy, build_document_hierarchy
from pdf_vlm.index.hierarchical_indexer import build_hierarchical_index
from pdf_vlm.index.text_indexer import build_text_only_index
from pdf_vlm.retrieve.compare import compare_retrievers, gold_answer_contained
from pdf_vlm.retrieve.hierarchical_retriever import HierarchicalRetriever
from pdf_vlm.retrieve.page_retriever import PageRetriever
from pdf_vlm.retrieve.scoring import EmbeddingScorer
from pdf_vlm.schemas import DocumentArtifact, PageArtifact, QAExample, SectionNode


def _long_doc() -> DocumentArtifact:
    pages = [
        PageArtifact(page_id=0, markdown="# Overview\nAcme Corp was founded in 1998 in Seoul."),
        PageArtifact(page_id=1, markdown="# Products\nFlagship product is VisionX-4."),
        PageArtifact(page_id=2, markdown="# Financials\nRevenue 2024 was 12.5M."),
        PageArtifact(page_id=3, markdown="# Support\nWarranty is 24 months for VisionX-4."),
        PageArtifact(page_id=4, markdown="# Offices\nHQ Seoul, branch Busan."),
    ]
    return DocumentArtifact(
        doc_id="long_toy",
        source_path="toy.pdf",
        num_pages=5,
        pages=pages,
        sections=[],
        full_markdown="\n\n".join(p.markdown for p in pages),
    )


def test_hierarchy_markdown_strategy():
    doc = _long_doc()
    sections, used = build_document_hierarchy(doc, strategy="markdown")
    assert used == "markdown"
    assert len(sections) >= 4
    assert any("Products" in s.title for s in sections)


def test_shared_scorer_interface(tmp_path: Path):
    scorer = EmbeddingScorer(device="cpu")
    scores = scorer.score("warranty", ["warranty 24 months", "unrelated cats"])
    assert scores[0] > scores[1]
    ranked = scorer.rank("warranty", ["a", "warranty 24 months", "b"], top_k=1)
    assert ranked[0][0] == 1


def test_hierarchical_coarse_to_fine_path(tmp_path: Path):
    doc = apply_hierarchy(_long_doc(), strategy="markdown")
    page_dir = tmp_path / "page_text"
    hier_dir = tmp_path / "hier_text"
    cfg = {"embedder": {"device": "cpu"}}
    build_text_only_index([doc], page_dir, mode="page", retrieval_cfg=cfg)
    build_hierarchical_index(
        [doc], hier_dir, modality="text", retrieval_cfg=cfg, hierarchy_strategy="markdown", fine_unit="page"
    )

    page_ret = PageRetriever(page_dir, device="cpu")
    hier_ret = HierarchicalRetriever(hier_dir, device="cpu", coarse_k=2, fine_k=2)

    detailed = hier_ret.retrieve_detailed("What is the warranty length?", top_k=2)
    assert detailed.mode == "hierarchical"
    assert detailed.coarse_hits, "must expose coarse section candidates"
    assert detailed.fine_hits, "must expose fine candidates"
    trace = detailed.format_trace()
    assert "COARSE" in trace and "FINE" in trace
    assert detailed.latency_ms >= 0

    page_det = page_ret.retrieve_detailed("What is the warranty length?", top_k=2)
    assert page_det.mode == "page"
    assert page_det.final_hits


def test_compare_metrics(tmp_path: Path):
    doc = apply_hierarchy(_long_doc(), strategy="markdown")
    page_dir = tmp_path / "page_text"
    hier_dir = tmp_path / "hier_text"
    cfg = {"embedder": {"device": "cpu"}}
    build_text_only_index([doc], page_dir, mode="page", retrieval_cfg=cfg)
    build_hierarchical_index([doc], hier_dir, modality="text", retrieval_cfg=cfg, hierarchy_strategy="markdown")

    examples = [
        QAExample(
            example_id="q1",
            doc_id="long_toy",
            question="How long is the warranty?",
            answers=["24 months"],
            evidence_pages=[3],
        )
    ]
    summary = compare_retrievers(
        PageRetriever(page_dir, device="cpu"),
        HierarchicalRetriever(hier_dir, device="cpu", coarse_k=3, fine_k=3),
        examples,
        top_k=3,
        print_traces=False,
    )
    assert "recall@3_mean" in summary["page"]
    assert "gold_answer_contained_rate" in summary["hierarchical"]
    assert "latency_ms_mean" in summary["page"]


def test_gold_answer_contained():
    assert gold_answer_contained(["Warranty is 24 months."], ["24 months"])
    assert not gold_answer_contained(["nope"], ["24 months"])
