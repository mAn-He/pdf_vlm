"""Unit tests for metrics and normalization (no heavy deps)."""

from __future__ import annotations

from pdf_vlm.eval.anls import anls
from pdf_vlm.eval.em_f1 import exact_match, best_token_f1
from pdf_vlm.eval.recall import page_hit_at_k, recall_at_k
from pdf_vlm.eval import evaluate_predictions
from pdf_vlm.ocr.normalize import normalize_document
from pdf_vlm.schemas import QAPrediction


def test_anls_exact():
    assert anls("Hello World", ["hello world"]) == 1.0


def test_anls_threshold():
    assert anls("abc", ["xyz"], threshold=0.5) == 0.0


def test_em_and_f1():
    assert exact_match("Paris", ["paris"]) == 1.0
    assert best_token_f1("the cat sat", ["cat sat"]) > 0.5


def test_recall():
    assert recall_at_k([0, 2, 5], [2, 9], k=3) == 0.5
    assert page_hit_at_k([1, 2], [2], k=2) == 1.0


def test_normalize_document():
    page_metas = [{"page_id": 0, "image_path": "x.png", "width": 100, "height": 100}]
    raw = [{"markdown": "# Title\n\nBody text", "ocr_texts": ["Title", "Body text"]}]
    doc = normalize_document("doc1", "a.pdf", page_metas, raw)
    assert doc.num_pages == 1
    assert "Body" in doc.pages[0].markdown
    assert doc.sections


def test_evaluate_predictions():
    preds = [
        QAPrediction(
            example_id="1",
            doc_id="d",
            question="q",
            prediction="answer",
            gold_answers=["answer"],
            retrieved_page_ids=[0],
            evidence_pages=[0],
        )
    ]
    m = evaluate_predictions(preds, dataset="custom_5")
    assert m["n"] == 1
    assert m["anls_mean"] == 1.0
    assert m["page_hit@1"] == 1.0
