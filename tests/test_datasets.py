"""Tests for dataset loaders, question types, and stats."""

from __future__ import annotations

from pathlib import Path

from pdf_vlm.data import compute_dataset_stats, get_dataset, load_bundle
from pdf_vlm.data.question_types import infer_question_type
from pdf_vlm.schemas import QuestionType

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def test_infer_question_types():
    assert infer_question_type(evidence_pages=[0, 2]) == QuestionType.CROSS_PAGE
    assert infer_question_type(evidence_modality=["table"]) == QuestionType.TABLE
    assert infer_question_type(evidence_modality=["chart"]) == QuestionType.CHART
    assert infer_question_type(unanswerable=True) == QuestionType.UNANSWERABLE
    assert infer_question_type(evidence_pages=[1], evidence_modality=["text"]) == QuestionType.TEXT


def test_mp_docvqa_fixture():
    bundle = load_bundle("mp_docvqa", root=FIXTURES / "mp_docvqa", split="val")
    assert bundle.n_docs == 2
    assert bundle.n_questions == 3
    doc_a = next(d for d in bundle.documents if d.doc_id == "docA")
    assert doc_a.page_count == 2
    assert any(qa.question_type == QuestionType.TABLE for qa in doc_a.qa_pairs)
    examples = get_dataset("mp_docvqa", root=FIXTURES / "mp_docvqa").to_examples()
    assert len(examples) == 3
    assert examples[0].question_type is not None


def test_mmlongbench_subset_fixture():
    bundle = load_bundle(
        "mmlongbench",
        root=FIXTURES / "mmlongbench",
        max_docs=5,
        max_questions=20,
        seed=0,
    )
    assert bundle.n_docs >= 1
    assert bundle.n_questions >= 1
    types = {qa.question_type for doc in bundle.documents for qa in doc.qa_pairs}
    assert QuestionType.CROSS_PAGE in types or QuestionType.CHART in types
    # filter to table only
    filtered = load_bundle(
        "mmlongbench",
        root=FIXTURES / "mmlongbench",
        question_types=["table"],
        max_questions=20,
        seed=0,
    )
    assert filtered.n_questions >= 1
    assert all(
        qa.question_type == QuestionType.TABLE for doc in filtered.documents for qa in doc.qa_pairs
    )


def test_custom_bucket_fixture():
    bundle = load_bundle("custom_5", root=FIXTURES / "custom" / "5")
    assert bundle.n_docs == 1
    assert bundle.n_questions == 3
    stats = compute_dataset_stats(bundle)
    assert stats["avg_pages"] == 5
    assert "text" in stats["question_type_distribution"]
    assert "cross-page" in stats["question_type_distribution"]


def test_eval_pipeline_interface():
    """Eval can iterate three sources with the same fields."""
    specs = [
        ("mp_docvqa", {"root": FIXTURES / "mp_docvqa"}),
        ("mmlongbench", {"root": FIXTURES / "mmlongbench", "max_questions": 10, "seed": 1}),
        ("custom_5", {"root": FIXTURES / "custom" / "5"}),
    ]
    for name, kwargs in specs:
        bundle = get_dataset(name, **kwargs).load()
        assert bundle.n_questions > 0
        for doc, qa in bundle.iter_qa():
            assert doc.doc_id
            assert isinstance(doc.pages, list)
            assert qa.question
            _ = qa.primary_answer
            assert qa.question_type in QuestionType
