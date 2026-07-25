"""Tests for evaluation harness (config, scoring, aggregate, dry-run smoke)."""

from __future__ import annotations

from pathlib import Path

from pdf_vlm.eval.aggregate import aggregate_all
from pdf_vlm.eval.config import load_harness_config
from pdf_vlm.eval.harness import EvaluationHarness
from pdf_vlm.eval.report import build_markdown_report, export_reports
from pdf_vlm.eval.rows import EvalRow
from pdf_vlm.eval.scoring_rows import prediction_to_row, summarize_rows
from pdf_vlm.eval.seed import set_global_seed
from pdf_vlm.schemas import QAPrediction
from pdf_vlm.utils.io import resolve_path


def test_load_harness_config_and_cells():
    cfg = load_harness_config("configs/experiments/matrix_2x2.yaml")
    assert cfg.seed == 42
    cells = list(cfg.iter_cells())
    # 2 datasets × 2 pipelines × 2 retrievals × 3 top_k = 24
    assert len(cells) == 24
    assert cfg.index_dirname("hierarchical") == "hier_text"
    assert cfg.index_dirname("page") == "page_text"


def test_set_seed_snapshot():
    snap = set_global_seed(123)
    assert snap["seed"] == 123


def test_prediction_to_row_and_aggregate(tmp_path: Path):
    pred = QAPrediction(
        example_id="q1",
        doc_id="d1",
        question="When?",
        prediction="1998",
        gold_answers=["1998"],
        retrieved_page_ids=[0, 1],
        evidence_pages=[0],
        e2e_latency_ms=12.5,
        retrieval_latency_ms=2.0,
        generation_latency_ms=10.0,
        peak_rss_mb=100.0,
    )
    row = prediction_to_row(
        pred,
        run_id="run_test",
        dataset="custom_5",
        pipeline_type="text",
        retrieval_type="page",
        top_k=3,
    )
    assert row.page_bucket == 5
    assert row.correctness == 1.0
    assert row.anls == 1.0
    assert row.page_hit_at_1 == 1.0

    row2 = row.model_copy(
        update={
            "question_id": "q2",
            "pipeline_type": "multimodal",
            "question_type": "table",
            "correctness": 0.5,
            "anls": 0.5,
        }
    )
    ag = aggregate_all([row, row2])
    assert ag["overall"][0]["n"] == 2
    assert any(x["group"] == "text" for x in ag["by_pipeline"])
    assert any(x["group"] == "table" for x in ag["by_question_type"])
    assert any(x["group"] == "5" for x in ag["by_page_bucket"])

    paths = export_reports(
        tmp_path,
        run_id="run_test",
        rows=[row, row2],
        aggregates=ag,
        config_snapshot={"seed": 1, "datasets": ["custom_5"], "dry_run": True},
        seed_snapshot={"seed": 1},
        skipped=[],
        report_dir=tmp_path / "reports",
    )
    assert Path(paths["predictions_csv"]).exists()
    assert Path(paths["report_md"]).exists()
    md = Path(paths["report_md"]).read_text(encoding="utf-8")
    assert "Evaluation Report" in md
    assert "By page-length bucket" in md


def test_markdown_table_empty():
    md = build_markdown_report(
        run_id="x",
        config_snapshot={"seed": 0, "datasets": [], "pipelines": [], "retrievals": [], "top_k": [], "dry_run": True},
        aggregates={"overall": []},
        n_rows=0,
    )
    assert "Evaluation Report" in md


def test_harness_dry_run_smoke():
    """End-to-end dry-run on custom_5 if indexes exist; otherwise skip gracefully."""
    index = resolve_path("indices/acme_demo_5pages_de31152fee/page_text")
    if not index.exists():
        return

    harness = EvaluationHarness.from_yaml(
        "configs/experiments/eval_harness.yaml",
        dry_run=True,
        datasets=["custom_5"],
        pipelines=["text", "multimodal"],
        retrievals=["page", "hierarchical"],
        top_k=[3],
        limit=2,
        seed=7,
    )
    result = harness.run()
    assert result["n_rows"] >= 2
    assert Path(result["run_dir"], "predictions.csv").exists()
    assert Path(result["run_dir"], "report.md").exists()
    assert Path(result["run_dir"], "config_snapshot.json").exists()
    assert Path(result["run_dir"], "seed_snapshot.json").exists()
    overall = result["aggregates"]["overall"][0]
    assert overall["n"] == result["n_rows"]
    # dry-run: empty answers → ANLS 0, but retrieval recall should be computable
    assert "recall@3_mean" in overall


def test_summarize_rows_empty():
    assert summarize_rows([], label="x")["n"] == 0
