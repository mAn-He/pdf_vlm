"""Aggregate EvalRows into overall / question-type / page-bucket tables."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pdf_vlm.eval.rows import EvalRow
from pdf_vlm.eval.scoring_rows import summarize_rows


def _key_parts(row: EvalRow) -> dict[str, Any]:
    return {
        "pipeline_type": row.pipeline_type,
        "retrieval_type": row.retrieval_type,
        "top_k": row.top_k,
        "dataset": row.dataset,
        "page_bucket": row.page_bucket,
        "question_type": row.question_type,
    }


def aggregate_all(rows: list[EvalRow]) -> dict[str, list[dict[str, Any]]]:
    """Build resume-ready summary tables."""
    overall = [summarize_rows(rows, label="overall")]

    by_pipeline: dict[str, list[EvalRow]] = defaultdict(list)
    by_retrieval: dict[str, list[EvalRow]] = defaultdict(list)
    by_cell: dict[str, list[EvalRow]] = defaultdict(list)
    by_qtype: dict[str, list[EvalRow]] = defaultdict(list)
    by_bucket: dict[str, list[EvalRow]] = defaultdict(list)
    by_pipeline_bucket: dict[str, list[EvalRow]] = defaultdict(list)
    by_dataset: dict[str, list[EvalRow]] = defaultdict(list)

    for r in rows:
        by_pipeline[r.pipeline_type].append(r)
        by_retrieval[r.retrieval_type].append(r)
        cell = f"{r.pipeline_type}×{r.retrieval_type}"
        by_cell[cell].append(r)
        by_qtype[r.question_type or "unknown"].append(r)
        bucket = str(r.page_bucket) if r.page_bucket is not None else "unknown"
        by_bucket[bucket].append(r)
        by_pipeline_bucket[f"{r.pipeline_type}|{bucket}"].append(r)
        by_dataset[r.dataset].append(r)

    def _sorted_summaries(groups: dict[str, list[EvalRow]]) -> list[dict[str, Any]]:
        return [summarize_rows(groups[k], label=k) for k in sorted(groups.keys())]

    # Cross table: pipeline × retrieval × bucket
    cross: dict[str, list[EvalRow]] = defaultdict(list)
    for r in rows:
        bucket = str(r.page_bucket) if r.page_bucket is not None else "?"
        cross[f"{r.pipeline_type}|{r.retrieval_type}|bucket={bucket}|k={r.top_k}"].append(r)

    return {
        "overall": overall,
        "by_pipeline": _sorted_summaries(by_pipeline),
        "by_retrieval": _sorted_summaries(by_retrieval),
        "by_cell": _sorted_summaries(by_cell),
        "by_question_type": _sorted_summaries(by_qtype),
        "by_page_bucket": _sorted_summaries(by_bucket),
        "by_pipeline_x_bucket": _sorted_summaries(by_pipeline_bucket),
        "by_dataset": _sorted_summaries(by_dataset),
        "by_full_cell": _sorted_summaries(cross),
    }
