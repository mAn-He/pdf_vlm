"""Aggregate evaluation over QAPrediction list + harness exports."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from pdf_vlm.eval.anls import anls
from pdf_vlm.eval.em_f1 import best_token_f1, exact_match
from pdf_vlm.eval.mmlong_score import generalized_acc_f1, score_answer
from pdf_vlm.eval.recall import page_hit_at_k, recall_at_k
from pdf_vlm.schemas import QAPrediction

if TYPE_CHECKING:
    from pdf_vlm.eval.config import HarnessConfig
    from pdf_vlm.eval.harness import EvaluationHarness
    from pdf_vlm.eval.rows import EvalRow


def evaluate_predictions(
    preds: list[QAPrediction],
    *,
    dataset: str = "custom",
    ks: list[int] | None = None,
) -> dict[str, Any]:
    ks = ks or [1, 3, 5]
    if not preds:
        return {"n": 0}

    anls_scores = [anls(p.prediction, p.gold_answers) for p in preds]
    em_scores = [exact_match(p.prediction, p.gold_answers) for p in preds]
    f1_scores = [best_token_f1(p.prediction, p.gold_answers) for p in preds]

    metrics: dict[str, Any] = {
        "n": len(preds),
        "dataset": dataset,
        "anls_mean": sum(anls_scores) / len(preds),
        "em_mean": sum(em_scores) / len(preds),
        "token_f1_mean": sum(f1_scores) / len(preds),
        "latency_ms_mean": sum(p.e2e_latency_ms for p in preds) / len(preds),
        "retrieval_latency_ms_mean": sum(p.retrieval_latency_ms for p in preds) / len(preds),
        "generation_latency_ms_mean": sum(p.generation_latency_ms for p in preds) / len(preds),
    }

    rss = [p.peak_rss_mb for p in preds if p.peak_rss_mb is not None]
    vram = [p.peak_vram_mb for p in preds if p.peak_vram_mb is not None]
    if rss:
        metrics["peak_rss_mb_max"] = max(rss)
    if vram:
        metrics["peak_vram_mb_max"] = max(vram)

    for k in ks:
        metrics[f"recall@{k}"] = sum(recall_at_k(p.retrieved_page_ids, p.evidence_pages, k) for p in preds) / len(
            preds
        )
        metrics[f"page_hit@{k}"] = sum(page_hit_at_k(p.retrieved_page_ids, p.evidence_pages, k) for p in preds) / len(
            preds
        )

    if dataset.startswith("mmlong"):
        una = [bool(p.meta.get("unanswerable")) for p in preds]
        scores = [
            score_answer(p.prediction, p.gold_answers, unanswerable=u) for p, u in zip(preds, una, strict=False)
        ]
        metrics.update({f"mmlong_{k}": v for k, v in generalized_acc_f1(scores, una).items()})

    if dataset.startswith("mp_docvqa"):
        metrics["primary"] = "anls"
        metrics["primary_score"] = metrics["anls_mean"]
    elif dataset.startswith("mmlong"):
        metrics["primary"] = "mmlong_f1"
        metrics["primary_score"] = metrics.get("mmlong_f1", 0.0)
    else:
        metrics["primary"] = "anls"
        metrics["primary_score"] = metrics["anls_mean"]

    return metrics


def __getattr__(name: str):
    if name == "EvaluationHarness":
        from pdf_vlm.eval.harness import EvaluationHarness

        return EvaluationHarness
    if name == "run_harness":
        from pdf_vlm.eval.harness import run_harness

        return run_harness
    if name == "HarnessConfig":
        from pdf_vlm.eval.config import HarnessConfig

        return HarnessConfig
    if name == "load_harness_config":
        from pdf_vlm.eval.config import load_harness_config

        return load_harness_config
    if name == "EvalRow":
        from pdf_vlm.eval.rows import EvalRow

        return EvalRow
    raise AttributeError(name)


__all__ = [
    "evaluate_predictions",
    "EvaluationHarness",
    "run_harness",
    "HarnessConfig",
    "load_harness_config",
    "EvalRow",
]
