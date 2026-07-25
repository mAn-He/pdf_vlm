"""Compare page-level vs hierarchical retrieval with shared metrics."""

from __future__ import annotations

import re
import time
from typing import Any, Sequence

from pdf_vlm.eval.recall import page_hit_at_k, recall_at_k
from pdf_vlm.retrieve.result import RetrievalResult
from pdf_vlm.schemas import QAExample
from pdf_vlm.utils.logging import get_logger

logger = get_logger("retrieve.compare")


def _normalize_ans(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def gold_answer_contained(retrieved_texts: Sequence[str], gold_answers: Sequence[str]) -> bool:
    """Whether any gold answer string appears in concatenated retrieved evidence."""
    if not gold_answers:
        return False
    blob = _normalize_ans("\n".join(retrieved_texts))
    if not blob:
        return False
    for g in gold_answers:
        gn = _normalize_ans(str(g))
        if gn and gn in blob:
            return True
    return False


def evaluate_retrieval_result(
    result: RetrievalResult,
    example: QAExample,
    *,
    ks: list[int] | None = None,
) -> dict[str, Any]:
    ks = ks or [1, 3, 5]
    pages = result.page_ids
    texts = [h.text or "" for h in result.final_hits]
    contained = gold_answer_contained(texts, example.answers)
    metrics: dict[str, Any] = {
        "example_id": example.example_id,
        "mode": result.mode,
        "latency_ms": result.latency_ms,
        "retrieved_pages": pages,
        "evidence_pages": list(example.evidence_pages),
        "gold_answer_contained": contained,
        "n_final_hits": len(result.final_hits),
        "n_coarse": len(result.coarse_hits),
        "trace": result.format_trace(),
    }
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(pages, example.evidence_pages, k)
        metrics[f"page_hit@{k}"] = page_hit_at_k(pages, example.evidence_pages, k)
    return metrics


def compare_retrievers(
    page_retriever,
    hier_retriever,
    examples: list[QAExample],
    *,
    top_k: int = 3,
    ks: list[int] | None = None,
    print_traces: bool = True,
) -> dict[str, Any]:
    """Run both retrievers on the same questions; return per-example + aggregate metrics."""
    ks = ks or [1, 3, 5]
    rows: list[dict[str, Any]] = []
    for ex in examples:
        page_res = page_retriever.retrieve_detailed(ex.question, top_k=top_k)
        hier_res = hier_retriever.retrieve_detailed(ex.question, top_k=top_k)
        if print_traces:
            logger.info("==== PAGE TRACE ====\n%s", page_res.format_trace())
            logger.info("==== HIERARCHICAL TRACE ====\n%s", hier_res.format_trace())
        page_m = evaluate_retrieval_result(page_res, ex, ks=ks)
        hier_m = evaluate_retrieval_result(hier_res, ex, ks=ks)
        rows.append(
            {
                "example_id": ex.example_id,
                "question": ex.question,
                "page": page_m,
                "hierarchical": hier_m,
            }
        )

    def _agg(mode: str) -> dict[str, Any]:
        items = [r[mode] for r in rows]
        out: dict[str, Any] = {"n": len(items)}
        if not items:
            return out
        out["latency_ms_mean"] = sum(i["latency_ms"] for i in items) / len(items)
        out["gold_answer_contained_rate"] = sum(1 for i in items if i["gold_answer_contained"]) / len(items)
        for k in ks:
            out[f"recall@{k}_mean"] = sum(i[f"recall@{k}"] for i in items) / len(items)
            out[f"page_hit@{k}_mean"] = sum(i[f"page_hit@{k}"] for i in items) / len(items)
        return out

    summary = {
        "top_k": top_k,
        "page": _agg("page"),
        "hierarchical": _agg("hierarchical"),
        "examples": rows,
    }
    return summary
