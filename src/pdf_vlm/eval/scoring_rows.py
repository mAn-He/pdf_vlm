"""Score a single QAPrediction into an EvalRow."""

from __future__ import annotations

from typing import Any

from pdf_vlm.data.custom_pdfs import parse_bucket
from pdf_vlm.eval.anls import anls
from pdf_vlm.eval.em_f1 import best_token_f1, exact_match
from pdf_vlm.eval.recall import page_hit_at_k, recall_at_k
from pdf_vlm.eval.rows import EvalRow
from pdf_vlm.retrieve.compare import gold_answer_contained
from pdf_vlm.schemas import QAExample, QAPrediction


def infer_page_bucket(dataset: str, example: QAExample | None = None) -> int | None:
    bucket = parse_bucket(dataset)
    if bucket is not None:
        return bucket
    if example is not None:
        meta_bucket = example.meta.get("page_bucket") or example.meta.get("num_pages")
        if meta_bucket is not None:
            try:
                return int(meta_bucket)
            except (TypeError, ValueError):
                pass
    return None


def question_type_str(example: QAExample) -> str:
    qt = example.question_type
    if qt is None:
        return str(example.meta.get("question_type") or "text")
    return qt.value if hasattr(qt, "value") else str(qt)


def prediction_to_row(
    pred: QAPrediction,
    *,
    run_id: str,
    dataset: str,
    pipeline_type: str,
    retrieval_type: str,
    top_k: int,
    example: QAExample | None = None,
    primary_metric: str = "anls",
    page_bucket: int | None = None,
    monitor_rss: float | None = None,
    monitor_vram: float | None = None,
) -> EvalRow:
    golds = list(pred.gold_answers)
    gold_joined = " | ".join(golds)
    anls_s = anls(pred.prediction, golds)
    em_s = exact_match(pred.prediction, golds)
    f1_s = best_token_f1(pred.prediction, golds)

    primary = primary_metric.lower()
    if primary == "em":
        correctness = em_s
    elif primary in {"f1", "token_f1"}:
        correctness = f1_s
    else:
        correctness = anls_s
        primary = "anls"

    # Gold span present in retrieved evidence (retrieval quality probe)
    hit_texts = [(h.text or "") for h in pred.hits] if pred.hits else []
    if not hit_texts and pred.prediction:
        hit_texts = [pred.prediction]
    try:
        contained = gold_answer_contained(hit_texts, golds)
    except Exception:
        contained = False

    rss = pred.peak_rss_mb if pred.peak_rss_mb is not None else monitor_rss
    vram = pred.peak_vram_mb if pred.peak_vram_mb is not None else monitor_vram

    if page_bucket is None and example is not None:
        page_bucket = infer_page_bucket(dataset, example)
    elif page_bucket is None:
        page_bucket = infer_page_bucket(dataset)

    qtype = "text"
    if example is not None:
        qtype = question_type_str(example)
    elif pred.meta.get("question_type"):
        qtype = str(pred.meta["question_type"])

    image_load = pred.meta.get("image_load_latency_ms")
    if image_load is not None:
        try:
            image_load = float(image_load)
        except (TypeError, ValueError):
            image_load = None

    return EvalRow(
        run_id=run_id,
        dataset=dataset,
        doc_id=pred.doc_id,
        question_id=pred.example_id,
        question_type=qtype,
        page_bucket=page_bucket,
        pipeline_type=pipeline_type,
        retrieval_type=retrieval_type,
        top_k=top_k,
        question=pred.question,
        answer=pred.prediction,
        gold_answer=gold_joined,
        gold_answers=golds,
        primary_metric=primary,
        correctness=float(correctness),
        anls=float(anls_s),
        em=float(em_s),
        token_f1=float(f1_s),
        latency_ms=float(pred.e2e_latency_ms),
        retrieval_latency_ms=float(pred.retrieval_latency_ms),
        generation_latency_ms=float(pred.generation_latency_ms),
        image_load_latency_ms=image_load,
        peak_rss_mb=float(rss) if rss is not None else None,
        peak_vram_mb=float(vram) if vram is not None else None,
        retrieved_pages=list(pred.retrieved_page_ids),
        evidence_pages=list(pred.evidence_pages),
        recall_at_1=float(recall_at_k(pred.retrieved_page_ids, pred.evidence_pages, 1)),
        recall_at_3=float(recall_at_k(pred.retrieved_page_ids, pred.evidence_pages, 3)),
        recall_at_5=float(recall_at_k(pred.retrieved_page_ids, pred.evidence_pages, 5)),
        page_hit_at_1=float(page_hit_at_k(pred.retrieved_page_ids, pred.evidence_pages, 1)),
        page_hit_at_3=float(page_hit_at_k(pred.retrieved_page_ids, pred.evidence_pages, 3)),
        page_hit_at_5=float(page_hit_at_k(pred.retrieved_page_ids, pred.evidence_pages, 5)),
        gold_answer_contained=bool(contained),
        meta={
            "unanswerable": bool((pred.meta or {}).get("unanswerable")),
            "n_hits": len(pred.hits),
            **{k: v for k, v in (pred.meta or {}).items() if k in {"n_images_used", "vision_fallback", "modality"}},
        },
    )


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize_rows(rows: list[EvalRow], *, label: str = "all") -> dict[str, Any]:
    if not rows:
        return {"group": label, "n": 0}
    out: dict[str, Any] = {
        "group": label,
        "n": len(rows),
        "correctness_mean": mean([r.correctness for r in rows]),
        "anls_mean": mean([r.anls for r in rows]),
        "em_mean": mean([r.em for r in rows]),
        "token_f1_mean": mean([r.token_f1 for r in rows]),
        "latency_ms_mean": mean([r.latency_ms for r in rows]),
        "retrieval_latency_ms_mean": mean([r.retrieval_latency_ms for r in rows]),
        "generation_latency_ms_mean": mean([r.generation_latency_ms for r in rows]),
        "recall@1_mean": mean([r.recall_at_1 for r in rows]),
        "recall@3_mean": mean([r.recall_at_3 for r in rows]),
        "recall@5_mean": mean([r.recall_at_5 for r in rows]),
        "page_hit@1_mean": mean([r.page_hit_at_1 for r in rows]),
        "page_hit@3_mean": mean([r.page_hit_at_3 for r in rows]),
        "gold_answer_contained_rate": mean([1.0 if r.gold_answer_contained else 0.0 for r in rows]),
    }
    rss = [r.peak_rss_mb for r in rows if r.peak_rss_mb is not None]
    vram = [r.peak_vram_mb for r in rows if r.peak_vram_mb is not None]
    if rss:
        out["peak_rss_mb_max"] = max(rss)
        out["peak_rss_mb_mean"] = mean(rss)
    if vram:
        out["peak_vram_mb_max"] = max(vram)
        out["peak_vram_mb_mean"] = mean(vram)
    return out
