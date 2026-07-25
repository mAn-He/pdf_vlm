#!/usr/bin/env python
"""Run QA over a dataset against a built index (optionally dry-run without LLM)."""

from __future__ import annotations

import argparse
from datetime import datetime

from pdf_vlm.data import load_dataset
from pdf_vlm.eval import evaluate_predictions
from pdf_vlm.llm.gemma_llama_cpp import build_llm
from pdf_vlm.rag.pipeline import RAGPipeline, build_retriever
from pdf_vlm.utils.io import load_named_config, resolve_path, save_json
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="custom_5")
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--modality", choices=["text", "multimodal"], default="text")
    parser.add_argument("--retrieval", choices=["page", "hierarchical"], default="page")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM; empty predictions")
    args = parser.parse_args()

    variant = f"{'page' if args.retrieval == 'page' else 'hier'}_{'text' if args.modality == 'text' else 'mm'}"
    index_dir = resolve_path(f"indices/{args.doc_id}/{variant}")
    ret_cfg = load_named_config(f"retrieval/{variant}.yaml")
    retriever = build_retriever(
        index_dir, retrieval=args.retrieval, modality=args.modality, retrieval_cfg=ret_cfg
    )

    examples = load_dataset(args.dataset, limit=args.limit)
    if not examples:
        raise SystemExit(f"No examples for dataset={args.dataset}")
    # Bind examples to the indexed document when custom packs use a placeholder doc_id
    for ex in examples:
        ex.doc_id = args.doc_id

    preds = []
    if args.dry_run:
        from pdf_vlm.schemas import QAPrediction
        import time

        for ex in examples:
            t0 = time.perf_counter()
            hits = retriever.retrieve(ex.question, top_k=args.top_k)
            ms = (time.perf_counter() - t0) * 1000
            pages = []
            for h in hits:
                for p in h.page_ids:
                    if p not in pages:
                        pages.append(p)
            preds.append(
                QAPrediction(
                    example_id=ex.example_id,
                    doc_id=ex.doc_id,
                    question=ex.question,
                    prediction="",
                    gold_answers=list(ex.answers),
                    retrieved_page_ids=pages,
                    evidence_pages=list(ex.evidence_pages),
                    retrieval_latency_ms=ms,
                    e2e_latency_ms=ms,
                    hits=hits,
                    meta={"unanswerable": ex.unanswerable},
                )
            )
    else:
        model_cfg = load_named_config("models/gemma3_4b_qat.yaml")
        model_cfg["local_path"] = str(resolve_path(model_cfg["local_path"]))
        llm = build_llm(model_cfg)
        pipe = RAGPipeline(
            llm,
            retriever,
            modality=args.modality,
            top_k=args.top_k,
            generation_cfg={"max_tokens": 256, "temperature": 0.1},
        )
        for ex in examples:
            preds.append(pipe.answer(ex))

    metrics = evaluate_predictions(preds, dataset=args.dataset)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{variant}"
    out_dir = resolve_path(f"results/runs/{run_id}")
    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "predictions.json", [p.model_dump(mode="json") for p in preds])
    logger.info("metrics=%s", metrics)
    logger.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
