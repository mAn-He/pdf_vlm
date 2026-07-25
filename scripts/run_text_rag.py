#!/usr/bin/env python
"""Run text-only RAG baseline (page or section retrieval).

Example:
  python scripts/build_text_index.py --doc-id <DOC> --enrich-pdf-text
  python scripts/run_text_rag.py --doc-id <DOC> --retrieval page --dry-run
  python scripts/run_text_rag.py --doc-id <DOC> --retrieval section --dataset custom_5
"""

from __future__ import annotations

import argparse
from datetime import datetime

from pdf_vlm.data import load_dataset
from pdf_vlm.eval import evaluate_predictions
from pdf_vlm.rag.text_only import TextOnlyRAGPipeline
from pdf_vlm.utils.io import resolve_path, save_json
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="Text-only RAG baseline runner")
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--dataset", default="custom_5")
    parser.add_argument("--retrieval", choices=["page", "section", "hierarchical"], default="page")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--coarse-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="Retrieve + log only (no Gemma)")
    parser.add_argument(
        "--question",
        default=None,
        help="Optional single question (skips dataset file)",
    )
    args = parser.parse_args()

    variant = {
        "page": "page_text",
        "section": "section_text",
        "hierarchical": "hier_text",
    }[args.retrieval]
    index_dir = resolve_path(f"indices/{args.doc_id}/{variant}")
    if not index_dir.exists():
        raise SystemExit(
            f"Missing index: {index_dir}\n"
            f"Build with: python scripts/build_retrieval_indexes.py --doc-id {args.doc_id} "
            f"--enrich-pdf-text   # or build_text_index.py for page/section"
        )

    pipe = TextOnlyRAGPipeline.from_paths(
        index_dir,
        retrieval_mode=args.retrieval,  # type: ignore[arg-type]
        top_k=args.top_k,
        device=args.device,
        dry_run=args.dry_run,
    )

    if args.question:
        from pdf_vlm.schemas import QAExample

        examples = [
            QAExample(
                example_id="cli_q0",
                doc_id=args.doc_id,
                question=args.question,
                answers=[],
                evidence_pages=[],
            )
        ]
    else:
        examples = load_dataset(args.dataset, limit=args.limit)
        if not examples:
            raise SystemExit(f"No examples for dataset={args.dataset}")
        for ex in examples:
            ex.doc_id = args.doc_id

    run_name = (
        f"text_rag_{args.retrieval}_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    result = pipe.run(examples, run_name=run_name)
    preds = result["predictions"]
    metrics = evaluate_predictions(preds, dataset=args.dataset)
    save_json(resolve_path(result["out_dir"]) / "metrics.json", metrics)
    logger.info("metrics=%s", metrics)
    logger.info("out_dir=%s", result["out_dir"])


if __name__ == "__main__":
    main()
