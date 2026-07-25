#!/usr/bin/env python
"""Compare page-level vs hierarchical retrieval (recall@k, gold containment, latency)."""

from __future__ import annotations

import argparse
from datetime import datetime

from pdf_vlm.data import load_dataset
from pdf_vlm.retrieve.compare import compare_retrievers
from pdf_vlm.retrieve.hierarchical_retriever import HierarchicalRetriever
from pdf_vlm.retrieve.page_retriever import PageRetriever
from pdf_vlm.utils.io import resolve_path, save_json
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="Page vs hierarchical retrieval comparison")
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--dataset", default="custom_5")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--coarse-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quiet-traces", action="store_true")
    args = parser.parse_args()

    page_dir = resolve_path(f"indices/{args.doc_id}/page_text")
    hier_dir = resolve_path(f"indices/{args.doc_id}/hier_text")
    if not page_dir.exists() or not hier_dir.exists():
        raise SystemExit(
            "Missing indexes. Run:\n"
            f"  python scripts/build_retrieval_indexes.py --doc-id {args.doc_id} --enrich-pdf-text"
        )

    page_ret = PageRetriever(page_dir, modality="text", device=args.device)
    hier_ret = HierarchicalRetriever(
        hier_dir, modality="text", device=args.device, coarse_k=args.coarse_k, fine_k=args.top_k
    )

    examples = load_dataset(args.dataset, limit=args.limit)
    if not examples:
        raise SystemExit(f"No examples for {args.dataset}")
    for ex in examples:
        ex.doc_id = args.doc_id

    summary = compare_retrievers(
        page_ret,
        hier_ret,
        examples,
        top_k=args.top_k,
        print_traces=not args.quiet_traces,
    )

    run_id = "retrieval_cmp_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out = resolve_path(f"results/runs/{run_id}")
    save_json(out / "comparison.json", summary)
    # compact table
    compact = {
        "page": summary["page"],
        "hierarchical": summary["hierarchical"],
        "top_k": args.top_k,
        "doc_id": args.doc_id,
        "dataset": args.dataset,
    }
    save_json(out / "summary.json", compact)
    logger.info("PAGE aggregate: %s", summary["page"])
    logger.info("HIER aggregate: %s", summary["hierarchical"])
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
