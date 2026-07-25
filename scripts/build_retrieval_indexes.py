#!/usr/bin/env python
"""Build page + hierarchical indexes for retrieval granularity experiments."""

from __future__ import annotations

import argparse

from pdf_vlm.index.hierarchical_indexer import build_hierarchical_index
from pdf_vlm.index.text_indexer import build_text_only_index
from pdf_vlm.ocr.paddle_structure import load_artifact
from pdf_vlm.rag.text_only import enrich_artifact_from_pdf_text
from pdf_vlm.utils.io import load_named_config, resolve_path
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--enrich-pdf-text", action="store_true")
    parser.add_argument("--hierarchy-strategy", default="auto")
    parser.add_argument("--fine-unit", choices=["page", "paragraph"], default="page")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--hash-embedder",
        action="store_true",
        help="Use hashing embedder configs (low RAM / Colab)",
    )
    args = parser.parse_args()

    if args.enrich_pdf_text:
        enrich_artifact_from_pdf_text(args.doc_id)

    art = load_artifact(args.doc_id)
    page_name = "retrieval/page_text_hash.yaml" if args.hash_embedder else "retrieval/page_text.yaml"
    hier_name = "retrieval/hier_text_hash.yaml" if args.hash_embedder else "retrieval/hier_text.yaml"
    page_cfg = load_named_config(page_name)
    page_cfg.setdefault("embedder", {})["device"] = args.device
    page_out = resolve_path(f"indices/{args.doc_id}/page_text")
    build_text_only_index([art], page_out, mode="page", retrieval_cfg=page_cfg)

    hier_cfg = load_named_config(hier_name)
    hier_cfg.setdefault("embedder", {})["device"] = args.device
    hier_out = resolve_path(f"indices/{args.doc_id}/hier_text")
    build_hierarchical_index(
        [art],
        hier_out,
        modality="text",
        retrieval_cfg=hier_cfg,
        hierarchy_strategy=args.hierarchy_strategy,
        fine_unit=args.fine_unit,
    )
    logger.info("page_index=%s", page_out)
    logger.info("hier_index=%s", hier_out)


if __name__ == "__main__":
    main()
