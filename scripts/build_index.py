#!/usr/bin/env python
"""Build retrieval indices for processed documents."""

from __future__ import annotations

import argparse

from pdf_vlm.index.hierarchical_indexer import build_hierarchical_index
from pdf_vlm.index.page_indexer import build_page_index
from pdf_vlm.ocr.paddle_structure import load_artifact
from pdf_vlm.utils.io import load_named_config, resolve_path
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--modality", choices=["text", "multimodal"], default="text")
    parser.add_argument("--retrieval", choices=["page", "hierarchical"], default="page")
    args = parser.parse_args()

    variant = f"{'page' if args.retrieval == 'page' else 'hier'}_{'text' if args.modality == 'text' else 'mm'}"
    cfg = load_named_config(f"retrieval/{variant}.yaml")
    art = load_artifact(args.doc_id)
    out = resolve_path(f"indices/{args.doc_id}/{variant}")
    if args.retrieval == "page":
        build_page_index([art], out, modality=args.modality, retrieval_cfg=cfg)
    else:
        build_hierarchical_index([art], out, modality=args.modality, retrieval_cfg=cfg)
    logger.info("index=%s", out)


if __name__ == "__main__":
    main()
