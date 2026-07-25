#!/usr/bin/env python
"""Build strict text-only indexes (page | section) from processed OCR JSON."""

from __future__ import annotations

import argparse

from pdf_vlm.index.text_indexer import build_text_only_index
from pdf_vlm.ocr.paddle_structure import load_artifact
from pdf_vlm.rag.text_only import enrich_artifact_from_pdf_text
from pdf_vlm.utils.io import load_named_config, resolve_path
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build text-only RAG indexes")
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--mode", choices=["page", "section", "both"], default="both")
    parser.add_argument(
        "--enrich-pdf-text",
        action="store_true",
        help="If OCR markdown is stub/empty, fill from PDF text layer before indexing",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.enrich_pdf_text:
        enrich_artifact_from_pdf_text(args.doc_id)

    art = load_artifact(args.doc_id)
    modes = ["page", "section"] if args.mode == "both" else [args.mode]
    for mode in modes:
        cfg_name = "retrieval/page_text.yaml" if mode == "page" else "retrieval/section_text.yaml"
        cfg = load_named_config(cfg_name)
        cfg.setdefault("embedder", {})["device"] = args.device
        out = resolve_path(f"indices/{args.doc_id}/{mode}_text")
        build_text_only_index([art], out, mode=mode, retrieval_cfg=cfg)  # type: ignore[arg-type]
        logger.info("index ready: %s", out)


if __name__ == "__main__":
    main()
