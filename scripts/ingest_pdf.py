#!/usr/bin/env python
"""Ingest one or more PDFs into data/processed."""

from __future__ import annotations

import argparse
from pathlib import Path

from pdf_vlm.ocr.paddle_structure import ingest_pdf
from pdf_vlm.utils.io import load_named_config
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ocr_cfg = load_named_config("ocr/pp_structure_v3.yaml")
    for pdf in args.pdfs:
        art = ingest_pdf(pdf, ocr_cfg=ocr_cfg, force=args.force, use_stub=args.stub)
        logger.info("doc_id=%s pages=%d", art.doc_id, art.num_pages)


if __name__ == "__main__":
    main()
