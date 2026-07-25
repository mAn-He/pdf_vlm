#!/usr/bin/env python
"""Check PaddleOCR is importable and run a 1-page Structure/OCR smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

from pdf_vlm.ocr.paddle_structure import ingest_pdf, paddle_available
from pdf_vlm.utils.io import load_named_config, resolve_path
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        type=Path,
        default=resolve_path("data/custom/5/hyundai_wia_qa_report_5p.pdf"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    flags = paddle_available()
    logger.info("paddle_available=%s", flags)
    if not flags.get("paddleocr"):
        raise SystemExit(
            "FAIL: paddleocr not importable in this Python.\n"
            "Fix: activate the env where you installed it, then:\n"
            "  python -c \"import paddleocr; print(paddleocr.__version__)\"\n"
            "  python -m pip install paddlepaddle paddleocr"
        )

    if not args.pdf.exists():
        raise SystemExit(f"Missing PDF: {args.pdf}")

    cfg = load_named_config("ocr/pp_structure_v3.yaml")
    art = ingest_pdf(args.pdf, ocr_cfg=cfg, force=True, use_stub=False)
    nonempty = sum(1 for p in art.pages if (p.markdown or "").strip())
    logger.info(
        "OK doc_id=%s pages=%d nonempty_markdown=%d backend=%s stub=%s",
        art.doc_id,
        art.num_pages,
        nonempty,
        art.meta.get("ocr_backend"),
        art.meta.get("stub"),
    )
    if art.meta.get("stub"):
        raise SystemExit("FAIL: artifact still marked stub")
    if nonempty == 0:
        raise SystemExit("FAIL: OCR produced empty markdown on all pages")
    preview = next((p.markdown[:200] for p in art.pages if p.markdown.strip()), "")
    print(preview)
    print(f"SUCCESS backend={art.meta.get('ocr_backend')}")


if __name__ == "__main__":
    main()
