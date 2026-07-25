#!/usr/bin/env python
"""Create a tiny multi-page PDF + questions for custom_5 MVP."""

from __future__ import annotations

from pathlib import Path

import fitz

from pdf_vlm.utils.io import ensure_dir, project_root, save_json
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()

PAGES = [
    "Company Overview\nAcme Corp was founded in 1998 in Seoul.\nEmployees: 420.",
    "Products\nAcme sells industrial sensors and OCR scanners.\nFlagship product: VisionX-4.",
    "Financials\nRevenue 2024: $12.5M\nOperating margin: 18%.",
    "Offices\nHQ: Seoul\nBranch: Busan\nSupport email: support@acme.example",
    "FAQ\nQ: What is the warranty?\nA: VisionX-4 includes a 24-month warranty.",
]

QUESTIONS = [
    {
        "example_id": "c5_q1",
        "doc_id": "PLACEHOLDER",
        "question": "When was Acme Corp founded?",
        "answers": ["1998"],
        "evidence_pages": [0],
    },
    {
        "example_id": "c5_q2",
        "doc_id": "PLACEHOLDER",
        "question": "What is the flagship product?",
        "answers": ["VisionX-4"],
        "evidence_pages": [1],
    },
    {
        "example_id": "c5_q3",
        "doc_id": "PLACEHOLDER",
        "question": "How long is the VisionX-4 warranty?",
        "answers": ["24-month", "24 months"],
        "evidence_pages": [4],
    },
]


def main() -> None:
    out_dir = ensure_dir(project_root() / "data" / "custom" / "5")
    pdf_path = out_dir / "acme_demo_5pages.pdf"
    doc = fitz.open()
    for text in PAGES:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), text, fontsize=14)
    doc.save(pdf_path)
    doc.close()

    # doc_id filled after ingest; keep filename stem hint
    qs = []
    for q in QUESTIONS:
        item = dict(q)
        item["doc_id"] = "acme_demo_5pages"
        qs.append(item)
    save_json(out_dir / "questions.json", qs)
    logger.info("Wrote %s and questions.json", pdf_path)


if __name__ == "__main__":
    main()
