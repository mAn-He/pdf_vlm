#!/usr/bin/env python
"""Prepare Hyundai WIA quarterly report (QA_report_HW.pdf) as custom 5/10/20/50/100 packs.

Creates length-truncated PDFs (from cover page), questions.json with gold answers,
optionally ingests + builds retrieval indexes.

Evidence pages are remapped relative to the truncated PDF (0-indexed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fitz

from pdf_vlm.utils.io import ensure_dir, project_root, resolve_path, save_json
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()

# Absolute 0-based page indices in the full QA_report_HW.pdf
COVER_PAGE = 4  # 표지 (회사명)
START = COVER_PAGE  # all buckets begin at cover so early pages are informative

# Absolute evidence pages in full PDF
ABS = {
    "company": 4,
    "overview": 8,
    "products": 8,
    "contracts": 15,
    "domestic_prod": 19,
    "global_prod": 20,
    "finance_summary": 31,
    "finance_pl": 32,
}


def _doc_id_for(pdf_path: Path) -> str:
    digest = hashlib.sha1(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{pdf_path.stem}_{digest}"


def _rel(abs_page: int) -> int:
    return abs_page - START


def slice_pdf(src: Path, dst: Path, start: int, n_pages: int) -> int:
    doc = fitz.open(src)
    out = fitz.open()
    end = min(start + n_pages, len(doc))
    out.insert_pdf(doc, from_page=start, to_page=end - 1)
    ensure_dir(dst.parent)
    out.save(dst)
    n = len(out)
    out.close()
    doc.close()
    return n


def build_questions(doc_id: str, bucket: int) -> list[dict]:
    """Questions whose evidence fits in [0, bucket). Same themes for every bucket when possible."""

    def ok(abs_page: int) -> bool:
        return _rel(abs_page) < bucket

    qs: list[dict] = []

    # Always available from cover (rel 0) when START=4
    qs.append(
        {
            "example_id": f"hw{bucket}_q_company",
            "doc_id": doc_id,
            "question": "이 분기보고서의 회사명은 무엇인가?",
            "answers": ["현대위아 주식회사", "현대위아"],
            "evidence_pages": [_rel(ABS["company"])],
            "question_type": "text",
            "evidence_modality": ["text"],
        }
    )

    if ok(ABS["overview"]):
        qs.append(
            {
                "example_id": f"hw{bucket}_q_overview",
                "doc_id": doc_id,
                "question": "당사(현대위아)의 사업의 개요에서 주요 사업 부문은 무엇인가?",
                "answers": [
                    "차량부품, 모빌리티솔루션 및 특수사업",
                    "차량부품, 모빌리티솔루션, 특수사업",
                ],
                "evidence_pages": [_rel(ABS["overview"])],
                "question_type": "text",
                "evidence_modality": ["text"],
            }
        )

    if ok(ABS["products"]):
        qs.append(
            {
                "example_id": f"hw{bucket}_q_products",
                "doc_id": doc_id,
                "question": "차량부품 부문이 글로벌 완성차 업체를 대상으로 생산·공급하는 주요 제품·서비스는 무엇인가?",
                "answers": [
                    "엔진, 모듈, 등속조인트, 4WD 부품 및 열관리 관련 부품",
                    "엔진, 모듈, 등속조인트, 4WD, 열관리 부품",
                ],
                "evidence_pages": [_rel(ABS["products"])],
                "question_type": "text",
                "evidence_modality": ["text"],
            }
        )
        qs.append(
            {
                "example_id": f"hw{bucket}_q_mobility",
                "doc_id": doc_id,
                "question": "모빌리티솔루션 부문은 어떤 분야를 중심으로 솔루션을 제공하는가?",
                "answers": [
                    "로봇 및 스마트팩토리",
                    "로봇, 스마트팩토리",
                ],
                "evidence_pages": [_rel(ABS["products"])],
                "question_type": "text",
                "evidence_modality": ["text"],
            }
        )

    if ok(ABS["contracts"]):
        qs.append(
            {
                "example_id": f"hw{bucket}_q_contract",
                "doc_id": doc_id,
                "question": "공작기계사업부문 매각 계약의 거래금액은 얼마인가?",
                "answers": ["3,400억원", "3400억원", "3,400억"],
                "evidence_pages": [_rel(ABS["contracts"])],
                "question_type": "text",
                "evidence_modality": ["text", "table"],
            }
        )
        qs.append(
            {
                "example_id": f"hw{bucket}_q_contract_counterparty",
                "doc_id": doc_id,
                "question": "공작기계사업부문 매각의 거래 상대방은 누구인가?",
                "answers": [
                    "에이치엠티테크 & 에이치엠티솔루션 컨소시엄",
                    "에이치엠티테크와 에이치엠티솔루션 컨소시엄",
                ],
                "evidence_pages": [_rel(ABS["contracts"])],
                "question_type": "text",
                "evidence_modality": ["text"],
            }
        )

    if ok(ABS["domestic_prod"]):
        qs.append(
            {
                "example_id": f"hw{bucket}_q_kr_prod",
                "doc_id": doc_id,
                "question": "2026년 1분기 국내 자동차 생산 실적은 몇 대인가?",
                "answers": ["1,025,981대", "1025981대", "1,025,981"],
                "evidence_pages": [_rel(ABS["domestic_prod"])],
                "question_type": "table",
                "evidence_modality": ["text", "table"],
            }
        )

    if ok(ABS["global_prod"]):
        qs.append(
            {
                "example_id": f"hw{bucket}_q_global_prod",
                "doc_id": doc_id,
                "question": "2026년 글로벌 자동차 생산실적 및 전망에서, 2026년 글로벌 자동차 생산량은 어떻게 전망되는가?",
                "answers": [
                    "전년 대비 제한적인 성장세",
                    "제한적인 성장세가 전망",
                    "전년 대비 제한적 성장",
                ],
                "evidence_pages": [_rel(ABS["global_prod"])],
                "question_type": "text",
                "evidence_modality": ["text"],
            }
        )

    if ok(ABS["finance_summary"]):
        qs.append(
            {
                "example_id": f"hw{bucket}_q_assets",
                "doc_id": doc_id,
                "question": "요약 연결재무제표 기준 제51기 1분기(2026년 3월말) 자산총계는 얼마인가? (단위: 백만원)",
                "answers": ["7,219,019", "7219019", "7,219,019백만원"],
                "evidence_pages": [_rel(ABS["finance_summary"])],
                "question_type": "table",
                "evidence_modality": ["text", "table"],
            }
        )
        qs.append(
            {
                "example_id": f"hw{bucket}_q_liab",
                "doc_id": doc_id,
                "question": "요약 연결재무제표 기준 제51기 1분기(2026년 3월말) 부채총계는 얼마인가? (단위: 백만원)",
                "answers": ["3,071,872", "3071872"],
                "evidence_pages": [_rel(ABS["finance_summary"])],
                "question_type": "table",
                "evidence_modality": ["text", "table"],
            }
        )

    if ok(ABS["finance_pl"]):
        qs.append(
            {
                "example_id": f"hw{bucket}_q_revenue",
                "doc_id": doc_id,
                "question": "별도 요약손익 기준 제51기 1분기(2026.1.1~2026.3.31) 매출액은 얼마인가? (단위: 백만원)",
                "answers": ["1,918,322", "1918322"],
                "evidence_pages": [_rel(ABS["finance_pl"])],
                "question_type": "table",
                "evidence_modality": ["text", "table"],
            }
        )

    return qs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        type=Path,
        default=project_root() / "QA_report_HW.pdf",
        help="Full Hyundai WIA quarterly report PDF",
    )
    parser.add_argument("--buckets", default="5,10,20,50,100")
    parser.add_argument("--ingest", action="store_true", help="Ingest PDFs with PaddleOCR (real OCR by default)")
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use stub OCR instead of PaddleOCR (offline only)",
    )
    parser.add_argument(
        "--enrich-pdf-text",
        action="store_true",
        help="After OCR, merge PDF embedded text layer into page markdown",
    )
    parser.add_argument("--build-index", action="store_true", help="Build page_text + hier_text indexes")
    parser.add_argument(
        "--hash-embedder",
        action="store_true",
        help="Use hashing embedder (Colab/low-RAM); writes to page_text / hier_text dirs",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    src = Path(args.pdf)
    if not src.exists():
        raise SystemExit(f"PDF not found: {src}")

    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]
    prepared: list[dict] = []

    for n in buckets:
        out_dir = ensure_dir(resolve_path(f"data/custom/{n}"))
        # remove old demo pdf noise? keep acme if present; write HW pdf alongside
        pdf_name = f"hyundai_wia_qa_report_{n}p.pdf"
        dst = out_dir / pdf_name
        got = slice_pdf(src, dst, START, n)
        if got != n:
            logger.warning("Requested %d pages but got %d (PDF shorter?)", n, got)

        doc_id = _doc_id_for(dst)
        questions = build_questions(doc_id, n)
        save_json(out_dir / "questions.json", questions)
        save_json(
            out_dir / "manifest.json",
            {
                "bucket": n,
                "source_pdf": str(src.resolve()),
                "slice_start_abs_page": START,
                "documents": [
                    {
                        "doc_id": doc_id,
                        "pdf": pdf_name,
                        "num_pages": got,
                        "stem": dst.stem,
                    }
                ],
                "n_questions": len(questions),
            },
        )
        prepared.append({"bucket": n, "pdf": str(dst), "doc_id": doc_id, "n_questions": len(questions)})
        logger.info("bucket=%d pdf=%s doc_id=%s questions=%d", n, dst.name, doc_id, len(questions))

    save_json(resolve_path("data/custom/hw_prepared.json"), {"items": prepared})

    if args.ingest:
        from pdf_vlm.ocr.paddle_structure import ingest_pdf, paddle_available
        from pdf_vlm.utils.io import load_named_config

        flags = paddle_available()
        if not args.stub and not flags.get("paddleocr"):
            raise SystemExit(
                "PaddleOCR not importable in this Python.\n"
                "Install into the SAME env you run this script with:\n"
                "  python -m pip install paddlepaddle paddleocr\n"
                "Or pass --stub for offline placeholder OCR."
            )
        logger.info("paddle_available=%s stub=%s", flags, args.stub)

        ocr_cfg = load_named_config("ocr/pp_structure_v3.yaml")
        for item in prepared:
            art = ingest_pdf(
                item["pdf"],
                ocr_cfg=ocr_cfg,
                force=args.force,
                use_stub=bool(args.stub),
            )
            if args.enrich_pdf_text:
                from pdf_vlm.rag.text_only import enrich_artifact_from_pdf_text

                enrich_artifact_from_pdf_text(art.doc_id)
            assert art.doc_id == item["doc_id"], (art.doc_id, item["doc_id"])
            logger.info(
                "ingested %s (%d pages, stub=%s, backend=%s)",
                art.doc_id,
                art.num_pages,
                art.meta.get("stub"),
                art.meta.get("ocr_backend"),
            )

    if args.build_index:
        from pdf_vlm.index.hierarchical_indexer import build_hierarchical_index
        from pdf_vlm.index.text_indexer import build_text_only_index
        from pdf_vlm.ocr.paddle_structure import load_artifact
        from pdf_vlm.utils.io import load_named_config

        page_cfg_name = "retrieval/page_text_hash.yaml" if args.hash_embedder else "retrieval/page_text.yaml"
        hier_cfg_name = "retrieval/hier_text_hash.yaml" if args.hash_embedder else "retrieval/hier_text.yaml"
        for item in prepared:
            doc_id = item["doc_id"]
            art = load_artifact(doc_id)
            page_cfg = load_named_config(page_cfg_name)
            hier_cfg = load_named_config(hier_cfg_name)
            page_out = resolve_path(f"indices/{doc_id}/page_text")
            hier_out = resolve_path(f"indices/{doc_id}/hier_text")
            build_text_only_index([art], page_out, mode="page", retrieval_cfg=page_cfg)
            build_hierarchical_index(
                [art],
                hier_out,
                modality="text",
                retrieval_cfg=hier_cfg,
                hierarchy_strategy="auto",
                fine_unit="page",
            )
            logger.info("indexes ready for %s (embedder=%s)", doc_id, "hash" if args.hash_embedder else "bge")

    print(json.dumps(prepared, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
