#!/usr/bin/env python
"""Ingest + index existing data/custom/{5,20,50,100} packs (Colab / no full source PDF).

Uses stub OCR + PDF text enrich by default, and hashing embedder for low RAM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdf_vlm.index.hierarchical_indexer import build_hierarchical_index
from pdf_vlm.index.text_indexer import build_text_only_index
from pdf_vlm.ocr.paddle_structure import ingest_pdf, load_artifact
from pdf_vlm.rag.text_only import enrich_artifact_from_pdf_text
from pdf_vlm.utils.io import load_json, load_named_config, resolve_path, save_json
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def _manifest_docs(bucket: int) -> list[dict]:
    man = resolve_path(f"data/custom/{bucket}/manifest.json")
    if not man.exists():
        return []
    data = load_json(man)
    return list(data.get("documents") or [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buckets", default="5,20")
    parser.add_argument("--stub", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enrich-pdf-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hash-embedder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]
    page_name = "retrieval/page_text_hash.yaml" if args.hash_embedder else "retrieval/page_text.yaml"
    hier_name = "retrieval/hier_text_hash.yaml" if args.hash_embedder else "retrieval/hier_text.yaml"
    ocr_cfg = load_named_config("ocr/pp_structure_v3.yaml")
    prepared: list[dict] = []

    for n in buckets:
        docs = _manifest_docs(n)
        if not docs:
            logger.warning("No manifest documents for bucket=%d", n)
            continue
        for d in docs:
            pdf = resolve_path(f"data/custom/{n}/{d['pdf']}")
            if not pdf.exists():
                logger.warning("Missing PDF %s", pdf)
                continue
            art = ingest_pdf(str(pdf), ocr_cfg=ocr_cfg, force=args.force, use_stub=bool(args.stub))
            if args.enrich_pdf_text:
                enrich_artifact_from_pdf_text(art.doc_id)
                art = load_artifact(art.doc_id)
            page_cfg = load_named_config(page_name)
            hier_cfg = load_named_config(hier_name)
            page_out = resolve_path(f"indices/{art.doc_id}/page_text")
            hier_out = resolve_path(f"indices/{art.doc_id}/hier_text")
            build_text_only_index([art], page_out, mode="page", retrieval_cfg=page_cfg)
            build_hierarchical_index(
                [art],
                hier_out,
                modality="text",
                retrieval_cfg=hier_cfg,
                hierarchy_strategy="auto",
                fine_unit="page",
            )
            item = {
                "bucket": n,
                "doc_id": art.doc_id,
                "pdf": str(pdf),
                "stub": bool(art.meta.get("stub")),
                "backend": art.meta.get("ocr_backend"),
            }
            prepared.append(item)
            logger.info("ready %s", item)

    out = resolve_path("data/custom/colab_prepared.json")
    save_json(out, {"items": prepared})
    print(json.dumps(prepared, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
