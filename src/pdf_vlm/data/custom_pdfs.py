"""Custom PDF dataset loader for 5/20/50/100 page buckets.

Expected layout:
  data/custom/<bucket>/
    *.pdf
    questions.json | questions.jsonl
    manifest.json   # optional

questions.json item:
  {
    "example_id"|"qa_id": "...",
    "doc_id": "acme_demo_5pages",   # stem or full processed id
    "question": "...",
    "answers"|"answer": ["..."],
    "evidence_pages": [0],
    "question_type": "text|table|chart|cross-page",
    ...
  }

manifest.json (optional):
  {
    "bucket": 5,
    "documents": [{"doc_id": "...", "pdf": "file.pdf", "num_pages": 5}]
  }
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from pdf_vlm.data.base import BaseDataset
from pdf_vlm.data.question_types import as_int_list, infer_question_type, normalize_modalities
from pdf_vlm.schemas import DatasetBundle, DatasetDocument, PageRef, QAPair
from pdf_vlm.utils.io import load_json, resolve_path
from pdf_vlm.utils.logging import get_logger

logger = get_logger("data.custom")

CUSTOM_BUCKETS = (5, 20, 50, 100)


def dataset_dir(name: str) -> Path:
    if name.startswith("custom_"):
        n = name.split("_", 1)[1]
        return resolve_path(f"data/custom/{n}")
    if name.isdigit():
        return resolve_path(f"data/custom/{name}")
    return resolve_path(f"data/custom/{name}")


def parse_bucket(name: str) -> int | None:
    if name.startswith("custom_"):
        part = name.split("_", 1)[1]
        return int(part) if part.isdigit() else None
    if name.isdigit():
        return int(name)
    return None


def _count_pdf_pages(pdf_path: Path) -> int:
    try:
        import fitz

        doc = fitz.open(pdf_path)
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


def _load_question_rows(root: Path) -> list[dict[str, Any]]:
    jsonl = root / "questions.jsonl"
    json_path = root / "questions.json"
    rows: list[dict[str, Any]] = []
    if jsonl.exists():
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(load_json_line(line))
        return rows
    if json_path.exists():
        raw = load_json(json_path)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            return list(raw.get("questions") or raw.get("qas") or [])
        raise ValueError(f"Expected list in {json_path}")
    return rows


def load_json_line(line: str) -> dict[str, Any]:
    import json

    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError("questions.jsonl lines must be objects")
    return obj


def _resolve_pdf_for_doc(root: Path, doc_id: str, pdfs: dict[str, Path]) -> Path | None:
    if doc_id in pdfs:
        return pdfs[doc_id]
    # fuzzy: stem prefix
    for stem, path in pdfs.items():
        if doc_id.startswith(stem) or stem.startswith(doc_id):
            return path
    return None


def load_custom_bundle(
    dataset: str = "custom_5",
    *,
    root: str | Path | None = None,
    limit_questions: int | None = None,
) -> DatasetBundle:
    root = Path(root) if root else dataset_dir(dataset)
    bucket = parse_bucket(dataset) or parse_bucket(root.name)
    if not root.exists():
        logger.warning("Custom dataset dir missing: %s", root)
        return DatasetBundle(name=dataset, documents=[], meta={"root": str(root), "bucket": bucket})

    pdf_paths = sorted(root.glob("*.pdf"))
    pdfs = {p.stem: p for p in pdf_paths}
    rows = _load_question_rows(root)

    manifest_docs: dict[str, dict[str, Any]] = {}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        man = load_json(manifest_path)
        for row in man.get("documents") or []:
            manifest_docs[str(row.get("doc_id"))] = row

    by_doc: dict[str, list[QAPair]] = defaultdict(list)
    for i, item in enumerate(rows):
        doc_id = str(item.get("doc_id") or item.get("document_id") or "")
        if not doc_id and len(pdfs) == 1:
            doc_id = next(iter(pdfs.keys()))
        if not doc_id:
            doc_id = f"custom_doc_{i}"

        answers = item.get("answers") or item.get("answer") or []
        if isinstance(answers, str):
            answers = [answers]
        evidence = as_int_list(item.get("evidence_pages") or item.get("page_ids") or item.get("page"))
        modalities = normalize_modalities(item.get("evidence_modality") or item.get("evidence_sources") or [])
        una = bool(item.get("unanswerable", False))
        qtype = infer_question_type(
            evidence_pages=evidence,
            evidence_modality=modalities,
            unanswerable=una,
            explicit=item.get("question_type"),
        )
        qa = QAPair(
            qa_id=str(item.get("example_id") or item.get("qa_id") or item.get("id") or f"{dataset}_q{i}"),
            question=str(item.get("question") or ""),
            answer=[str(a) for a in answers],
            question_type=qtype,
            evidence_pages=evidence,
            unanswerable=una,
            evidence_modality=modalities,
            meta={"bucket": bucket},
        )
        by_doc[doc_id].append(qa)
        if limit_questions is not None and sum(len(v) for v in by_doc.values()) >= limit_questions:
            break

    # include PDFs that have no questions yet (still useful for ingest stats)
    for stem in pdfs:
        by_doc.setdefault(stem, [])

    documents: list[DatasetDocument] = []
    for doc_id, qas in by_doc.items():
        pdf = _resolve_pdf_for_doc(root, doc_id, pdfs)
        pages: list[PageRef] = []
        num_pages = None
        source = None
        if pdf is not None:
            source = str(pdf)
            num_pages = _count_pdf_pages(pdf)
            pages = [PageRef(page_id=i, pdf_path=str(pdf)) for i in range(num_pages)]
        elif doc_id in manifest_docs:
            num_pages = int(manifest_docs[doc_id].get("num_pages") or 0) or None
            if num_pages:
                pages = [PageRef(page_id=i) for i in range(num_pages)]
            pdf_name = manifest_docs[doc_id].get("pdf")
            if pdf_name:
                source = str(root / pdf_name)

        if bucket is not None and num_pages and num_pages != bucket:
            logger.warning(
                "Document %s has %s pages but bucket=%s (kept; check packing)",
                doc_id,
                num_pages,
                bucket,
            )

        documents.append(
            DatasetDocument(
                doc_id=doc_id,
                pages=pages,
                qa_pairs=qas,
                source_path=source,
                source_dataset=dataset,
                num_pages=num_pages or len(pages) or bucket,
                meta={"bucket": bucket},
            )
        )

    # drop empty docs with no qa and no pdf
    documents = [d for d in documents if d.qa_pairs or d.source_path]

    return DatasetBundle(
        name=dataset,
        documents=documents,
        split=f"custom_{bucket}" if bucket else "custom",
        meta={"root": str(root), "bucket": bucket, "n_pdfs": len(pdf_paths)},
    )


class CustomPDFDataset(BaseDataset):
    def __init__(self, dataset: str = "custom_5", root: str | Path | None = None, limit_questions: int | None = None):
        self.dataset = dataset
        self.root = root
        self.limit_questions = limit_questions
        self.name = dataset

    def load(self) -> DatasetBundle:
        return load_custom_bundle(self.dataset, root=self.root, limit_questions=self.limit_questions)


def load_custom_qa(dataset: str = "custom_5"):
    from pdf_vlm.data.base import bundle_to_examples

    return bundle_to_examples(load_custom_bundle(dataset))


def list_custom_pdfs(dataset: str = "custom_5") -> list[Path]:
    root = dataset_dir(dataset)
    return sorted(root.glob("*.pdf"))
