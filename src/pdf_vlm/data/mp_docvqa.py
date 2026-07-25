"""MP-DocVQA local loader.

Expected layout under data/raw/mp_docvqa/ (or --root):

  data/raw/mp_docvqa/
    val_questions.json | questions.json
    documents/
      <doc_id>/
        page_0000.png ...
      OR <doc_id>.pdf
    documents.json   # optional: [{doc_id, num_pages, ...}]

Questions JSON item fields (flexible):
  questionId / id, question, answers / answer,
  doc_id / ucsf_document_id / document_id,
  page_ids / evidence_pages / page / page_no
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from pdf_vlm.data.base import BaseDataset
from pdf_vlm.data.question_types import as_int_list, infer_question_type, normalize_modalities
from pdf_vlm.schemas import DatasetBundle, DatasetDocument, PageRef, QAPair, QuestionType
from pdf_vlm.utils.io import load_json, resolve_path
from pdf_vlm.utils.logging import get_logger

logger = get_logger("data.mp_docvqa")


def _discover_pages(doc_id: str, docs_root: Path) -> tuple[list[PageRef], str | None]:
    """Resolve page images or a single PDF for a document id."""
    pdf = docs_root / f"{doc_id}.pdf"
    if pdf.exists():
        n = None
        try:
            import fitz

            n = len(fitz.open(pdf))
        except Exception:
            n = None
        if n is None:
            return [], str(pdf)
        pages = [PageRef(page_id=i, pdf_path=str(pdf)) for i in range(n)]
        return pages, str(pdf)

    folder = docs_root / doc_id
    if folder.is_dir():
        images = sorted(
            list(folder.glob("page_*.png"))
            + list(folder.glob("page_*.jpg"))
            + list(folder.glob("*.png"))
            + list(folder.glob("*.jpg"))
        )
        # de-dup while preserving order
        seen: set[str] = set()
        uniq: list[Path] = []
        for p in images:
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        pages = [PageRef(page_id=i, image_path=str(p)) for i, p in enumerate(uniq)]
        return pages, str(folder)

    return [], None


def _load_question_items(root: Path, split: str) -> list[dict[str, Any]]:
    candidates = [
        root / f"{split}_questions.json",
        root / f"qas_{split}.json",
        root / "questions.json",
        root / f"{split}.json",
    ]
    for path in candidates:
        if path.exists():
            raw = load_json(path)
            if isinstance(raw, list):
                return raw
            if isinstance(raw, dict):
                return list(raw.get("data") or raw.get("questions") or raw.get("qas") or [])
    return []


def load_mp_docvqa_bundle(
    *,
    root: str | Path | None = None,
    split: str = "val",
    limit_questions: int | None = None,
    limit_docs: int | None = None,
) -> DatasetBundle:
    root = Path(root or resolve_path("data/raw/mp_docvqa"))
    docs_root = root / "documents"
    items = _load_question_items(root, split)
    if not items:
        logger.warning("MP-DocVQA questions not found under %s", root)
        return DatasetBundle(name="mp_docvqa", documents=[], split=split, meta={"root": str(root)})

    # optional document metadata
    meta_pages: dict[str, int] = {}
    meta_path = root / "documents.json"
    if meta_path.exists():
        meta_raw = load_json(meta_path)
        meta_list = meta_raw if isinstance(meta_raw, list) else meta_raw.get("documents") or []
        for row in meta_list:
            did = str(row.get("doc_id") or row.get("document_id") or "")
            if did:
                meta_pages[did] = int(row.get("num_pages") or row.get("n_pages") or 0)

    by_doc: dict[str, list[QAPair]] = defaultdict(list)
    for i, item in enumerate(items):
        doc_id = str(
            item.get("doc_id")
            or item.get("ucsf_document_id")
            or item.get("document_id")
            or item.get("docId")
            or f"unknown_{i}"
        )
        answers = item.get("answers") or item.get("answer") or []
        if isinstance(answers, str):
            answers = [answers]
        evidence = as_int_list(
            item.get("page_ids")
            or item.get("evidence_pages")
            or item.get("page")
            or item.get("page_no")
            or item.get("answer_page")
        )
        # MP-DocVQA: evidence is typically a single page
        modalities = normalize_modalities(item.get("evidence_modality") or item.get("modality") or ["text"])
        qtype = infer_question_type(
            evidence_pages=evidence,
            evidence_modality=modalities,
            explicit=item.get("question_type"),
        )
        # If only one page, keep text/table/chart; do not force cross-page
        if len(set(evidence)) <= 1 and qtype == QuestionType.CROSS_PAGE:
            qtype = QuestionType.TEXT

        qa = QAPair(
            qa_id=str(item.get("questionId") or item.get("question_id") or item.get("id") or f"mp_{i}"),
            question=str(item.get("question") or item.get("query") or ""),
            answer=[str(a) for a in answers],
            question_type=qtype,
            evidence_pages=evidence,
            evidence_modality=modalities,
            meta={"split": split, "raw_keys": sorted(item.keys())},
        )
        by_doc[doc_id].append(qa)
        if limit_questions is not None and sum(len(v) for v in by_doc.values()) >= limit_questions:
            break

    documents: list[DatasetDocument] = []
    for doc_id, qas in by_doc.items():
        pages, source = _discover_pages(doc_id, docs_root) if docs_root.exists() else ([], None)
        num_pages = len(pages) if pages else meta_pages.get(doc_id)
        # if we know page count but no files, synthesize page refs
        if not pages and num_pages:
            pages = [PageRef(page_id=i) for i in range(int(num_pages))]
        documents.append(
            DatasetDocument(
                doc_id=doc_id,
                pages=pages,
                qa_pairs=qas,
                source_path=source,
                source_dataset="mp_docvqa",
                num_pages=num_pages or len(pages) or None,
                meta={"split": split},
            )
        )
        if limit_docs is not None and len(documents) >= limit_docs:
            break

    return DatasetBundle(
        name="mp_docvqa",
        documents=documents,
        split=split,
        meta={"root": str(root), "n_raw_questions": len(items)},
    )


class MPDocVQADataset(BaseDataset):
    name = "mp_docvqa"

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "val",
        limit_questions: int | None = None,
        limit_docs: int | None = None,
    ):
        self.root = root
        self.split = split
        self.limit_questions = limit_questions
        self.limit_docs = limit_docs

    def load(self) -> DatasetBundle:
        return load_mp_docvqa_bundle(
            root=self.root,
            split=self.split,
            limit_questions=self.limit_questions,
            limit_docs=self.limit_docs,
        )


def load_mp_docvqa(
    split: str = "val",
    root: str | Path | None = None,
    limit: int | None = None,
):
    """Backward-compatible flat QAExample list."""
    from pdf_vlm.data.base import bundle_to_examples

    bundle = load_mp_docvqa_bundle(root=root, split=split, limit_questions=limit)
    return bundle_to_examples(bundle)
