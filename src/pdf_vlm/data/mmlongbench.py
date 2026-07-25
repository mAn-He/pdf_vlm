"""MMLongBench-Doc subset loader.

Full benchmark is large; this loader always works with an explicit subset policy.

Expected layout:
  data/raw/mmlongbench/
    questions.json
    documents/<doc_id>.pdf  OR documents/<doc_id>/page_*.png
    subset.yaml             # optional default subset policy

Subset selection (any combination):
  - max_docs / max_questions
  - question_types: [text, table, chart, cross-page, ...]
  - evidence_modalities: [text, table, chart, image, layout]
  - include_unanswerable: bool
  - doc_ids: explicit allow-list
  - seed: for reproducible sampling
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from pdf_vlm.data.base import BaseDataset
from pdf_vlm.data.mp_docvqa import _discover_pages
from pdf_vlm.data.question_types import as_int_list, infer_question_type, normalize_modalities
from pdf_vlm.schemas import DatasetBundle, DatasetDocument, PageRef, QAPair, QuestionType
from pdf_vlm.utils.io import load_json, load_yaml, resolve_path
from pdf_vlm.utils.logging import get_logger

logger = get_logger("data.mmlongbench")


def _load_items(root: Path) -> list[dict[str, Any]]:
    for name in ("questions.json", "qas.json", "annotations.json"):
        path = root / name
        if path.exists():
            raw = load_json(path)
            if isinstance(raw, list):
                return raw
            if isinstance(raw, dict):
                return list(raw.get("data") or raw.get("questions") or raw.get("qas") or [])
    return []


def _default_subset_cfg(root: Path) -> dict[str, Any]:
    for name in ("subset.yaml", "subset.yml", "subset.json"):
        path = root / name
        if path.exists():
            if path.suffix == ".json":
                return load_json(path)
            return load_yaml(path)
    return {
        "max_docs": 20,
        "max_questions": 100,
        "include_unanswerable": True,
        "seed": 42,
    }


def _match_types(qa_type: QuestionType, wanted: Sequence[str] | None) -> bool:
    if not wanted:
        return True
    return qa_type.value.lower() in {w.lower() for w in wanted}


def _match_modalities(mods: Sequence[str], wanted: Sequence[str] | None) -> bool:
    if not wanted:
        return True
    wanted_l = {w.lower() for w in wanted}
    return bool(wanted_l.intersection({m.lower() for m in mods}))


def load_mmlongbench_bundle(
    *,
    root: str | Path | None = None,
    subset: dict[str, Any] | None = None,
    max_docs: int | None = None,
    max_questions: int | None = None,
    question_types: Sequence[str] | None = None,
    evidence_modalities: Sequence[str] | None = None,
    include_unanswerable: bool | None = None,
    doc_ids: Sequence[str] | None = None,
    seed: int | None = None,
) -> DatasetBundle:
    root = Path(root or resolve_path("data/raw/mmlongbench"))
    cfg = _default_subset_cfg(root)
    if subset:
        cfg.update(subset)

    # CLI/function overrides win
    if max_docs is not None:
        cfg["max_docs"] = max_docs
    if max_questions is not None:
        cfg["max_questions"] = max_questions
    if question_types is not None:
        cfg["question_types"] = list(question_types)
    if evidence_modalities is not None:
        cfg["evidence_modalities"] = list(evidence_modalities)
    if include_unanswerable is not None:
        cfg["include_unanswerable"] = include_unanswerable
    if doc_ids is not None:
        cfg["doc_ids"] = list(doc_ids)
    if seed is not None:
        cfg["seed"] = seed

    items = _load_items(root)
    if not items:
        logger.warning("MMLongBench-Doc questions not found under %s", root)
        return DatasetBundle(name="mmlongbench", documents=[], meta={"root": str(root), "subset": cfg})

    allow_docs = set(str(x) for x in (cfg.get("doc_ids") or []))
    qtypes = cfg.get("question_types")
    emods = cfg.get("evidence_modalities")
    keep_una = bool(cfg.get("include_unanswerable", True))

    parsed: list[tuple[str, QAPair]] = []
    for i, item in enumerate(items):
        doc_id = str(item.get("doc_id") or item.get("document_id") or item.get("doc") or f"mmlb_doc_{i}")
        if allow_docs and doc_id not in allow_docs:
            continue

        answers = item.get("answer") or item.get("answers") or []
        if not isinstance(answers, list):
            answers = [answers]
        evidence = as_int_list(
            item.get("evidence_pages")
            or item.get("page_ids")
            or item.get("doc_page")
            or item.get("pages")
        )
        modalities = normalize_modalities(
            item.get("evidence_sources")
            or item.get("evidence_modality")
            or item.get("evidence_source")
            or item.get("modality")
        )
        una = bool(
            item.get("unanswerable")
            or str(item.get("answer_type", "")).lower() == "unanswerable"
            or str(item.get("answer", "")).lower() in {"unanswerable", "none"}
        )
        if una and not keep_una:
            continue

        qtype = infer_question_type(
            evidence_pages=evidence,
            evidence_modality=modalities,
            unanswerable=una,
            explicit=item.get("question_type"),
        )
        if not _match_types(qtype, qtypes):
            continue
        if not _match_modalities(modalities, emods):
            continue

        qa = QAPair(
            qa_id=str(item.get("id") or item.get("question_id") or f"mmlb_{i}"),
            question=str(item.get("question") or ""),
            answer=[str(a) for a in answers if a is not None],
            question_type=qtype,
            evidence_pages=evidence,
            unanswerable=una,
            answer_type=item.get("answer_type"),
            evidence_modality=modalities,
            meta={"source": "mmlongbench"},
        )
        parsed.append((doc_id, qa))

    rng = random.Random(int(cfg.get("seed", 42)))
    rng.shuffle(parsed)

    max_q = cfg.get("max_questions")
    if max_q is not None:
        parsed = parsed[: int(max_q)]

    by_doc: dict[str, list[QAPair]] = defaultdict(list)
    for doc_id, qa in parsed:
        by_doc[doc_id].append(qa)

    doc_ids_ordered = list(by_doc.keys())
    rng.shuffle(doc_ids_ordered)
    max_d = cfg.get("max_docs")
    if max_d is not None:
        doc_ids_ordered = doc_ids_ordered[: int(max_d)]

    docs_root = root / "documents"
    documents: list[DatasetDocument] = []
    for doc_id in doc_ids_ordered:
        pages, source = _discover_pages(doc_id, docs_root) if docs_root.exists() else ([], None)
        qas = by_doc[doc_id]
        # synthesize page refs from evidence if files missing
        if not pages:
            max_page = max((max(qa.evidence_pages) for qa in qas if qa.evidence_pages), default=-1)
            if max_page >= 0:
                pages = [PageRef(page_id=i) for i in range(max_page + 1)]
        documents.append(
            DatasetDocument(
                doc_id=doc_id,
                pages=pages,
                qa_pairs=qas,
                source_path=source,
                source_dataset="mmlongbench",
                num_pages=len(pages) or None,
                meta={},
            )
        )

    return DatasetBundle(
        name="mmlongbench",
        documents=documents,
        split="subset",
        meta={"root": str(root), "subset": cfg, "n_raw_questions": len(items)},
    )


class MMLongBenchDataset(BaseDataset):
    name = "mmlongbench"

    def __init__(self, root: str | Path | None = None, **subset_kwargs: Any):
        self.root = root
        self.subset_kwargs = subset_kwargs

    def load(self) -> DatasetBundle:
        return load_mmlongbench_bundle(root=self.root, **self.subset_kwargs)


def load_mmlongbench(root: str | Path | None = None, limit: int | None = None):
    from pdf_vlm.data.base import bundle_to_examples

    bundle = load_mmlongbench_bundle(root=root, max_questions=limit)
    return bundle_to_examples(bundle)
