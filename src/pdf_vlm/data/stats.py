"""Dataset statistics for DatasetBundle / BaseDataset."""

from __future__ import annotations

from collections import Counter
from typing import Any

from pdf_vlm.data.base import BaseDataset
from pdf_vlm.schemas import DatasetBundle


def compute_dataset_stats(bundle: DatasetBundle) -> dict[str, Any]:
    n_docs = bundle.n_docs
    n_questions = bundle.n_questions
    page_counts = [d.page_count for d in bundle.documents]
    avg_pages = (sum(page_counts) / len(page_counts)) if page_counts else 0.0

    type_counter: Counter[str] = Counter()
    modality_counter: Counter[str] = Counter()
    una = 0
    cross = 0
    for doc in bundle.documents:
        for qa in doc.qa_pairs:
            type_counter[qa.question_type.value] += 1
            if qa.unanswerable:
                una += 1
            if len(set(qa.evidence_pages)) >= 2:
                cross += 1
            for m in qa.evidence_modality:
                modality_counter[m] += 1

    return {
        "name": bundle.name,
        "split": bundle.split,
        "n_docs": n_docs,
        "n_questions": n_questions,
        "avg_pages": round(avg_pages, 2),
        "min_pages": min(page_counts) if page_counts else 0,
        "max_pages": max(page_counts) if page_counts else 0,
        "question_type_distribution": dict(sorted(type_counter.items())),
        "evidence_modality_distribution": dict(sorted(modality_counter.items())),
        "n_unanswerable": una,
        "n_cross_page_evidence": cross,
        "meta": bundle.meta,
    }


def print_dataset_stats(bundle: DatasetBundle | BaseDataset) -> dict[str, Any]:
    if isinstance(bundle, BaseDataset):
        bundle = bundle.load()
    stats = compute_dataset_stats(bundle)
    print(f"=== Dataset stats: {stats['name']} ===")
    print(f"split              : {stats['split']}")
    print(f"documents          : {stats['n_docs']}")
    print(f"questions          : {stats['n_questions']}")
    print(f"avg / min / max pages : {stats['avg_pages']} / {stats['min_pages']} / {stats['max_pages']}")
    print("question_type_distribution:")
    for k, v in stats["question_type_distribution"].items():
        print(f"  - {k}: {v}")
    if stats["evidence_modality_distribution"]:
        print("evidence_modality_distribution:")
        for k, v in stats["evidence_modality_distribution"].items():
            print(f"  - {k}: {v}")
    print(f"unanswerable       : {stats['n_unanswerable']}")
    print(f"cross-page evidence: {stats['n_cross_page_evidence']}")
    return stats
