"""Dataset registry: one entrypoint for eval pipelines."""

from __future__ import annotations

from typing import Any

from pdf_vlm.data.base import BaseDataset, UnifiedDataset, bundle_to_examples
from pdf_vlm.data.custom_pdfs import CustomPDFDataset, load_custom_bundle, load_custom_qa
from pdf_vlm.data.mmlongbench import MMLongBenchDataset, load_mmlongbench, load_mmlongbench_bundle
from pdf_vlm.data.mp_docvqa import MPDocVQADataset, load_mp_docvqa, load_mp_docvqa_bundle
from pdf_vlm.data.stats import compute_dataset_stats, print_dataset_stats
from pdf_vlm.schemas import DatasetBundle, QAExample


def get_dataset(name: str, **kwargs: Any) -> BaseDataset:
    """Factory for dataset adapters used by the evaluation pipeline."""
    key = name.lower().strip()
    if key.startswith("custom"):
        return CustomPDFDataset(dataset=name, **kwargs)
    if key.startswith("mp_docvqa") or key in {"mpdocvqa", "mp-docvqa"}:
        return MPDocVQADataset(**kwargs)
    if key.startswith("mmlong") or key in {"mmlongbench", "mmlongbench-doc"}:
        return MMLongBenchDataset(**kwargs)
    raise ValueError(f"Unknown dataset: {name}")


def load_bundle(name: str, **kwargs: Any) -> DatasetBundle:
    return get_dataset(name, **kwargs).load()


def load_dataset(name: str, limit: int | None = None, **kwargs: Any) -> list[QAExample]:
    """Flat QAExample list (backward compatible with RAG scripts)."""
    key = name.lower().strip()
    if key.startswith("custom"):
        examples = load_custom_qa(name)
    elif key.startswith("mp_docvqa") or key in {"mpdocvqa", "mp-docvqa"}:
        examples = load_mp_docvqa(limit=limit, **kwargs)
    elif key.startswith("mmlong") or key in {"mmlongbench", "mmlongbench-doc"}:
        examples = load_mmlongbench(limit=limit, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    if limit is not None:
        examples = examples[:limit]
    return examples


__all__ = [
    "BaseDataset",
    "UnifiedDataset",
    "CustomPDFDataset",
    "MPDocVQADataset",
    "MMLongBenchDataset",
    "get_dataset",
    "load_bundle",
    "load_dataset",
    "load_custom_bundle",
    "load_mp_docvqa_bundle",
    "load_mmlongbench_bundle",
    "bundle_to_examples",
    "compute_dataset_stats",
    "print_dataset_stats",
    "QAExample",
    "DatasetBundle",
]
