"""Harness config loading and experiment cell expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from pdf_vlm.utils.io import load_yaml, merge_dicts, resolve_path


DEFAULT_INDEX_VARIANTS = {
    "page": "page_text",
    "hierarchical": "hier_text",
    "section": "section_text",
}


@dataclass(frozen=True)
class ExperimentCell:
    """One runnable cell in the evaluation matrix."""

    dataset: str
    pipeline_type: str
    retrieval_type: str
    top_k: int

    @property
    def cell_id(self) -> str:
        pipe = "text" if self.pipeline_type == "text" else "mm"
        ret = "page" if self.retrieval_type == "page" else (
            "hier" if self.retrieval_type in {"hierarchical", "hier"} else self.retrieval_type
        )
        return f"{self.dataset}__{pipe}_{ret}_k{self.top_k}"


@dataclass
class HarnessConfig:
    name: str = "eval_harness"
    description: str = ""
    seed: int = 42
    datasets: list[str] = field(default_factory=lambda: ["custom_5"])
    pipelines: list[str] = field(default_factory=lambda: ["text", "multimodal"])
    retrievals: list[str] = field(default_factory=lambda: ["page", "hierarchical"])
    top_k: list[int] = field(default_factory=lambda: [3])
    coarse_k: int = 5
    max_images: int | None = None
    index_variants: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_INDEX_VARIANTS))
    generation: dict[str, Any] = field(default_factory=lambda: {"max_tokens": 256, "temperature": 0.1})
    model: str = "models/gemma3_4b_qat.yaml"
    system_metrics: dict[str, bool] = field(
        default_factory=lambda: {"track_rss": True, "track_vram": True}
    )
    skip_missing_dataset: bool = True
    skip_missing_index: bool = True
    skip_missing_artifact: bool = True
    device: str = "cpu"
    dry_run: bool = False
    output_dir: str = "results/runs"
    report_dir: str = "results/reports"
    figures_dir: str = "results/figures"
    metrics: dict[str, Any] = field(default_factory=lambda: {"primary": "anls", "ks": [1, 3, 5]})
    limit: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def index_dirname(self, retrieval_type: str) -> str:
        key = "hierarchical" if retrieval_type in {"hierarchical", "hier"} else retrieval_type
        return self.index_variants.get(key, DEFAULT_INDEX_VARIANTS.get(key, f"{key}_text"))

    def resolve_index_dir(self, doc_id: str, retrieval_type: str) -> Path:
        return resolve_path(f"indices/{doc_id}/{self.index_dirname(retrieval_type)}")

    def iter_cells(self) -> Iterator[ExperimentCell]:
        for dataset in self.datasets:
            for pipeline in self.pipelines:
                for retrieval in self.retrievals:
                    for k in self.top_k:
                        yield ExperimentCell(
                            dataset=dataset,
                            pipeline_type=pipeline,
                            retrieval_type=retrieval,
                            top_k=int(k),
                        )

    def to_snapshot(self) -> dict[str, Any]:
        """Serializable config snapshot for reproducibility."""
        return {
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "datasets": list(self.datasets),
            "pipelines": list(self.pipelines),
            "retrievals": list(self.retrievals),
            "top_k": list(self.top_k),
            "coarse_k": self.coarse_k,
            "max_images": self.max_images,
            "index_variants": dict(self.index_variants),
            "generation": dict(self.generation),
            "model": self.model,
            "system_metrics": dict(self.system_metrics),
            "skip_missing_dataset": self.skip_missing_dataset,
            "skip_missing_index": self.skip_missing_index,
            "skip_missing_artifact": self.skip_missing_artifact,
            "device": self.device,
            "dry_run": self.dry_run,
            "output_dir": self.output_dir,
            "report_dir": self.report_dir,
            "figures_dir": self.figures_dir,
            "metrics": dict(self.metrics),
            "limit": self.limit,
        }


def load_harness_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> HarnessConfig:
    path = Path(path)
    if not path.is_absolute():
        path = resolve_path(path)
    raw = load_yaml(path)
    if overrides:
        raw = merge_dicts(raw, overrides)

    # Backward compat: variants: [{modality, retrieval}, ...]
    if "variants" in raw and "pipelines" not in raw:
        pipelines = sorted({v["modality"] for v in raw["variants"]})
        retrievals = sorted({v["retrieval"] for v in raw["variants"]})
        raw.setdefault("pipelines", pipelines)
        raw.setdefault("retrievals", retrievals)

    top_k = raw.get("top_k", [3])
    if isinstance(top_k, int):
        top_k = [top_k]

    return HarnessConfig(
        name=str(raw.get("name") or path.stem),
        description=str(raw.get("description") or ""),
        seed=int(raw.get("seed", 42)),
        datasets=list(raw.get("datasets") or ["custom_5"]),
        pipelines=list(raw.get("pipelines") or ["text", "multimodal"]),
        retrievals=list(raw.get("retrievals") or ["page", "hierarchical"]),
        top_k=[int(k) for k in top_k],
        coarse_k=int(raw.get("coarse_k", 5)),
        max_images=raw.get("max_images"),
        index_variants={**DEFAULT_INDEX_VARIANTS, **(raw.get("index_variants") or {})},
        generation=dict(raw.get("generation") or {"max_tokens": 256, "temperature": 0.1}),
        model=str(raw.get("model") or "models/gemma3_4b_qat.yaml"),
        system_metrics=dict(raw.get("system_metrics") or {"track_rss": True, "track_vram": True}),
        skip_missing_dataset=bool(raw.get("skip_missing_dataset", True)),
        skip_missing_index=bool(raw.get("skip_missing_index", True)),
        skip_missing_artifact=bool(raw.get("skip_missing_artifact", True)),
        device=str(raw.get("device") or "cpu"),
        dry_run=bool(raw.get("dry_run", False)),
        output_dir=str(raw.get("output_dir") or "results/runs"),
        report_dir=str(raw.get("report_dir") or "results/reports"),
        figures_dir=str(raw.get("figures_dir") or "results/figures"),
        metrics=dict(raw.get("metrics") or {"primary": "anls", "ks": [1, 3, 5]}),
        limit=raw.get("limit"),
        raw=raw,
    )
