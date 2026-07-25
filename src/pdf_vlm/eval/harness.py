"""Evaluation harness runner: modality × retrieval × page-bucket matrix."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pdf_vlm.data import load_dataset
from pdf_vlm.eval.aggregate import aggregate_all
from pdf_vlm.eval.config import ExperimentCell, HarnessConfig, load_harness_config
from pdf_vlm.eval.report import export_reports
from pdf_vlm.eval.rows import EvalRow
from pdf_vlm.eval.scoring_rows import prediction_to_row
from pdf_vlm.eval.seed import set_global_seed
from pdf_vlm.eval.system_metrics import SystemMonitor, get_rss_mb
from pdf_vlm.ocr.paddle_structure import load_artifact
from pdf_vlm.rag.multimodal import MultimodalRAGPipeline
from pdf_vlm.rag.text_only import TextOnlyRAGPipeline, build_text_retriever
from pdf_vlm.schemas import QAExample
from pdf_vlm.utils.io import ensure_dir, load_named_config, resolve_path, save_json
from pdf_vlm.utils.logging import get_logger

logger = get_logger("eval.harness")


class EvaluationHarness:
    """Config-driven matrix runner with CSV/JSON/Markdown reports."""

    def __init__(self, config: HarnessConfig):
        self.config = config
        self.run_id = f"eval_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.skipped: list[str] = []
        self._text_llm = None
        self._mm_llm = None
        self._pipe_cache: dict[tuple, Any] = {}
        self.monitor = SystemMonitor(
            track_rss=bool(config.system_metrics.get("track_rss", True)),
            track_vram=bool(config.system_metrics.get("track_vram", True)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "EvaluationHarness":
        cfg = load_harness_config(path, overrides=overrides or None)
        return cls(cfg)

    def _load_model_cfg(self) -> dict[str, Any]:
        model_rel = str(self.config.model)
        if model_rel.startswith("configs/"):
            model_rel = model_rel[len("configs/") :]
        return load_named_config(model_rel)

    def _release_llms(self) -> None:
        """Free GPU/CPU llama contexts before loading the other modality."""
        import gc

        if self._text_llm is not None:
            self._text_llm.unload()
            self._text_llm = None
        if self._mm_llm is not None:
            self._mm_llm.unload()
            self._mm_llm = None
        self._pipe_cache.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ensure_llm_for(self, pipeline_type: str) -> None:
        """Load at most one Gemma instance (text XOR multimodal) — Colab VRAM safe."""
        if self.config.dry_run:
            return
        from pdf_vlm.llm.gemma_llama_cpp import build_llm

        cfg = self._load_model_cfg()
        want_mm = pipeline_type in {"multimodal", "mm"}

        if want_mm:
            if self._text_llm is not None:
                logger.info("Switching LLM: unload text before multimodal")
                self._text_llm.unload()
                self._text_llm = None
                self._pipe_cache.clear()
            if self._mm_llm is None:
                mm_cfg = dict(cfg)
                mm_cfg["local_path"] = str(resolve_path(mm_cfg["local_path"]))
                if mm_cfg.get("mmproj_local_path"):
                    mm_cfg["mmproj_local_path"] = str(resolve_path(mm_cfg["mmproj_local_path"]))
                mm_cfg["enable_vision"] = True
                # Smaller context for vision on limited VRAM
                mm_cfg["n_ctx"] = min(int(mm_cfg.get("n_ctx") or 8192), 4096)
                mm_cfg["n_batch"] = min(int(mm_cfg.get("n_batch") or 512), 256)
                self._mm_llm = build_llm(mm_cfg)
        else:
            if self._mm_llm is not None:
                logger.info("Switching LLM: unload multimodal before text")
                self._mm_llm.unload()
                self._mm_llm = None
                self._pipe_cache.clear()
            if self._text_llm is None:
                text_cfg = dict(cfg)
                text_cfg["local_path"] = str(resolve_path(text_cfg["local_path"]))
                text_cfg["enable_vision"] = False
                text_cfg["mmproj_local_path"] = None
                text_cfg["n_ctx"] = min(int(text_cfg.get("n_ctx") or 8192), 4096)
                self._text_llm = build_llm(text_cfg)

    def _ensure_llms(self) -> None:
        # Backward-compatible entry: prefer lazy load in run(); keep no-op warm path.
        if self.config.dry_run:
            return
        # Do not preload both — that OOMs Colab (Failed to create llama_context).
        logger.info("LLM load deferred until first text/multimodal cell (VRAM-safe)")

    def _get_pipeline(self, cell: ExperimentCell, doc_id: str):
        self._ensure_llm_for(cell.pipeline_type)
        key = (cell.pipeline_type, cell.retrieval_type, doc_id, cell.top_k)
        if key in self._pipe_cache:
            return self._pipe_cache[key]

        index_dir = self.config.resolve_index_dir(doc_id, cell.retrieval_type)
        if not index_dir.exists():
            return None

        retriever = build_text_retriever(
            index_dir,
            retrieval=cell.retrieval_type,
            device=self.config.device,
            coarse_k=self.config.coarse_k,
        )
        gen = self.config.generation

        if cell.pipeline_type == "text":
            pipe = TextOnlyRAGPipeline(
                self._text_llm,
                retriever,
                retrieval_mode=cell.retrieval_type,  # type: ignore[arg-type]
                top_k=cell.top_k,
                generation_cfg=gen,
                dry_run=self.config.dry_run,
            )
        elif cell.pipeline_type in {"multimodal", "mm"}:
            try:
                artifact = load_artifact(doc_id)
            except Exception as e:
                if self.config.skip_missing_artifact:
                    logger.warning("Missing artifact for %s: %s", doc_id, e)
                    return None
                raise
            pipe = MultimodalRAGPipeline(
                self._mm_llm,
                retriever,
                artifact,
                retrieval_mode=cell.retrieval_type,  # type: ignore[arg-type]
                top_k=cell.top_k,
                max_images=self.config.max_images,
                generation_cfg=gen,
                dry_run=self.config.dry_run,
            )
        else:
            raise ValueError(f"Unknown pipeline_type={cell.pipeline_type}")

        self._pipe_cache[key] = pipe
        return pipe

    def _load_examples(self, dataset: str) -> list[QAExample] | None:
        try:
            examples = load_dataset(dataset, limit=self.config.limit)
        except Exception as e:
            if self.config.skip_missing_dataset:
                self.skipped.append(f"dataset={dataset}: load error ({e})")
                logger.warning("Skip dataset %s: %s", dataset, e)
                return None
            raise
        if not examples:
            if self.config.skip_missing_dataset:
                self.skipped.append(f"dataset={dataset}: empty")
                logger.warning("Skip empty dataset %s", dataset)
                return None
            raise RuntimeError(f"Empty dataset: {dataset}")
        return examples

    def run_cell(self, cell: ExperimentCell, examples: list[QAExample]) -> list[EvalRow]:
        rows: list[EvalRow] = []
        primary = str(self.config.metrics.get("primary") or "anls")

        # Group by doc_id so we reuse pipelines
        by_doc: dict[str, list[QAExample]] = {}
        for ex in examples:
            by_doc.setdefault(ex.doc_id, []).append(ex)

        for doc_id, doc_examples in by_doc.items():
            index_dir = self.config.resolve_index_dir(doc_id, cell.retrieval_type)
            if not index_dir.exists():
                msg = f"{cell.cell_id} doc={doc_id}: missing index {index_dir}"
                if self.config.skip_missing_index:
                    self.skipped.append(msg)
                    logger.warning("Skip %s", msg)
                    continue
                raise FileNotFoundError(msg)

            pipe = self._get_pipeline(cell, doc_id)
            if pipe is None:
                msg = f"{cell.cell_id} doc={doc_id}: could not build pipeline"
                self.skipped.append(msg)
                logger.warning("Skip %s", msg)
                continue

            for ex in doc_examples:
                self.monitor.sample()
                pred = pipe.answer(ex)
                rss = pred.peak_rss_mb
                vram = pred.peak_vram_mb
                if rss is None and self.monitor.track_rss:
                    rss = get_rss_mb()
                row = prediction_to_row(
                    pred,
                    run_id=self.run_id,
                    dataset=cell.dataset,
                    pipeline_type=(
                        "multimodal" if cell.pipeline_type in {"mm", "multimodal"} else "text"
                    ),
                    retrieval_type=(
                        "hierarchical"
                        if cell.retrieval_type in {"hierarchical", "hier"}
                        else cell.retrieval_type
                    ),
                    top_k=cell.top_k,
                    example=ex,
                    primary_metric=primary,
                    monitor_rss=rss,
                    monitor_vram=vram if vram is not None else self.monitor.peak_vram_mb,
                )
                rows.append(row)

        logger.info(
            "cell %s -> %d rows (correctness_mean=%.4f)",
            cell.cell_id,
            len(rows),
            sum(r.correctness for r in rows) / len(rows) if rows else 0.0,
        )
        return rows

    def run(self) -> dict[str, Any]:
        seed_snapshot = set_global_seed(self.config.seed)
        run_dir = ensure_dir(
            resolve_path(self.config.output_dir) / self.run_id
        )
        report_dir = ensure_dir(resolve_path(self.config.report_dir))
        figures_dir = ensure_dir(resolve_path(self.config.figures_dir))

        config_snapshot = self.config.to_snapshot()
        config_snapshot["run_id"] = self.run_id
        save_json(run_dir / "config_snapshot.json", config_snapshot)
        save_json(run_dir / "seed_snapshot.json", seed_snapshot)

        self.monitor.start()
        try:
            self._ensure_llms()
        except Exception as e:
            if self.config.dry_run:
                logger.info("dry_run: LLM not loaded (%s)", e)
            else:
                logger.warning("LLM load failed (%s); forcing dry_run", e)
                self.config.dry_run = True
                config_snapshot["dry_run"] = True
                config_snapshot["llm_load_error"] = str(e)

        all_rows: list[EvalRow] = []
        cell_summaries: list[dict[str, Any]] = []

        # Cache examples per dataset
        examples_cache: dict[str, list[QAExample] | None] = {}

        # Run all text cells before multimodal so we only hold one llama context.
        cells = list(self.config.iter_cells())
        cells.sort(key=lambda c: 0 if c.pipeline_type == "text" else 1)

        for cell in cells:
            if cell.dataset not in examples_cache:
                examples_cache[cell.dataset] = self._load_examples(cell.dataset)
            examples = examples_cache[cell.dataset]
            if not examples:
                continue

            try:
                rows = self.run_cell(cell, examples)
            except Exception as e:
                msg = f"{cell.cell_id}: {e}"
                self.skipped.append(msg)
                logger.warning("Skip cell after error: %s", msg)
                # Free context after hard llama failures so later cells can proceed
                self._release_llms()
                continue
            if not rows:
                continue
            all_rows.extend(rows)

            cell_dir = ensure_dir(run_dir / "cells" / cell.cell_id)
            save_json(cell_dir / "predictions.json", [r.model_dump(mode="json") for r in rows])
            from pdf_vlm.eval.scoring_rows import summarize_rows

            cell_sum = summarize_rows(rows, label=cell.cell_id)
            cell_sum.update(
                {
                    "dataset": cell.dataset,
                    "pipeline_type": cell.pipeline_type,
                    "retrieval_type": cell.retrieval_type,
                    "top_k": cell.top_k,
                }
            )
            save_json(cell_dir / "metrics.json", cell_sum)
            cell_summaries.append(cell_sum)

        self.monitor.sample()
        peak = self.monitor.stop()

        aggregates = aggregate_all(all_rows)
        paths = export_reports(
            run_dir,
            run_id=self.run_id,
            rows=all_rows,
            aggregates=aggregates,
            config_snapshot=config_snapshot,
            seed_snapshot=seed_snapshot,
            skipped=self.skipped,
            report_dir=report_dir,
        )

        # Optional figures
        fig_paths: dict[str, str] = {}
        try:
            from pdf_vlm.eval.viz import plot_all

            fig_paths = plot_all(all_rows, figures_dir / self.run_id)
            paths.update({f"fig_{k}": v for k, v in fig_paths.items()})
        except Exception as e:
            logger.warning("Visualization skipped: %s", e)

        result = {
            "run_id": self.run_id,
            "run_dir": str(run_dir),
            "n_rows": len(all_rows),
            "n_cells": len(cell_summaries),
            "skipped": self.skipped,
            "peak_rss_mb": peak.get("peak_rss_mb"),
            "peak_vram_mb": peak.get("peak_vram_mb"),
            "aggregates": aggregates,
            "paths": paths,
            "cell_summaries": cell_summaries,
        }
        save_json(run_dir / "harness_result.json", {
            **{k: v for k, v in result.items() if k != "aggregates"},
            "overall": (aggregates.get("overall") or [{}])[0],
        })
        logger.info("Harness complete: %s (%d rows) -> %s", self.run_id, len(all_rows), run_dir)
        return result


def run_harness(config_path: str | Path, **overrides: Any) -> dict[str, Any]:
    return EvaluationHarness.from_yaml(config_path, **overrides).run()
