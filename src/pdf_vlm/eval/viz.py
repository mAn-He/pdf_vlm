"""Visualization helpers for evaluation harness results.

Plots (saved as PNG):
  1. length (page bucket) vs accuracy
  2. pipeline latency comparison
  3. question-type performance bars

Requires optional dependency: matplotlib
  pip install matplotlib
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from pdf_vlm.eval.rows import EvalRow
from pdf_vlm.eval.scoring_rows import mean
from pdf_vlm.utils.io import ensure_dir
from pdf_vlm.utils.logging import get_logger

logger = get_logger("eval.viz")


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plots. Install with: pip install matplotlib"
        ) from e


def _group_mean(rows: Sequence[EvalRow], key_fn, value_fn) -> list[tuple[str, float, int]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        buckets[str(key_fn(r))].append(float(value_fn(r)))
    out = []
    for k in sorted(buckets.keys(), key=lambda x: (x == "None", x.zfill(8) if x.isdigit() else x)):
        vals = buckets[k]
        out.append((k, mean(vals), len(vals)))
    return out


def plot_length_vs_accuracy(rows: Sequence[EvalRow], out_path: Path) -> Path:
    plt = _require_matplotlib()
    ensure_dir(out_path.parent)

    # Split by pipeline for dual series
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for pipe, marker, color in (
        ("text", "o", "#1f4e79"),
        ("multimodal", "s", "#c45c26"),
    ):
        subset = [r for r in rows if r.pipeline_type == pipe]
        if not subset:
            continue
        series = _group_mean(
            subset,
            key_fn=lambda r: r.page_bucket if r.page_bucket is not None else "?",
            value_fn=lambda r: r.correctness,
        )
        xs = [s[0] for s in series]
        ys = [s[1] for s in series]
        ax.plot(xs, ys, marker=marker, color=color, linewidth=2, label=pipe)

    ax.set_xlabel("Page-length bucket")
    ax.set_ylabel("Mean correctness (ANLS)")
    ax.set_title("Document length vs accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pipeline_latency(rows: Sequence[EvalRow], out_path: Path) -> Path:
    plt = _require_matplotlib()
    ensure_dir(out_path.parent)

    series = _group_mean(rows, key_fn=lambda r: r.pipeline_type, value_fn=lambda r: r.latency_ms)
    labels = [s[0] for s in series]
    vals = [s[1] for s in series]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = ["#1f4e79" if l == "text" else "#c45c26" for l in labels]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    ax.set_ylabel("Mean end-to-end latency (ms)")
    ax.set_title("Latency by pipeline")
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_question_type_bars(rows: Sequence[EvalRow], out_path: Path) -> Path:
    plt = _require_matplotlib()
    ensure_dir(out_path.parent)

    # Grouped bars: question_type × pipeline
    qtypes = sorted({r.question_type or "unknown" for r in rows})
    pipelines = sorted({r.pipeline_type for r in rows})
    if not qtypes or not pipelines:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    import numpy as np

    x = np.arange(len(qtypes))
    width = 0.35 if len(pipelines) <= 2 else 0.8 / max(len(pipelines), 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = {"text": "#1f4e79", "multimodal": "#c45c26"}

    for i, pipe in enumerate(pipelines):
        means = []
        for qt in qtypes:
            vals = [r.correctness for r in rows if r.question_type == qt and r.pipeline_type == pipe]
            means.append(mean(vals) if vals else 0.0)
        offset = (i - (len(pipelines) - 1) / 2) * width
        ax.bar(x + offset, means, width=width, label=pipe, color=palette.get(pipe, f"C{i}"))

    ax.set_xticks(x)
    ax.set_xticklabels(qtypes)
    ax.set_ylabel("Mean correctness (ANLS)")
    ax.set_xlabel("Question type")
    ax.set_title("Performance by question type")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_retrieval_comparison(rows: Sequence[EvalRow], out_path: Path) -> Path:
    """Bonus: page vs hierarchical accuracy bars."""
    plt = _require_matplotlib()
    ensure_dir(out_path.parent)
    series = _group_mean(rows, key_fn=lambda r: r.retrieval_type, value_fn=lambda r: r.correctness)
    labels = [s[0] for s in series]
    vals = [s[1] for s in series]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(labels, vals, color=["#2a6f4e", "#6b4c9a"][: len(labels)], width=0.55)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean correctness (ANLS)")
    ax.set_title("Accuracy by retrieval type")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_all(rows: Sequence[EvalRow], out_dir: Path | str) -> dict[str, str]:
    out_dir = ensure_dir(Path(out_dir))
    paths: dict[str, str] = {}
    if not rows:
        logger.warning("No rows to plot")
        return paths

    mapping = {
        "length_vs_accuracy": plot_length_vs_accuracy,
        "pipeline_latency": plot_pipeline_latency,
        "question_type": plot_question_type_bars,
        "retrieval_accuracy": plot_retrieval_comparison,
    }
    for name, fn in mapping.items():
        path = out_dir / f"{name}.png"
        try:
            fn(rows, path)
            paths[name] = str(path)
            logger.info("Wrote figure %s", path)
        except Exception as e:
            logger.warning("Failed plot %s: %s", name, e)
    return paths
