#!/usr/bin/env python
"""Benchmark Gemma 3 4B quantized local inference practicality.

Measures:
  - time-to-first-token (TTFT)
  - end-to-end response latency
  - tokens/sec
  - peak RSS / VRAM

Sweeps:
  - text-only vs image+text
  - top-k evidence under page buckets 5/20/50/100

Examples:
  # Mock (no GGUF) — plumbing / CI
  python scripts/bench_gemma_inference.py --mock

  # Real local run (requires downloaded GGUF)
  python scripts/download_models.py --with-mmproj
  python scripts/bench_gemma_inference.py --repeats 3

  # Faster subset
  python scripts/bench_gemma_inference.py --modalities text --top-ks 1,3 --page-buckets 5,20
"""

from __future__ import annotations

import argparse
from typing import Any

from pdf_vlm.bench.inference import BenchCase, default_cases, run_inference_benchmark
from pdf_vlm.utils.io import load_yaml, resolve_path
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def _ints(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _strs(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma local inference practicality bench")
    parser.add_argument("--config", default="configs/experiments/inference_bench.yaml")
    parser.add_argument("--mock", action="store_true", help="Synthetic timings (no GGUF)")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-ks", default=None, help="e.g. 1,3,5")
    parser.add_argument("--page-buckets", default=None, help="e.g. 5,20,50,100")
    parser.add_argument("--modalities", default=None, help="text,multimodal")
    parser.add_argument("--chars-per-page", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--text-only", action="store_true", help="Skip multimodal cases")
    args = parser.parse_args()

    cfg_path = resolve_path(args.config)
    cfg: dict[str, Any] = load_yaml(cfg_path) if cfg_path.exists() else {}

    max_tokens = args.max_tokens if args.max_tokens is not None else int(cfg.get("max_tokens", 64))
    repeats = args.repeats if args.repeats is not None else int(cfg.get("repeats", 2))
    warmup = args.warmup if args.warmup is not None else int(cfg.get("warmup", 1))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    chars = args.chars_per_page if args.chars_per_page is not None else int(cfg.get("chars_per_page", 800))

    top_ks = _ints(args.top_ks) or [int(x) for x in cfg.get("top_ks", [1, 3, 5])]
    buckets = _ints(args.page_buckets) or [int(x) for x in cfg.get("page_buckets", [5, 20, 50, 100])]
    modalities = _strs(args.modalities) or list(cfg.get("modalities") or ["text", "multimodal"])
    if args.text_only:
        modalities = ["text"]

    cases = default_cases(
        top_ks=top_ks,
        page_buckets=buckets,
        modalities=modalities,
        chars_per_page=chars,
        max_tokens=max_tokens,
    )

    # Auto-mock if weights missing and not explicitly real-only
    mock = args.mock
    if not mock:
        from pdf_vlm.utils.io import load_named_config

        model_rel = str(cfg.get("model") or "models/gemma3_4b_qat.yaml")
        if model_rel.startswith("configs/"):
            model_rel = model_rel[len("configs/") :]
        mcfg = load_named_config(model_rel)
        gguf = resolve_path(mcfg["local_path"])
        if not gguf.exists():
            logger.warning("GGUF missing at %s — running --mock. Download weights for real numbers.", gguf)
            mock = True

    out_dir = args.out_dir or cfg.get("output_dir")
    result = run_inference_benchmark(
        cases=cases,
        repeats=repeats,
        warmup=warmup,
        out_dir=out_dir,
        mock=mock,
        max_tokens=max_tokens,
        seed=seed,
    )
    prac = result["practicality"]
    logger.info("practicality_score=%s", prac.get("practicality_score"))
    for line in prac.get("highlights") or []:
        logger.info("• %s", line)
    print(result["out_dir"])


if __name__ == "__main__":
    main()
