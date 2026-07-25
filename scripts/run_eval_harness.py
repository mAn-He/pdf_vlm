#!/usr/bin/env python
"""Run the evaluation harness (modality × retrieval × page-bucket matrix).

Examples:
  # Full config, dry-run (no Gemma weights required)
  python scripts/run_eval_harness.py --config configs/experiments/eval_harness.yaml --dry-run

  # Slim 2×2 on custom_5 only
  python scripts/run_eval_harness.py --config configs/experiments/matrix_2x2.yaml \\
      --dry-run --datasets custom_5 --top-k 3

  # Real generation (requires GGUF + mmproj)
  python scripts/run_eval_harness.py --config configs/experiments/eval_harness.yaml
"""

from __future__ import annotations

import argparse
from typing import Any

from pdf_vlm.eval.harness import EvaluationHarness
from pdf_vlm.utils.io import resolve_path
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def _parse_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation harness runner")
    parser.add_argument(
        "--config",
        default="configs/experiments/eval_harness.yaml",
        help="Experiment YAML (pipelines/retrievals/datasets selectable)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Retrieve only; skip Gemma generation")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max questions per dataset")
    parser.add_argument("--device", default=None)
    parser.add_argument("--datasets", default=None, help="Comma-separated override, e.g. custom_5,custom_20")
    parser.add_argument("--pipelines", default=None, help="Comma-separated: text,multimodal")
    parser.add_argument("--retrievals", default=None, help="Comma-separated: page,hierarchical")
    parser.add_argument("--top-k", default=None, help="Comma-separated ints, e.g. 1,3,5")
    parser.add_argument("--no-skip-missing", action="store_true", help="Fail hard on missing data/index")
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.dry_run:
        overrides["dry_run"] = True
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.limit is not None:
        overrides["limit"] = args.limit
    if args.device:
        overrides["device"] = args.device
    ds = _parse_list(args.datasets)
    if ds:
        overrides["datasets"] = ds
    pipes = _parse_list(args.pipelines)
    if pipes:
        overrides["pipelines"] = pipes
    rets = _parse_list(args.retrievals)
    if rets:
        overrides["retrievals"] = rets
    if args.top_k:
        overrides["top_k"] = [int(x) for x in args.top_k.split(",") if x.strip()]
    if args.no_skip_missing:
        overrides["skip_missing_dataset"] = False
        overrides["skip_missing_index"] = False
        overrides["skip_missing_artifact"] = False

    harness = EvaluationHarness.from_yaml(resolve_path(args.config), **overrides)
    result = harness.run()

    overall = (result.get("aggregates") or {}).get("overall") or [{}]
    o = overall[0] if overall else {}
    logger.info(
        "done run_id=%s n_rows=%s correctness=%.4f latency_ms=%.1f report=%s",
        result["run_id"],
        result["n_rows"],
        o.get("correctness_mean") or 0.0,
        o.get("latency_ms_mean") or 0.0,
        (result.get("paths") or {}).get("report_md"),
    )
    print(result["run_dir"])


if __name__ == "__main__":
    main()
