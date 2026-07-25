#!/usr/bin/env python
"""Backward-compatible entrypoint → evaluation harness.

Prefer: python scripts/run_eval_harness.py ...
"""

from __future__ import annotations

import argparse
from typing import Any

from pdf_vlm.eval.harness import EvaluationHarness
from pdf_vlm.utils.io import resolve_path
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment matrix (harness)")
    parser.add_argument("--matrix", default="configs/experiments/matrix_2x2.yaml")
    parser.add_argument("--doc-id", default=None, help="Unused (doc_id comes from dataset); kept for compat")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset override")
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.dry_run:
        overrides["dry_run"] = True
    if args.limit is not None:
        overrides["limit"] = args.limit
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.datasets:
        overrides["datasets"] = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if args.doc_id:
        logger.warning("--doc-id ignored; harness uses example.doc_id from the dataset")

    result = EvaluationHarness.from_yaml(resolve_path(args.matrix), **overrides).run()
    logger.info("matrix complete -> %s", result["run_dir"])
    print(result["run_dir"])


if __name__ == "__main__":
    main()
