#!/usr/bin/env python
"""Plot evaluation harness results from a run directory or predictions.json.

Examples:
  python scripts/plot_eval_results.py --run-dir results/runs/eval_...
  python scripts/plot_eval_results.py --predictions results/runs/eval_.../predictions.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pdf_vlm.eval.rows import EvalRow
from pdf_vlm.eval.viz import plot_all
from pdf_vlm.utils.io import load_json, resolve_path
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.predictions:
        pred_path = resolve_path(args.predictions)
        out_dir = resolve_path(args.out_dir) if args.out_dir else pred_path.parent / "figures"
    elif args.run_dir:
        run_dir = resolve_path(args.run_dir)
        pred_path = run_dir / "predictions.json"
        out_dir = resolve_path(args.out_dir) if args.out_dir else run_dir / "figures"
    else:
        raise SystemExit("Provide --run-dir or --predictions")

    raw = load_json(pred_path)
    rows = [EvalRow.model_validate(r) for r in raw]
    paths = plot_all(rows, out_dir)
    for k, v in paths.items():
        logger.info("%s -> %s", k, v)
    if not paths:
        raise SystemExit("No figures written (empty rows or missing matplotlib)")


if __name__ == "__main__":
    main()
