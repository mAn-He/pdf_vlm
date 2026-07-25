#!/usr/bin/env python
"""Smoke test for dataset loaders + stats (uses fixtures by default)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pdf_vlm.data import (
    compute_dataset_stats,
    get_dataset,
    load_bundle,
    print_dataset_stats,
)
from pdf_vlm.utils.io import project_root, save_json
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset layer smoke test")
    parser.add_argument(
        "--fixtures",
        action="store_true",
        default=True,
        help="Load tiny fixtures under data/fixtures (default)",
    )
    parser.add_argument("--no-fixtures", action="store_true", help="Use real data/raw paths")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    use_fixtures = not args.no_fixtures

    root = project_root()
    fixture_root = root / "data" / "fixtures"

    specs = []
    if use_fixtures:
        specs = [
            ("mp_docvqa", {"root": fixture_root / "mp_docvqa", "split": "val"}),
            (
                "mmlongbench",
                {
                    "root": fixture_root / "mmlongbench",
                    "max_docs": 5,
                    "max_questions": 20,
                    "seed": 0,
                },
            ),
            ("custom_5", {"root": fixture_root / "custom" / "5"}),
        ]
    else:
        specs = [
            ("mp_docvqa", {"split": "val", "limit_questions": 50}),
            ("mmlongbench", {"max_docs": 10, "max_questions": 50, "seed": 42}),
            ("custom_5", {}),
        ]

    all_stats = {}
    for name, kwargs in specs:
        logger.info("Loading %s kwargs=%s", name, kwargs)
        ds = get_dataset(name, **kwargs)
        bundle = ds.load()
        # verify common interface fields
        for doc in bundle.documents:
            assert doc.doc_id
            assert isinstance(doc.pages, list)
            assert isinstance(doc.qa_pairs, list)
            for qa in doc.qa_pairs:
                assert qa.question is not None
                assert qa.answer is not None
                assert qa.question_type is not None

        examples = ds.to_examples()
        stats = print_dataset_stats(bundle)
        stats["n_flat_examples"] = len(examples)
        all_stats[name] = stats
        logger.info(
            "%s -> docs=%s questions=%s avg_pages=%s types=%s",
            name,
            stats["n_docs"],
            stats["n_questions"],
            stats["avg_pages"],
            stats["question_type_distribution"],
        )

    out = args.out or (root / "artifacts" / "smoke" / "dataset_stats.json")
    save_json(out, all_stats)
    logger.info("Wrote %s", out)

    # Fail if fixtures are empty (indicates broken packaging)
    if use_fixtures:
        for name, st in all_stats.items():
            if st["n_questions"] == 0:
                raise SystemExit(f"Smoke failed: {name} has 0 questions")
    logger.info("Dataset smoke OK")


if __name__ == "__main__":
    main()
