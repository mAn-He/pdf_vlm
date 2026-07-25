#!/usr/bin/env python
"""Remap data/custom/*/questions.json doc_ids to match existing indices/{doc_id}/.

Use when questions were built on another machine (path-based ids) but OCR/index
ran on Colab (different absolute-path hashes). Matching is by PDF stem prefix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdf_vlm.utils.io import load_json, resolve_path, save_json
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def _index_doc_ids() -> dict[str, str]:
    """Map stem -> full doc_id from indices/*/page_text."""
    root = resolve_path("indices")
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "page_text").exists():
            continue
        # stem = everything before last _xxxxxxxx (10 hex) if present
        name = d.name
        stem = name.rsplit("_", 1)[0] if "_" in name else name
        out[stem] = name
        out[name] = name
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buckets", default="5,10,20,50,100")
    args = parser.parse_args()

    id_map = _index_doc_ids()
    if not id_map:
        raise SystemExit("No indices found under indices/*/page_text")

    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]
    report: list[dict] = []

    for n in buckets:
        qpath = resolve_path(f"data/custom/{n}/questions.json")
        mpath = resolve_path(f"data/custom/{n}/manifest.json")
        if not qpath.exists():
            logger.warning("missing %s", qpath)
            continue

        rows = load_json(qpath)
        if not isinstance(rows, list):
            raise SystemExit(f"Expected list in {qpath}")

        changed = 0
        new_rows = []
        for row in rows:
            old = str(row.get("doc_id") or "")
            stem = old.rsplit("_", 1)[0] if "_" in old else old
            # also try hyundai stem from pdf in bucket
            new_id = id_map.get(old) or id_map.get(stem)
            if not new_id:
                # fallback: any index stem starting with hyundai..._{n}p
                key = f"hyundai_wia_qa_report_{n}p"
                new_id = id_map.get(key)
            if new_id and new_id != old:
                row = dict(row)
                row["doc_id"] = new_id
                changed += 1
            new_rows.append(row)

        save_json(qpath, new_rows)

        if mpath.exists():
            man = load_json(mpath)
            docs = man.get("documents") or []
            for d in docs:
                old = str(d.get("doc_id") or "")
                stem = old.rsplit("_", 1)[0] if "_" in old else old
                key = f"hyundai_wia_qa_report_{n}p"
                new_id = id_map.get(old) or id_map.get(stem) or id_map.get(key)
                if new_id:
                    d["doc_id"] = new_id
            save_json(mpath, man)

        report.append({"bucket": n, "questions_remapped": changed, "n": len(new_rows)})
        logger.info("bucket=%d remapped=%d / %d", n, changed, len(new_rows))

    print(json.dumps({"index_ids": id_map, "buckets": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
