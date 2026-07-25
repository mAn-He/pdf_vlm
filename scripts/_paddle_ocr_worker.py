#!/usr/bin/env python
"""Single-image PaddleOCR worker (subprocess isolation against native crashes)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lang", default="korean")
    parser.add_argument("--backend", choices=["structurev3", "ocr"], default="ocr")
    args = parser.parse_args()

    try:
        import paddle

        try:
            paddle.set_flags({"FLAGS_use_mkldnn": False})
        except Exception:
            pass
    except Exception:
        pass

    image = str(Path(args.image).resolve())
    out_path = Path(args.out)
    payload: dict

    if args.backend == "structurev3":
        from paddleocr import PPStructureV3

        engine = PPStructureV3(
            lang=args.lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_seal_recognition=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_table_recognition=False,
            use_region_detection=False,
            layout_detection_model_name="PP-DocLayout-S",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
            text_det_limit_side_len=960,
            device="cpu",
        )
        raw = engine.predict(input=image)
        item = raw[0] if isinstance(raw, list) and raw else raw
        if hasattr(item, "json") and isinstance(item.json, dict):
            payload = dict(item.json)
        elif isinstance(item, dict):
            payload = item
        else:
            payload = {"raw": str(item)}
        payload["backend"] = "PPStructureV3"
    else:
        from paddleocr import PaddleOCR

        engine = PaddleOCR(
            lang=args.lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
        )
        raw = engine.predict(input=image)
        item = raw[0] if isinstance(raw, list) and raw else raw
        if hasattr(item, "json") and isinstance(item.json, dict):
            payload = dict(item.json)
        elif isinstance(item, dict):
            payload = item
        else:
            # extract rec_texts if possible
            payload = {"raw": str(item)}
            for attr in ("rec_texts", "json"):
                if hasattr(item, attr):
                    val = getattr(item, attr)
                    if attr == "rec_texts" and isinstance(val, list):
                        payload = {"ocr_texts": [str(x) for x in val], "markdown": "\n".join(str(x) for x in val)}
                    elif attr == "json" and isinstance(val, dict):
                        payload = dict(val)
        payload["backend"] = "PaddleOCR"

    payload["source_image"] = image
    if not payload.get("markdown") and payload.get("ocr_texts"):
        payload["markdown"] = "\n".join(str(x) for x in payload["ocr_texts"])
    if not payload.get("ocr_texts") and isinstance(payload.get("rec_texts"), list):
        payload["ocr_texts"] = [str(x) for x in payload["rec_texts"]]
        payload.setdefault("markdown", "\n".join(payload["ocr_texts"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
