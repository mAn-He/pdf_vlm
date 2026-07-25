#!/usr/bin/env python
"""Run Gemma 3 4B local inference smoke tests: text-only and image+text.

Writes:
  artifacts/smoke/gemma_text_ok.json
  artifacts/smoke/gemma_vision_ok.json
  artifacts/smoke/gemma_ok.json  (summary)

Exit code 0 only if both tests succeed (unless --text-only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from pdf_vlm.llm.gemma_llama_cpp import Gemma3LlamaCpp
from pdf_vlm.utils.io import ensure_dir, project_root, save_json
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def _make_sample_image(path: Path) -> Path:
    ensure_dir(path.parent)
    img = Image.new("RGB", (512, 384), color=(30, 90, 180))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 220, 180], fill=(240, 200, 40))
    draw.ellipse([280, 80, 460, 260], fill=(220, 60, 60))
    draw.text((50, 300), "GEMMA3-VISION-TEST", fill=(255, 255, 255))
    img.save(path)
    return path


def _result_dict(tag: str, result) -> dict:
    return {
        "ok": bool(result.text),
        "tag": tag,
        "text": result.text,
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "peak_rss_mb": result.peak_rss_mb,
        "peak_vram_mb": result.peak_vram_mb,
        "meta": result.meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-only", action="store_true", help="Skip vision test")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image path for vision test (default: generated sample)",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(project_root() / "artifacts" / "smoke")
    llm = Gemma3LlamaCpp.from_config()

    # --- Test 1: text-only ---
    text_res = llm.generate_text(
        "Reply with exactly one word: OK",
        max_tokens=args.max_tokens,
        temperature=0.0,
    )
    text_payload = _result_dict("text", text_res)
    save_json(out_dir / "gemma_text_ok.json", text_payload)
    logger.info("TEXT latency_ms=%.1f text=%r", text_res.latency_ms, text_res.text[:200])
    if not text_res.text:
        raise SystemExit("TEXT TEST FAILED: empty response")

    summary = {"text": text_payload, "vision": None, "vision_supported": llm.vision_supported}

    if args.text_only:
        save_json(out_dir / "gemma_ok.json", summary)
        logger.info("Skipped vision (--text-only). Text test passed.")
        return

    # --- Test 2: image + text ---
    image_path = args.image or (out_dir / "sample_vision.png")
    if args.image is None:
        _make_sample_image(image_path)

    vision_res = llm.generate_vision(
        "What shapes and colors do you see? Answer briefly.",
        image_path,
        max_tokens=args.max_tokens,
        temperature=0.1,
    )
    vision_payload = _result_dict("vision", vision_res)
    vision_payload["image_path"] = str(image_path)
    save_json(out_dir / "gemma_vision_ok.json", vision_payload)
    logger.info("VISION latency_ms=%.1f text=%r", vision_res.latency_ms, vision_res.text[:300])
    if not vision_res.text:
        raise SystemExit("VISION TEST FAILED: empty response")

    summary["vision"] = vision_payload
    save_json(out_dir / "gemma_ok.json", summary)
    logger.info("Both text and vision smoke tests passed.")


if __name__ == "__main__":
    main()
