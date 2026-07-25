#!/usr/bin/env python
"""Download official Gemma 3 4B QAT GGUF (+ mmproj for vision).

Requires HF license accept on:
  https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf
and `huggingface-cli login` (or HF_TOKEN).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from pdf_vlm.utils.io import ensure_dir, load_named_config, project_root, resolve_path, save_json
from pdf_vlm.utils.logging import setup_logging

logger = setup_logging()


def _download_file(repo_id: str, filename: str, dest: Path) -> Path:
    ensure_dir(dest.parent)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        logger.info("Already present: %s (%.2f MB)", dest, dest.stat().st_size / 1e6)
        return dest

    logger.info("Downloading %s / %s", repo_id, filename)
    try:
        cached = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "403" in msg or "GatedRepo" in msg or "restricted" in msg.lower():
            raise SystemExit(
                "Gated model download failed. Do both of:\n"
                "  1) Accept license: https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf\n"
                "  2) Login:  huggingface-cli login   (or set HF_TOKEN)\n"
                f"Original error: {exc}"
            ) from exc
        raise
    if dest.exists():
        dest.unlink()
    shutil.copy2(cached, dest)
    logger.info("Saved %s (%.2f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def download_gemma(model_cfg_path: str = "models/gemma3_4b_qat.yaml", with_mmproj: bool = True) -> dict:
    cfg = load_named_config(model_cfg_path)
    model_path = _download_file(cfg["repo_id"], cfg["filename"], resolve_path(cfg["local_path"]))
    result = {
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "mmproj_path": None,
        "mmproj_exists": False,
    }
    if with_mmproj and cfg.get("mmproj_filename"):
        mmproj_repo = cfg.get("mmproj_repo_id") or cfg["repo_id"]
        mmproj_path = _download_file(
            mmproj_repo,
            cfg["mmproj_filename"],
            resolve_path(cfg["mmproj_local_path"]),
        )
        result["mmproj_path"] = str(mmproj_path)
        result["mmproj_exists"] = mmproj_path.exists()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Gemma 3 4B QAT GGUF (+ mmproj)")
    parser.add_argument("--model-config", default="models/gemma3_4b_qat.yaml")
    parser.add_argument("--with-mmproj", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-marker", action="store_true")
    args = parser.parse_args()

    info = download_gemma(args.model_config, with_mmproj=args.with_mmproj)
    logger.info("download_ok=%s", info)
    if args.smoke_marker:
        save_json(project_root() / "artifacts" / "smoke" / "download_ok.json", info)


if __name__ == "__main__":
    main()
