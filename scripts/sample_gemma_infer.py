#!/usr/bin/env python
"""Minimal sample: call Gemma3LlamaCpp from Python."""

from __future__ import annotations

from pathlib import Path

from pdf_vlm.llm.gemma_llama_cpp import Gemma3LlamaCpp


def main() -> None:
    llm = Gemma3LlamaCpp.from_config()

    text = llm.generate_text("Write one short sentence about local LLM inference.")
    print("=== TEXT ===")
    print(text.text)
    print(f"latency_ms={text.latency_ms:.1f}")

    sample = Path("artifacts/smoke/sample_vision.png")
    if sample.exists() and llm.vision_supported:
        vision = llm.generate_vision("Describe this image in one sentence.", sample)
        print("=== VISION ===")
        print(vision.text)
        print(f"latency_ms={vision.latency_ms:.1f}")
    else:
        print("Vision skipped (generate sample via scripts/smoke_gemma.py first).")


if __name__ == "__main__":
    main()
