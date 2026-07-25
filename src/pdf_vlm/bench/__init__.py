"""Local inference practicality benchmarks."""

from pdf_vlm.bench.inference import (
    BenchCase,
    BenchRow,
    MockTimedLLM,
    default_cases,
    run_inference_benchmark,
)

__all__ = [
    "BenchCase",
    "BenchRow",
    "MockTimedLLM",
    "default_cases",
    "run_inference_benchmark",
]
