"""Tests for inference practicality benchmark (mock path, no GGUF)."""

from __future__ import annotations

from pathlib import Path

from pdf_vlm.bench.inference import (
    MockTimedLLM,
    build_evidence_prompt,
    build_markdown_report,
    default_cases,
    interpret_practicality,
    run_case,
    run_inference_benchmark,
)
from pdf_vlm.llm.gemma_llama_cpp import _estimate_completion_tokens, _throughput


def test_throughput_helpers():
    assert _estimate_completion_tokens("abcd", None) == 1
    assert _estimate_completion_tokens("x", 12) == 12
    tps = _throughput(11, e2e_ms=1100.0, ttft_ms=100.0)
    assert tps is not None
    assert abs(tps - 10.0) < 1e-6  # 10 tokens in 1.0s decode


def test_evidence_prompt_scales_with_top_k():
    p1 = build_evidence_prompt(question="q", top_k=1, page_bucket=100, chars_per_page=200)
    p5 = build_evidence_prompt(question="q", top_k=5, page_bucket=100, chars_per_page=200)
    assert len(p5) > len(p1)
    assert "top-5 of 100-page" in p5


def test_default_cases_grid():
    cases = default_cases(top_ks=[1, 3], page_buckets=[5, 20], modalities=["text"])
    assert len(cases) == 4  # 2 buckets × 2 ks
    assert all(c.modality == "text" for c in cases)


def test_mock_run_case():
    llm = MockTimedLLM()
    case = default_cases(top_ks=[1], page_buckets=[5], modalities=["text"], max_tokens=16)[0]
    row = run_case(llm, case, run_id="t", image_paths=None, repeat=0, load_rss_mb=100.0)
    assert row.e2e_ms > 0
    assert row.ttft_ms is not None
    assert row.tokens_per_sec is not None


def test_interpret_practicality_and_report():
    aggregates = [
        {
            "modality": "text",
            "page_bucket": 5,
            "top_k": 1,
            "n": 2,
            "prompt_chars_mean": 1000,
            "ttft_ms_mean": 200.0,
            "e2e_ms_mean": 800.0,
            "tokens_per_sec_mean": 25.0,
            "peak_rss_mb_max": 3500.0,
            "peak_vram_mb_max": None,
            "n_images": 0,
        },
        {
            "modality": "text",
            "page_bucket": 5,
            "top_k": 3,
            "n": 2,
            "prompt_chars_mean": 3000,
            "ttft_ms_mean": 350.0,
            "e2e_ms_mean": 1200.0,
            "tokens_per_sec_mean": 24.0,
            "peak_rss_mb_max": 3600.0,
            "peak_vram_mb_max": None,
            "n_images": 0,
        },
        {
            "modality": "multimodal",
            "page_bucket": 5,
            "top_k": 3,
            "n": 2,
            "prompt_chars_mean": 3000,
            "ttft_ms_mean": 500.0,
            "e2e_ms_mean": 2400.0,
            "tokens_per_sec_mean": 15.0,
            "peak_rss_mb_max": 4000.0,
            "peak_vram_mb_max": None,
            "n_images": 3,
        },
    ]
    prac = interpret_practicality(aggregates, host={"platform": "test"})
    assert prac["practicality_score"] == "interactive_local"
    assert prac["highlights"]
    md = build_markdown_report(
        run_id="demo",
        host={"platform": "test", "python": "3.11"},
        aggregates=aggregates,
        practicality=prac,
        config={"seed": 1, "max_tokens": 64, "repeats": 2},
    )
    assert "TTFT" in md
    assert "Practicality summary" in md


def test_run_inference_benchmark_mock(tmp_path: Path):
    result = run_inference_benchmark(
        mock=True,
        cases=default_cases(
            top_ks=[1, 3],
            page_buckets=[5],
            modalities=["text", "multimodal"],
            max_tokens=8,
        ),
        repeats=1,
        warmup=0,
        out_dir=tmp_path / "bench",
        seed=0,
        max_tokens=8,
    )
    assert result["n_rows"] == 4
    out = Path(result["out_dir"])
    assert (out / "raw_results.csv").exists()
    assert (out / "aggregate.csv").exists()
    assert (out / "report.md").exists()
    assert (out / "practicality.json").exists()
    assert "practicality_score" in result["practicality"]
