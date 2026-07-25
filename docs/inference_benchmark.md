# Gemma 3 4B Local Inference Practicality Benchmark

This stage measures **deployment practicality** of quantized Gemma 3 4B on a
local laptop/desktop — not answer quality.

## Design

```text
prompt = RAG-style evidence (top-k pages × ~chars_per_page)
              │
     ┌────────┴────────┐
     │ text-only       │ image+text (k page images)
     └────────┬────────┘
              ▼
   streaming generate (llama.cpp)
              │
   TTFT · e2e_ms · tok/s · RSS/VRAM
              │
   sweep: page_bucket ∈ {5,20,50,100} × top_k ∈ {1,3,5}
              ▼
   CSV + Markdown + practicality summary
```

### Why this design

| Choice | Rationale |
|--------|-----------|
| Streaming TTFT | Interactive UX is dominated by first-token wait |
| Short `max_tokens` (64) | Local DocQA answers are short; long essays distort latency |
| top-k × page_bucket | Long PDFs should still retrieve few pages; measures prefill growth |
| Synthetic page images | Reproducible multimodal cost without requiring a specific PDF |
| Q4_0 GGUF | The model we actually deploy (~3GB weights) |

## Metrics

| Metric | Definition |
|--------|------------|
| **TTFT** | Time from request start to first streamed token |
| **e2e_ms** | Full response wall time |
| **tokens/sec** | Decode throughput after first token (fallback: e2e tok/s) |
| **peak_rss_mb** | Process RSS peak during the call |
| **peak_vram_mb** | GPU memory peak (if `pynvml` available) |

## Run

```bash
# CI / no weights
python scripts/bench_gemma_inference.py --mock

# Real machine (after HF login + download)
python scripts/download_models.py --with-mmproj
python scripts/bench_gemma_inference.py --repeats 3

# Subset
python scripts/bench_gemma_inference.py --modalities text --top-ks 1,3 --page-buckets 5,20
```

Config: `configs/experiments/inference_bench.yaml`

## Output layout

```text
results/bench/gemma_infer_bench_<timestamp>/
  raw_results.csv | raw_results.json
  aggregate.csv | aggregate.json
  host_info.json
  config_snapshot.json
  practicality.json
  report.md
  summary.json
  images/bench_page_*.png

results/reports/gemma_infer_bench_*.md
```

## Practicality score

| Score | Meaning |
|-------|---------|
| `interactive_local` | ≥20 tok/s and low TTFT — comfortable interactive QA |
| `practical_local_qa` | ≥8 tok/s and e2e under ~8s for short answers |
| `limited` | usable for offline/batch demos only |

Interpretation thresholds are documented in the generated Markdown report.

## Code map

| Path | Role |
|------|------|
| `src/pdf_vlm/llm/gemma_llama_cpp.py` | Streaming TTFT + tok/s on `GenerationResult` |
| `src/pdf_vlm/bench/inference.py` | Cases, runner, aggregates, practicality |
| `scripts/bench_gemma_inference.py` | CLI |
| `configs/experiments/inference_bench.yaml` | Defaults |
