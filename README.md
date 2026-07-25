# pdf-vlm — Local Multimodal Document QA

**Quantized Gemma 3 4B** + **PaddleOCR PP-StructureV3** for private, on-device document question answering.

This repo is a **product-shaped research stack**: reproducible configs, shared retrieval for fair text vs multimodal comparison, evaluation harness, and local inference benchmarks — not a notebook dump.

| Axis | Variants |
|------|----------|
| Generation | text-only RAG vs multimodal RAG (top-k page images) |
| Retrieval | page-level vs hierarchical (coarse → fine) |
| Length | custom PDFs at 5 / 20 / 50 / 100 pages |
| Benchmarks | MP-DocVQA · MMLongBench-Doc subset · custom packs |

> **Colab:** open [`notebooks/pdf_vlm_colab.ipynb`](notebooks/pdf_vlm_colab.ipynb) — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mAn-He/pdf_vlm/blob/main/notebooks/pdf_vlm_colab.ipynb)

> **Status note:** End-to-end ANLS with generation needs Gemma QAT GGUF (`HF_TOKEN` + license accept). Retrieval / dry-run harness works without weights; Colab defaults to a **hashing embedder** to avoid BGE-M3 OOM.

---

## 1. Problem

Long PDFs break naive RAG:

- OCR text alone loses layout, tables, and figures.
- Flat page retrieval over 50–100 pages is noisy and expensive for a small local VLM.
- Cloud multimodal APIs are strong but often unacceptable for **private documents**.

**Goal:** a **local** pipeline that answers document questions with a small multimodal model, measures when text-only fails, and keeps latency/memory practical on a laptop/desktop.

---

## 2. Why Gemma 3 4B?

| Criterion | Choice |
|-----------|--------|
| Size | 4B — fits local RAM/VRAM with headroom for OCR + embeddings |
| Modality | Native **text + image** (SigLIP) via official mmproj |
| Quality vs cost | Smallest Gemma 3 that is still multimodal; enough for short DocQA answers |
| Stack fit | Official **llama.cpp QAT GGUF** path (not Ollama HF-GGUF for vision) |

Refs: [Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3) · [llama.cpp integration](https://ai.google.dev/gemma/docs/integrations/llamacpp)

---

## 3. Why local quantized inference?

| Criterion | Choice |
|-----------|--------|
| Weights | **QAT Q4_0 GGUF** (~3GB) — Google-published, quality-aware quant |
| Runtime | **llama.cpp / llama-cpp-python** — Windows/Linux/macOS, CPU or GPU |
| Privacy | Documents never leave the machine |
| Practicality | Benchmarks TTFT, e2e latency, tok/s, RSS/VRAM (`scripts/bench_gemma_inference.py`) |

Quantization is treated as a **deployment constraint**, not an afterthought: the same Q4_0 build is what we measure and ship.

---

## 4. System pipeline

```text
PDF
  → PP-StructureV3 OCR / layout (or stub / PDF text enrich)
  → normalized DocumentArtifact (pages, sections, OCR text, page images)
  → text index (page_text | hier_text)   ← shared by both RAG modes
        │
        ├─ Text-only RAG ── top-k OCR chunks ── Gemma text generate
        │
        └─ Multimodal RAG ── same top-k pages ── page images + OCR ── Gemma vision
```

**Fairness rule:** multimodal RAG uses the **same text retriever/index** as text-only; images are attached only for the retrieved top-k pages (never all pages).

Hierarchical retrieval: **section (coarse) → page/paragraph (fine)** with logged search paths.

```text
configs/          YAML (model, OCR, retrieval, experiments)
src/pdf_vlm/      llm · ocr · data · index · retrieve · rag · eval · bench
scripts/          download · ingest · index · RAG · harness · bench
data/             raw benchmarks · custom/{5,20,50,100}
indices/          per-doc retrieval indexes
results/          runs · reports · figures · bench
docs/             design + stage docs + final report
```

---

## 5. Datasets

| Dataset | Role | Metrics |
|---------|------|---------|
| **MP-DocVQA** | Multi-page DocVQA | ANLS, page hit / recall@k |
| **MMLongBench-Doc** (subset) | Long multimodal documents | generalized Acc / F1 |
| **Custom 5/20/50/100** | Controlled length buckets | ANLS, EM, F1, recall@k, latency |

Unified schema: `DatasetBundle` → `DatasetDocument(doc_id, pages, qa_pairs)` → `QAExample`.

Demo pack: `data/custom/5/` (Acme 5-page PDF + 3 QA pairs). Longer buckets are scaffolded; fill `data/custom/{20,50,100}/` for full length curves.

---

## 6. Experiment design

**2 × 2 × length matrix**

| Pipeline | Retrieval | Length buckets |
|----------|-----------|----------------|
| text / multimodal | page / hierarchical | 5 · 20 · 50 · 100 |

Config-driven runner:

```bash
python scripts/run_eval_harness.py \
  --config configs/experiments/eval_harness.yaml \
  --datasets custom_5 \
  --pipelines text,multimodal \
  --retrievals page,hierarchical \
  --top-k 3
```

Also:

```bash
# retrieval-only compare
python scripts/compare_retrieval.py --doc-id <DOC> --dataset custom_5

# local inference practicality (TTFT / tok/s / memory)
python scripts/bench_gemma_inference.py --repeats 3
```

Reproducibility: `seed` + `config_snapshot.json` + `seed_snapshot.json` per run.

---

## 7. Key results (current measured)

> Generation quality (ANLS) requires GGUF. Numbers below are from **retrieval / dry-run harness** on `custom_5` (hashing embedder fallback if sentence-transformers missing).

### Retrieval (page vs hierarchical)

| Mode | recall@3 | gold in evidence | latency_ms (mean) |
|------|----------|------------------|-------------------|
| page | **1.00** | **1.00** | ~1.3 |
| hierarchical | **1.00** | **1.00** | ~2.6 |

Source: `results/runs/retrieval_cmp_*`

### Eval harness dry-run (2×2, top_k=3)

| Cell | recall@3 | retrieval latency_ms |
|------|----------|----------------------|
| text × page | 1.00 | ~1.5 |
| text × hierarchical | 1.00 | ~2.5 |
| multimodal × page | 1.00 | ~1.6 |
| multimodal × hierarchical | 1.00 | ~2.8 |

ANLS = 0 in dry-run (no generation). Source: `results/runs/eval_eval_harness_full_*`

### Inference bench

Scaffold + mock timings run without weights. **Real TTFT/tok/s/RSS** after model download — see [`docs/inference_benchmark.md`](docs/inference_benchmark.md).

Full narrative + error analysis: **[`final_report.md`](final_report.md)**

---

## 8. Error analysis (working hypotheses)

| Failure mode | Typical cause | Mitigation in this stack |
|--------------|---------------|--------------------------|
| Text-only misses table/chart QA | OCR flattens structure | Multimodal: attach page image |
| Wrong page @k=1 on long docs | Flat ranking noise | Hierarchical coarse filter |
| High e2e latency | Large top-k / many images | Cap `top_k` and `max_images` (≤3) |
| Empty / weak answers | Model not loaded / dry-run | Download GGUF; disable dry-run |
| Weak retrieval scores | Hashing embedder fallback | `pip install sentence-transformers` (BGE-M3) |

---

## 9. Limits & next steps

**Limits**

- Full GGUF + mmproj not on disk until HF auth.
- Custom 20/50/100 packs and full MP-DocVQA / MMLongBench dumps may be incomplete.
- Production PP-StructureV3 path may still fall back to stub/PDF-text enrich on some machines.
- Hashing embedder understates retrieval quality vs BGE-M3.

**Next**

1. Accept Gemma license + `huggingface-cli login` → `python scripts/download_models.py --with-mmproj`
2. Install `[llm,ocr,index]` extras; rebuild indexes with BGE-M3
3. Run full harness (no `--dry-run`) across length buckets
4. Fill ANLS / qtype / length curves into `docs/final_report.md`

---

## Quick start

### Google Colab

1. Open [pdf_vlm_colab.ipynb](https://colab.research.google.com/github/mAn-He/pdf_vlm/blob/main/notebooks/pdf_vlm_colab.ipynb) (GPU runtime).
2. Run all cells: clone → install → ingest `data/custom` packs → eval harness.
3. Optional: set Colab secret `HF_TOKEN`, flip `DOWNLOAD_GGUF = True`, then `DRY_RUN = False`.

```bash
# same flow locally / in Colab shell
python scripts/colab_prepare_custom.py --buckets 5,20 --stub --enrich-pdf-text --hash-embedder --force
python scripts/run_eval_harness.py --config configs/experiments/eval_hw_wia_colab.yaml --dry-run
```

### Local install

```bash
pip install -e .
# Windows CPU wheel for llama.cpp (recommended)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Optional extras
pip install -e ".[llm,ocr,index,gpu-metrics,viz,dev]"

# HF gated model (required for real generation)
# 1) https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf  (accept license)
# 2) huggingface-cli login   # or set HF_TOKEN
python scripts/download_models.py --with-mmproj
python scripts/smoke_gemma.py

# Demo PDF path
python scripts/ingest_pdf.py data/custom/5/acme_demo_5pages.pdf --stub
python scripts/build_retrieval_indexes.py --doc-id <DOC_ID> --enrich-pdf-text --hash-embedder
python scripts/run_eval_harness.py --datasets custom_5 --dry-run
```

CLI: `python -m pdf_vlm --help`

---

## Docs

| Doc | Topic |
|-----|-------|
| [`final_report.md`](final_report.md) | Experiment report + resume bullets |
| [`docs/resume_bullets.md`](docs/resume_bullets.md) | Resume bullets only |
| [`docs/design.md`](docs/design.md) | Architecture |
| [`docs/gemma_local_inference.md`](docs/gemma_local_inference.md) | Local Gemma setup |
| [`docs/text_only_rag.md`](docs/text_only_rag.md) | Text RAG |
| [`docs/multimodal_rag.md`](docs/multimodal_rag.md) | Multimodal RAG |
| [`docs/retrieval_granularity.md`](docs/retrieval_granularity.md) | Page vs hierarchical |
| [`docs/evaluation_harness.md`](docs/evaluation_harness.md) | Eval matrix |
| [`docs/inference_benchmark.md`](docs/inference_benchmark.md) | TTFT / tok/s / memory |

## License / model terms

Code in this repo: see project license if present. **Gemma weights** are subject to Google’s Gemma license on Hugging Face — accept before download.
