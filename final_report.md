# Final Experiment Report — Local Multimodal Document QA

**Project:** `pdf-vlm`  
**Stack:** Gemma 3 4B QAT Q4_0 (llama.cpp) · PaddleOCR PP-StructureV3 · FAISS / numpy · YAML harness  
**Comparisons:** text-only vs multimodal RAG · page vs hierarchical retrieval · length buckets 5/20/50/100  

**Report status:** Engineering-complete scaffolding with **measured retrieval / dry-run** results.  
**Blocked for full answer-quality curves:** Hugging Face auth + GGUF download (verified absent on authoring machine).

---

## 1. Executive summary

We built a **local, quantized multimodal DocQA system** that:

1. Parses PDFs with structure-aware OCR.
2. Retrieves with a **shared text index** so text-only and multimodal differ only at generation.
3. Optionally narrows search with **hierarchical (coarse→fine)** retrieval.
4. Evaluates via a config-driven harness (CSV / JSON / Markdown) and a separate **inference practicality** bench (TTFT, tok/s, memory).

**Headline measured facts (custom_5, dry-run / retrieval):**

| Finding | Number |
|---------|--------|
| Page retrieval recall@3 | **1.00** |
| Hierarchical recall@3 | **1.00** |
| Gold answer in retrieved evidence | **1.00** |
| Hier vs page retrieval latency | **~2.0×** (~2.6 ms vs ~1.3 ms on tiny index) |
| End-to-end ANLS with Gemma | **pending** (GGUF not on disk) |

---

## 2. Motivation & design principles

| Principle | Implementation |
|-----------|----------------|
| Privacy-first | Local GGUF only; no cloud LLM required |
| Fair modality A/B | Same retriever → multimodal adds top-k images only |
| Long-doc realism | Length buckets + hierarchical coarse filter |
| Product engineering | Config snapshots, seeds, CSV/MD reports, CLI scripts |

### Why Gemma 3 4B + Q4_0

- Smallest Gemma 3 with **native vision**, official **QAT GGUF + mmproj**.
- ~3GB weights leave RAM for OCR and embeddings on a 16–32GB laptop.
- llama.cpp path is reproducible on Windows (prebuilt CPU wheels) and GPU builds.

---

## 3. System overview

```text
PDF → OCR/layout → DocumentArtifact
                 → page_text / hier_text indexes
                 → Retriever (page | hierarchical)
                 → Generator
                      ├ text-only: OCR context
                      └ multimodal: OCR + top-k page images
```

Key modules: `src/pdf_vlm/{llm,ocr,data,index,retrieve,rag,eval,bench}`.

---

## 4. Datasets & protocol

| Corpus | Use |
|--------|-----|
| MP-DocVQA | Multi-page extractive QA (ANLS, page hit) |
| MMLongBench-Doc subset | Long multimodal docs (Acc/F1) |
| Custom 5/20/50/100 | Controlled length ablation |

**Matrix:** `{text, multimodal} × {page, hierarchical} × {5,20,50,100}` with fixed `top_k` (default 3), shared seed, config snapshot.

Harness: `scripts/run_eval_harness.py`  
Retrieval A/B: `scripts/compare_retrieval.py`  
Inference UX: `scripts/bench_gemma_inference.py`

---

## 5. Results

### 5.1 Retrieval quality (custom_5)

| Mode | recall@1 | recall@3 | gold_contained | latency_ms |
|------|----------|----------|----------------|------------|
| page | 0.67 | **1.00** | **1.00** | 1.3 |
| hierarchical | 0.67 | **1.00** | **1.00** | 2.6 |

On a 5-page demo, both modes recover evidence at k=3. Hierarchical adds coarse section filtering (logged KEEP/drop path) at modest extra latency.

### 5.2 Pipeline × retrieval dry-run (harness)

| Cell | recall@3 | retrieval_ms | notes |
|------|----------|--------------|-------|
| text × page | 1.00 | 1.5 | baseline |
| text × hierarchical | 1.00 | 2.5 | coarse→fine |
| multimodal × page | 1.00 | 1.6 | + image load path |
| multimodal × hierarchical | 1.00 | 2.8 | heaviest cell |

**ANLS/EM = 0** here because `--dry-run` skips generation (intentional until GGUF is available).

### 5.3 Document length vs performance

| Bucket | Status | Expected pattern (to confirm with full run) |
|--------|--------|-----------------------------------------------|
| 5 | measured retrieval | High recall@3; flat page retrieval sufficient |
| 20 | scaffold | Page@1 recall drops; hierarchical helps coarse recall |
| 50 | scaffold | Text-only struggles on layout-heavy pages |
| 100 | scaffold | Must keep top-k small; hier reduces fine-score candidates |

**Claim to validate after model download:** accuracy decays with length mainly via **retrieval miss @ small k**, not only via generator capacity. Hierarchical retrieval should flatten the decay if section heuristics fire.

### 5.4 Local inference practicality

Bench design measures TTFT, e2e, tok/s, RSS/VRAM for text vs image+text while sweeping top-k evidence under length buckets.

| Score label | Meaning |
|-------------|---------|
| `interactive_local` | ≥20 tok/s, low TTFT |
| `practical_local_qa` | ≥8 tok/s, short-answer e2e ≲ 8s |
| `limited` | batch / demo only |

**Authoring machine:** mock bench only (no GGUF). After download, fill real numbers into `results/bench/gemma_infer_bench_*/report.md`.

---

## 6. Analysis by theme

### 6.1 When text-only breaks

| Question type | Why text-only fails | Multimodal help |
|---------------|---------------------|-----------------|
| **table** | Cell alignment / headers lost in OCR dump | Page image preserves grid |
| **chart** | Numbers/legends are visual | Vision encoder reads plot |
| **layout / stamp** | Sparse OCR | Image context |
| **cross-page** | Need multi-hop evidence | Still needs good retrieval; images alone ≠ multi-page reasoning |

Text-only remains strong on clean prose spans (dates, names, short facts) when OCR is complete — as in the Acme demo “founded in 1998” style items.

### 6.2 Multimodal RAG advantages

1. **Same retrieval, richer generation** — isolates modality gain.
2. **Bounded cost** — only top-k images (never full PDF).
3. **Complements OCR** — recovers signal OCR systematically drops.

**Trade-off:** higher TTFT/e2e and memory; keep `max_images ≤ top_k ≤ 3` for local UX.

### 6.3 Hierarchical retrieval — pros / cons

| Pros | Cons |
|------|------|
| Coarse-to-fine matches long manuals / TOC structure | Heuristic sections (spacing/markdown) can merge whole docs |
| Shrinks fine-search space → potential latency win at scale | Extra section index + scoring (~2× on tiny demo) |
| Debuggable path logs (KEEP/drop) | Bad coarse recall permanently drops gold pages |

**Best when:** headings/TOC exist. **Fallback:** page-level if `auto` strategy collapses to one mega-section.

### 6.4 Local inference trade-offs

| Axis | Trade-off |
|------|-----------|
| Quality vs size | 4B Q4_0 ≪ cloud 70B+, but private & free at inference |
| Speed vs modality | Multimodal ≫ text latency; use only when needed |
| Context vs UX | Larger top-k / page_bucket ↑ prefill time |
| Memory | Q4_0 leaves room for OCR+FAISS on 16GB+ machines |

**Product stance:** default **text + page retrieval**; escalate to multimodal and/or hierarchical when question type or length demands it.

---

## 7. Error analysis checklist

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| ANLS=0, empty answers | dry-run / missing GGUF | Login HF → download → rerun harness |
| recall@3 low on long PDF | weak embedder / flat ranking | Install BGE-M3; try hierarchical |
| Hier drops gold page | coarse miss | Improve TOC/heading strategy; raise `coarse_k` |
| OOM / thrashing | n_ctx too large + vision | Lower `n_ctx`, `n_gpu_layers`, `max_images` |
| Vision errors | missing mmproj / wrong handler | `mmproj-model-f16-4B.gguf` + MTMDChatHandler |

---

## 8. Reproducibility gate (HF token)

Checked on authoring host (2026-07-25):

| Check | Result |
|-------|--------|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | **missing** |
| `~/.cache/huggingface/token` | **missing** |
| `huggingface_hub` login | **not logged in** |
| `models/gemma-3-4b-it-q4_0.gguf` | **missing** |
| `models/mmproj-model-f16-4B.gguf` | **missing** |

**Yes — without a Hugging Face token (and Gemma license accept), the gated GGUF cannot be downloaded, so full generation benchmarks cannot be completed.**

```bash
# Fix
# 1) Accept: https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf
# 2) Login:
huggingface-cli login
# or: setx HF_TOKEN "hf_..."
python scripts/download_models.py --with-mmproj
python scripts/smoke_gemma.py
python scripts/run_eval_harness.py --datasets custom_5 --top-k 3
python scripts/bench_gemma_inference.py --repeats 3
```

---

## 9. Limits & next steps

1. Obtain HF access → fill **ANLS / qtype / length** tables.  
2. Build custom_20/50/100 packs for real length curves.  
3. Swap hashing embedder → **BGE-M3**.  
4. Production PP-StructureV3 on scanned PDFs.  
5. Optional GPU llama-cpp build for interactive tok/s targets.

---

## 10. Resume bullets (draft)

1. **Built a local multimodal Document QA stack** (Gemma 3 4B QAT GGUF + PaddleOCR + FAISS) with a fair **text-only vs multimodal RAG** comparison that shares retrieval and attaches only top-k page images at generation time.

2. **Implemented page-level and hierarchical (coarse-to-fine) retrieval**, plus a config-driven evaluation harness reporting ANLS/EM/F1, recall@k, latency, and RSS/VRAM across 5–100 page length buckets (CSV/JSON/Markdown).

3. **Instrumented quantized local inference practicality** (TTFT, tokens/sec, memory) for text vs image+text workloads, documenting deployability trade-offs for private on-device DocQA on laptop/desktop hardware.

---

## Appendix — artifact map

| Artifact | Path pattern |
|----------|--------------|
| Retrieval compare | `results/runs/retrieval_cmp_*/` |
| Eval harness | `results/runs/eval_*/` · `results/reports/eval_*.md` |
| Inference bench | `results/bench/gemma_infer_bench_*/` |
| Stage docs | `docs/*.md` |
