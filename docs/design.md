# Design Notes

## Goals

Local, reproducible document QA experiments comparing:

1. text-only RAG vs multimodal RAG
2. page-level vs hierarchical retrieval

Generator is fixed: **Gemma 3 4B-IT QAT Q4_0 GGUF** via llama.cpp.

## Pipeline

1. PDF → page PNG (PyMuPDF)
2. PP-StructureV3 → `DocumentArtifact` (cached under `data/processed/`)
3. Index variant → FAISS/numpy under `indices/<doc_id>/<variant>/`
4. Retrieve → prompt → Gemma generate
5. Score with dataset-specific metrics + system metrics

## VRAM policy

OCR and LLM should not share GPU memory at the same time. Ingest/index offline, then load Gemma for QA.

## Fallbacks (dev / CI)

- Text embedder: hashing bag-of-tokens if sentence-transformers missing
- Vision embedder: color histogram if SigLIP missing
- OCR: `--stub` synthetic markdown if PaddleOCR missing
- Vector store: numpy if FAISS missing

These keep unit tests runnable without CUDA/Paddle/llama.cpp.

## Extensibility

- Swap GGUF via `configs/models/`
- Add retrievers under `src/pdf_vlm/retrieve/`
- Add datasets under `src/pdf_vlm/data/` returning `QAExample`
