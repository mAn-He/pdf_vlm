# Multimodal RAG Pipeline

Goal: test whether **page images help** on table/chart/caption questions vs text-only RAG,
without naively dumping every page into Gemma 3.

## Design

```text
Question
  │
  ▼
Text retriever (SAME page_text / section_text index as text-only)
  │  top-k hits
  ▼
Select unique pages (max_images <= top-k)  ← never all pages
  │
  ├─ OCR text for those pages
  └─ page PNG for those pages
  │
  ▼
Gemma 3 multimodal (text + images in → text out)
  │
  ▼
answer + page citation + latency logs
```

### Multimodal retrieval input structure

```python
mm_pages = [
  {
    "page_id": 1,
    "ocr_text": "...",
    "image_path": ".../page_0001.png",
    "score": 0.81,
    "exists": True,
  },
  ...  # only top-k
]
```

## Shared interface with text-only

| Method | TextOnlyRAGPipeline | MultimodalRAGPipeline |
|--------|---------------------|------------------------|
| `modality` | `"text"` | `"multimodal"` |
| `retrieve(q, top_k)` | text hits | **same text retriever** |
| `answer(QAExample)` | `QAPrediction` | `QAPrediction` |
| `run(examples)` | logs under `results/runs/` | same |

Fair comparison rule: both use `indices/<doc_id>/page_text` (or `section_text`).

## Prompt template

System: use OCR **and** images; cite `[page N]`.

User includes:
1. Question
2. OCR text evidence (top-k only)
3. Page image evidence manifest (attached images in order)
4. Instructions to cite supporting pages

Code: `src/pdf_vlm/rag/prompt_builder.py` (`MULTIMODAL_USER_TEMPLATE`)

## Run

```bash
# indexes must already exist (shared with text-only)
python scripts/build_text_index.py --doc-id <DOC> --enrich-pdf-text

python scripts/run_multimodal_rag.py --doc-id <DOC> --retrieval page --dry-run
python scripts/run_multimodal_rag.py --doc-id <DOC> --retrieval page --top-k 3 --max-images 3

# compare
python scripts/run_text_rag.py --doc-id <DOC> --retrieval page --dry-run
```

## Log format (`rag_log.json`)

```json
{
  "example_id": "...",
  "retrieved_pages": [1, 0],
  "used_images": [".../page_0001.png", ".../page_0000.png"],
  "final_answer": "... [page 1]",
  "latency": {
    "retrieval_ms": 1.2,
    "image_load_ms": 3.4,
    "generation_ms": 800.0,
    "e2e_ms": 805.0
  }
}
```
