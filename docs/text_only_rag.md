# Text-only RAG Baseline

Strict OCR-text baseline for fair comparison with later multimodal RAG.

## Design

```text
DocumentArtifact (parsed OCR JSON)
  -> text index (page_text | section_text)
  -> Retriever.retrieve(question, top_k)
  -> text-only prompt (no images)
  -> Gemma 3 4B generate_text
  -> QAPrediction + rag_log.json
```

| Mode | Chunk unit | Eval pages |
|------|------------|------------|
| `page` | page markdown (+ tables) | retrieved page ids |
| `section` | section summary_text | pages belonging to section |

Shared interface with future multimodal RAG:
- `Retriever.retrieve(query, top_k) -> list[RetrievalHit]`
- `pipeline.answer(QAExample) -> QAPrediction`
- logged fields: `answer`, `retrieved_pages`, latency breakdown

## Prompt template

System:
```text
Use ONLY the OCR text evidence... If insufficient, reply exactly: unanswerable.
```

User (see `TEXT_ONLY_USER_TEMPLATE` in `prompt_builder.py`):
```text
# Document Evidence (OCR text only)
[Evidence 1 | level=page|section | pages=[...] | score=...]
...

# Question
...

# Instructions
- Base the answer strictly on the evidence above.
- Output only the final answer.
```

## Build & run

```bash
# Enrich stub OCR from PDF text layer (optional but recommended for demo PDF)
python scripts/build_text_index.py --doc-id acme_demo_5pages_de31152fee --enrich-pdf-text

# Dry-run (retrieval + logs, no Gemma)
python scripts/run_text_rag.py --doc-id acme_demo_5pages_de31152fee --retrieval page --dry-run
python scripts/run_text_rag.py --doc-id acme_demo_5pages_de31152fee --retrieval section --dry-run

# Full generation (requires downloaded Gemma GGUF)
python scripts/run_text_rag.py --doc-id acme_demo_5pages_de31152fee --retrieval page
```

## Log format (`results/runs/<run_id>/rag_log.json`)

```json
[
  {
    "example_id": "c5_q1",
    "doc_id": "...",
    "question": "...",
    "answer": "...",
    "gold_answers": ["1998"],
    "retrieved_pages": [0, 1],
    "evidence_pages": [0],
    "latency": {
      "retrieval_ms": 12.3,
      "generation_ms": 450.0,
      "e2e_ms": 462.5
    },
    "retrieval_hits": [
      {
        "chunk_id": "...::page::0",
        "score": 0.82,
        "level": "page",
        "page_ids": [0],
        "section_id": null,
        "text_preview": "..."
      }
    ],
    "meta": {"baseline": "text_only_rag", "retrieval_mode": "page", "modality": "text"}
  }
]
```

Also written: `predictions.json`, `run_summary.json`, `metrics.json`.

## Code map

- Index: `src/pdf_vlm/index/text_indexer.py`
- Retrievers: `retrieve/page_retriever.py`, `retrieve/section_retriever.py`
- Pipeline: `src/pdf_vlm/rag/text_only.py` (`TextOnlyRAGPipeline`)
- Prompt: `src/pdf_vlm/rag/prompt_builder.py`
