# Retrieval Granularity: Page-level vs Hierarchical

Compare search units on long PDFs. Both modes share `EmbeddingScorer` and are
reusable by text-only and multimodal RAG.

## Design

### Page-level
```text
query → score every page chunk independently → top-k pages
```

### Hierarchical (coarse-to-fine)
```text
query
  → COARSE: score sections (doc structure)
  → keep top coarse_k sections → allowed pages
  → FINE: score page/paragraph chunks ONLY inside allowed pages
  → top-k final hits
```

## Section / hierarchy generation

`src/pdf_vlm/index/hierarchy.py` strategies (`auto` order):

| Strategy | Signal |
|----------|--------|
| `toc` | `artifact.meta['toc']` |
| `heading` | OCR title/heading blocks |
| `markdown` | `#` / numbered headings |
| `spacing` | blank-line groups / page windows |
| `page` | one section per page (fallback) |

## Common interfaces

```python
class Scorer(Protocol):
    def score(query, texts) -> list[float]

class DetailedRetriever(Protocol):
    def retrieve(query, top_k) -> list[RetrievalHit]
    def retrieve_detailed(query, top_k) -> RetrievalResult  # includes path/coarse/fine
```

`RetrievalResult.format_trace()` prints the visual search path.

## Run comparison

```bash
python scripts/build_retrieval_indexes.py --doc-id <DOC> --enrich-pdf-text
python scripts/compare_retrieval.py --doc-id <DOC> --dataset custom_5 --top-k 3
```

Outputs: `results/runs/retrieval_cmp_*/comparison.json` with
`recall@k`, `gold_answer_contained`, `latency_ms`, and per-example traces.

## Reuse in RAG

```python
# text-only / multimodal both accept hierarchical index
TextOnlyRAGPipeline.from_paths("indices/<doc>/hier_text", retrieval_mode="hierarchical")
MultimodalRAGPipeline.from_paths("indices/<doc>/hier_text", doc_id, retrieval_mode="hierarchical")
```

Note: multimodal still uses **text** hierarchical retrieval first, then loads
images only for final top-k pages.
