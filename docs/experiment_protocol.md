# Experiment Protocol

## Variants

| ID | modality | retrieval | config |
|----|----------|-----------|--------|
| page_text | text | page | `configs/retrieval/page_text.yaml` |
| hier_text | text | hierarchical | `configs/retrieval/hier_text.yaml` |
| page_mm | multimodal | page | `configs/retrieval/page_mm.yaml` |
| hier_mm | multimodal | hierarchical | `configs/retrieval/hier_mm.yaml` |

## Datasets

1. **custom_5 / 20 / 50 / 100** — local PDFs under `data/custom/<n>/` + `questions.json`
2. **MP-DocVQA** — place export in `data/raw/mp_docvqa/`; metric: ANLS + page recall@k
3. **MMLongBench-Doc subset** — place export in `data/raw/mmlongbench/`; metric: gen. Acc/F1

## Run steps

```bash
# For each document in the eval set
python scripts/ingest_pdf.py <pdf>
python scripts/build_index.py --doc-id <id> --modality text --retrieval page
python scripts/build_index.py --doc-id <id> --modality text --retrieval hierarchical
python scripts/build_index.py --doc-id <id> --modality multimodal --retrieval page
python scripts/build_index.py --doc-id <id> --modality multimodal --retrieval hierarchical

python scripts/run_experiment.py --doc-id <id>
```

## Outputs

- `results/runs/<run_id>/metrics.json`
- `results/runs/<run_id>/predictions.json`
- `results/tables/matrix_*.csv`

## Primary scores

- MP-DocVQA → `anls_mean`
- MMLongBench → `mmlong_f1`
- Custom → `anls_mean` (+ report `recall@k`, latency, VRAM)
