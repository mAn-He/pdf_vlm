# Evaluation harness

Config-driven matrix runner for local Document QA experiments.

## Comparisons

1. **Pipeline:** text-only RAG vs multimodal RAG  
2. **Retrieval:** page-level vs hierarchical (coarse-to-fine)  
3. **Length:** 5 / 20 / 50 / 100 page buckets (`custom_*` datasets)

Both pipelines share the same text indexes (`page_text` / `hier_text`) so
retrieval is fair; multimodal only adds top-k page images at generation time.

## Design

```text
configs/experiments/*.yaml
        │
        ▼
 EvaluationHarness
   ├─ set seed + write config/seed snapshot
   ├─ for each cell (dataset × pipeline × retrieval × top_k)
   │     └─ TextOnlyRAGPipeline | MultimodalRAGPipeline.answer()
   ├─ EvalRow per example (metrics + latency + memory)
   ├─ aggregate tables (overall / qtype / page bucket / cell)
   └─ export CSV + JSON + Markdown (+ optional PNG plots)
```

## Config example

See `configs/experiments/eval_harness.yaml` and `matrix_2x2.yaml`.

```yaml
seed: 42
datasets: [custom_5, custom_20, custom_50, custom_100]
pipelines: [text, multimodal]
retrievals: [page, hierarchical]
top_k: [3]
dry_run: false
skip_missing_dataset: true
skip_missing_index: true
```

CLI overrides:

```bash
python scripts/run_eval_harness.py \
  --config configs/experiments/eval_harness.yaml \
  --dry-run \
  --datasets custom_5 \
  --pipelines text,multimodal \
  --retrievals page,hierarchical \
  --top-k 3 \
  --seed 42
```

## Outputs

```text
results/runs/eval_<name>_<timestamp>/
  config_snapshot.json
  seed_snapshot.json
  predictions.csv | predictions.json
  aggregates.json
  tables/*.csv
  report.md
  cells/<cell_id>/metrics.json
  summary.json

results/reports/eval_*.md
results/figures/eval_*/{length_vs_accuracy,pipeline_latency,question_type}.png
```

### Per-row fields

`dataset`, `doc_id`, `question_id`, `question_type`, `pipeline_type`,
`retrieval_type`, `answer`, `gold_answer`, `correctness` (ANLS by default),
`latency_ms`, `recall@k`, `peak_rss_mb`, `peak_vram_mb`, …

## Reproducibility

- `seed` applied to Python / NumPy / Torch (when available)
- Full config + seed snapshot written next to results

## Plots

```bash
pip install matplotlib
python scripts/plot_eval_results.py --run-dir results/runs/eval_...
```
