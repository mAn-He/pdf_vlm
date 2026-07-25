# Dataset Loading Layer

공통 인터페이스로 MP-DocVQA / MMLongBench-Doc subset / custom(5·20·50·100)을 적재한다.

## 공통 포맷

```text
DatasetBundle
  name, split, meta
  documents: list[DatasetDocument]
    doc_id
    pages: list[PageRef]          # page_id, image_path|pdf_path, ...
    qa_pairs: list[QAPair]
      qa_id
      question
      answer                      # str | list[str]
      question_type               # text|table|chart|cross-page|...
      evidence_pages
      ...
```

평가 파이프라인은 아래처럼 동일하게 순회한다:

```python
from pdf_vlm.data import get_dataset, print_dataset_stats, bundle_to_examples

ds = get_dataset("mmlongbench", max_docs=10, max_questions=50, seed=42)
bundle = ds.load()
print_dataset_stats(bundle)

for doc in bundle.documents:
    for qa in doc.qa_pairs:
        ...

# RAG/eval flat form
examples = ds.to_examples()  # list[QAExample]
```

## 샘플 입력 경로

### MP-DocVQA
```text
data/raw/mp_docvqa/
  val_questions.json
  documents.json                 # optional
  documents/
    <doc_id>/page_0000.png ...
    <doc_id>.pdf                 # or single PDF
```

### MMLongBench-Doc subset
```text
data/raw/mmlongbench/
  questions.json
  subset.yaml                    # max_docs / max_questions / filters / seed
  documents/
    <doc_id>.pdf
```

`subset.yaml` 예:
```yaml
max_docs: 20
max_questions: 100
include_unanswerable: true
seed: 42
question_types: [text, table, chart, cross-page]
```

### Custom buckets
```text
data/custom/5/
  *.pdf
  questions.json
  manifest.json                  # optional
data/custom/20/
data/custom/50/
data/custom/100/
```

## Smoke

```bash
python scripts/smoke_dataset.py
# artifacts/smoke/dataset_stats.json
```

Fixtures live under `data/fixtures/` (offline).
