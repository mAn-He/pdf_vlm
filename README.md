# pdf-vlm — Local Multimodal Document QA

**Gemma 3 4B (QAT Q4_0)** + **PaddleOCR PP-StructureV3**. 200페이지 이상 사내 문서를
외부로 내보내지 않고 로컬에서 질의응답하는 파이프라인, 그리고 그 위에서 돌린 통제 실험.

> **Status:** Colab T4, 20/20 cells, 152 rows (38 questions × 4 conditions).
> 셀당 n=4–11. 모든 수치는 **방향성 지표이며 통계적 검정력 없음.**

---

## TL;DR

1. **멀티모달의 "실패"는 대부분 측정의 실패.** raw ANLS로는 MM 0.054 vs Text 0.411.
   인용 토큰을 떼고 한국어 출력을 강제해 재생성하면 같은 런에서 **MM 0.426 vs Text 0.300**으로 역전됨.
   지표와 출력 언어, 인용 포맷을 통제하지 않으면 모달리티 비교 자체가 성립하지 않음.
2. **검색을 고정해야 생성을 비교할 수 있음.** text와 multimodal이 동일한 텍스트 인덱스,
   동일한 retriever, 동일한 hit 리스트를 씀. MM은 거기에 페이지 이미지만 얹음(`max_images≤2`).
3. **환경을 넘나드는 평가는 조용히 죽음.** `doc_id`를 경로 해시에서 PDF 내용 해시로
   바꾸기 전까지 Windows와 Colab에서 전 셀이 `n_rows=0`으로 스킵됨.

---

## 구조

```
[오프라인]  PDF ─▶ 페이지 렌더링 ─▶ PP-StructureV3 (OCR+레이아웃)
                 └▶ DocumentArtifact ─┬▶ page_text  (페이지당 1청크)
                                      └▶ hier_text  (섹션 coarse + 페이지 fine)

[온라인]    질문 ─▶ retriever(page | hierarchical) ─▶ RetrievalHit
                 ├▶ text:       OCR 텍스트만        ─▶ Gemma 3 4B
                 └▶ multimodal: OCR + 페이지 PNG(≤2) ─▶ Gemma 3 4B + mmproj
```

비전은 **검색에 쓰지 않음.** 텍스트로 검색한 뒤 생성 단계에서만 이미지를 붙임.

| 설정 | 값 |
|---|---|
| 모델 / 런타임 | Gemma 3 4B QAT Q4_0 GGUF + mmproj / llama.cpp |
| 생성 | `max_tokens=128`, `temperature=0.1`, seed 42 |
| 검색 | `top_k=3` (hier: `coarse_k=5`) |
| 임베더 | Colab 기본 hashing (BGE-M3는 RAM 여유 시) |
| 데이터 | 200p+ 리포트를 5/10/20/50/100p로 절단한 길이 팩 |

---

## 결과

### 멀티모달 — 프로토콜이 점수를 만듦

| 조건 | MM | Text |
|---|---|---|
| raw ANLS | 0.054 | 0.411 |
| `[page N]` 오프라인 제거 | 0.159 | 0.411 (불변) |
| **한국어 강제 + 인용 제거로 재생성** | **0.426** | **0.300** |

MM 응답 76건 중 **57건이 영어**, 정답은 한국어. EM은 raw 기준 전 구간 0.0.

> ⚠️ 마지막 행은 **별도 런**. 프롬프트 언어와 인용 제거, 인덱스 재생성 3개가 동시에 바뀌었고
> 텍스트도 0.411에서 0.300으로 함께 떨어짐. 런 간 비교(0.054 → 0.426)가 아니라
> **같은 런 안의 비교(MM 0.426 vs Text 0.300)**만 주장 가능. 변수 분리는 미완.

### 길이별 (raw ANLS)

| Bucket | Text·page | Text·hier | MM·page | Recall@3 | n |
|---|---|---|---|---|---|
| 5p | **0.750** | 0.750 | 0.141 | 1.00 | 4 |
| 10p | 0.183 | 0.183 | 0.000 | 0.75 | 4 |
| 20p | 0.371 | 0.371 | 0.063 | 0.69 | 8 |
| 50p | 0.542 | 0.270 | 0.045 | 0.59 | 11 |
| 100p | 0.542 | 0.270 | 0.045 | 0.59 | 11 |

- **50p와 100p가 같은 건 버그가 아님.** 모든 질문의 정답 근거가 상대 28페이지 이내라
  두 팩이 동일한 11문항을 공유함. 100p는 distractor만 늘림.
  50에서 100으로 가는 구간은 **길이 축으로 식별되지 않음.**
- **hier가 page보다 낮은 결과는 임베더와 교란돼 있음.** coarse 단계는 의미 유사도가 필요한데
  hashing 임베더는 사실상 어휘 매칭이라 계층 검색이 구조적으로 불리함.
  BGE-M3로 재현하기 전까지 계층 검색이 열등하다고 말할 수 없음.
- `gold_answer_contained`가 **0.50**으로 양 파이프라인 동일. 근거 자체가 절반만 들어왔다는
  뜻이고 생성 성능의 상한이 0.5라는 의미. 이 실험의 진짜 병목은 생성이 아니라 검색.

### 실용성 (별도 벤치)

텍스트 디코딩 약 26 tok/s, TTFT 약 57 ms로 로컬 인터랙티브 가능. 멀티모달 e2e는 약 **2.9배**.
Peak VRAM은 8.8에서 15.5 GB 사이이며 text와 MM 컨텍스트를 동시에 적재하면 OOM이 나므로
순차 실행 필요.

---

## 한계

- 셀당 n=4–11. **방향성이지 결론이 아님.**
- 모달리티 순위는 출력 언어, 인용 포맷과 교란돼 있음(위 ⚠️).
- page와 hierarchical 비교는 임베더와 교란돼 있음.
- 50p와 100p는 독립적인 길이 대조가 아님.
- 근거 포함률 50%가 EM 상한을 막음.

**다음**: 재생성 런의 3개 변수 분리(인덱스를 고정한 뒤 프롬프트만 변경),
BGE-M3로 인덱스를 재구축한 뒤 page와 hierarchical 재측정,
50에서 99페이지 사이에 근거가 있는 문항 추가, LLM-as-judge 추가.

---

## Quick start

```bash
# Colab (GPU)
python scripts/colab_prepare_custom.py --buckets 5,10,20,50,100 --enrich-pdf-text
python scripts/sync_questions_to_indices.py
python scripts/run_eval_harness.py --config configs/experiments/eval_hw_wia_colab.yaml --pipelines text
python scripts/run_eval_harness.py --config configs/experiments/eval_hw_wia_colab.yaml --pipelines multimodal
```

```bash
# Local
pip install -e ".[llm,ocr,index,gpu-metrics,viz,dev]"
python scripts/download_models.py --with-mmproj   # HF에서 Gemma 라이선스 동의 필요
python scripts/run_eval_harness.py --config configs/experiments/eval_hw_wia_colab.yaml --datasets custom_5
```

`notebooks/pdf_vlm_colab.ipynb`에 전체 실행 흐름이 있음. CLI: `python -m pdf_vlm --help`

설계와 세부 결과: [`docs/design.md`](docs/design.md),
[`docs/multimodal_rag.md`](docs/multimodal_rag.md),
[`docs/retrieval_granularity.md`](docs/retrieval_granularity.md),
[`docs/evaluation_harness.md`](docs/evaluation_harness.md)

**Gemma 가중치는 Google의 Gemma 라이선스를 따름.**
