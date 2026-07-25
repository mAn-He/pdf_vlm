# Gemma 3 4B Quantized Local Inference (llama.cpp)

이 문서는 **RAG/OCR 없이** Gemma 3 4B를 로컬에서 실행하는 첫 구현 단계만 다룬다.

## 왜 이 경로인가 (단일 경로)

| 항목 | 선택 | 이유 |
|------|------|------|
| 런타임 | **llama.cpp** via `llama-cpp-python` | Google 공식 통합 문서가 있고, Windows/Linux/macOS 로컬 추론에 적합 |
| 모델 | **`google/gemma-3-4b-it-qat-q4_0-gguf`** | 공식 QAT(Q4_0). bf16급 품질을 ~3GB대로 유지 |
| Vision | 같은 레포의 **`mmproj-model-f16-4B.gguf`** | 공식 HF 레포에 text GGUF + mmproj가 함께 제공됨 |
| Python API | `Gemma3LlamaCpp` + `Gemma3ChatHandler`/`MTMDChatHandler` | text / image+text를 동일 클래스로 호출 |

공식 참고:
- [Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3)
- [Introducing Gemma 3](https://developers.googleblog.com/introducing-gemma3/)
- [Gemma + llama.cpp](https://ai.google.dev/gemma/docs/integrations/llamacpp)
- HF: https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf

### Vision 지원 요약

- Gemma 3 (4B/12B/27B)는 **multimodal** (SigLIP vision encoder).
- llama.cpp에서 이미지는 **mmproj**가 필요하다.
- 공식 CLI 예: `llama-gemma3-cli -hf google/gemma-3-4b-it-qat-q4_0-gguf --image <img> -p "..."`
- 본 레포 Python 경로는 mmproj + chat handler로 동일 기능을 감싼다.
- **Ollama의 HF GGUF 경로는 이미지 미지원**이므로 사용하지 않는다.

### 왜 quantized (QAT Q4_0)인가

1. **로컬 VRAM/RAM 현실성**: 4B QAT Q4_0 ≈ 3.16GB 수준으로 로드 가능 (공식 카드).
2. **품질 보존**: QAT는 post-training quant보다 bf16에 가까운 품질을 목표로 함.
3. **공식 권장 배포 포맷**: Google이 llama.cpp/Ollama용 QAT GGUF를 직접 배포.
4. **문서 QA 실험 전제**: OCR/임베딩과 VRAM을 나눠 쓰기 쉬움.

---

## 1. OS별 전제 조건

### 공통
- Python **3.11+**
- RAM **16GB+** 권장 (CPU 추론 시 더 여유 필요)
- 디스크 **8GB+** (GGUF ~3.2GB + mmproj ~0.8GB + 캐시)
- Hugging Face 계정 + **Gemma license accept**
- `huggingface-cli login` 또는 `HF_TOKEN`

### Windows 10/11 (이 프로젝트 기본)
- Visual Studio Build Tools (C++), 또는 **미리 빌드된 wheel** 사용
- GPU 가속(선택): CUDA Toolkit + GPU용 `llama-cpp-python` 빌드
- CPU만으로도 text/vision smoke 가능 (느림)

### Linux
- `build-essential`, `cmake`
- CUDA 사용 시 NVIDIA driver + CUDA toolkit

### macOS
- Apple Silicon: Metal 가속이 기본에 가깝게 동작
- Homebrew로 `llama.cpp` CLI를 쓸 수도 있으나, 이 레포는 Python wrapper를 표준으로 함

---

## 2. 설치 절차 (단계별)

### Step A — 프로젝트 의존성

```bash
cd d:\deeplearning\pdf_vlm
pip install -e .
```

### Step B — llama-cpp-python

**Windows (권장: prebuilt CPU wheel — VS Build Tools 불필요):**

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

이 환경에서 검증된 버전: `llama-cpp-python==0.3.34` (handlers: `MTMDChatHandler`).

**Linux/macOS CPU:**

```bash
pip install -U llama-cpp-python
```

**CUDA (선택, 빌드 도구 필요):**

```bash
# Windows
set CMAKE_ARGS=-DGGML_CUDA=on
pip install -U llama-cpp-python --force-reinstall --no-cache-dir

# Linux
CMAKE_ARGS="-DGGML_CUDA=on" pip install -U llama-cpp-python --force-reinstall --no-cache-dir
```

설치 확인:

```bash
python -c "from llama_cpp import Llama; print('llama-cpp-python OK')"
python -c "from llama_cpp.llama_chat_format import MTMDChatHandler; print('vision handler OK')"
```

### Step C — HF 로그인 + 라이선스

1. 브라우저에서 https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf 접속 후 license 동의
2. 토큰 로그인:

```bash
pip install -U huggingface_hub
huggingface-cli login
```

### Step D — 모델 다운로드 (text GGUF + mmproj)

```bash
python scripts/download_models.py --with-mmproj --smoke-marker
```

예상 산출물:
- `models/gemma-3-4b-it-q4_0.gguf`
- `models/mmproj-model-f16-4B.gguf`
- `artifacts/smoke/download_ok.json`

### Step E — Smoke 테스트

```bash
python scripts/smoke_gemma.py
```

성공 시:
- `artifacts/smoke/gemma_text_ok.json`
- `artifacts/smoke/gemma_vision_ok.json`
- `artifacts/smoke/gemma_ok.json`

Text만 먼저:

```bash
python scripts/smoke_gemma.py --text-only
```

---

## 3. 디렉터리 구조 (추론 관련)

```text
pdf_vlm/
├── configs/models/gemma3_4b_qat.yaml
├── models/
│   ├── gemma-3-4b-it-q4_0.gguf          # text weights (QAT Q4_0)
│   └── mmproj-model-f16-4B.gguf         # vision projector
├── src/pdf_vlm/llm/gemma_llama_cpp.py   # Python wrapper
├── scripts/
│   ├── download_models.py
│   ├── smoke_gemma.py                   # text + vision tests
│   └── sample_gemma_infer.py
├── artifacts/smoke/
│   ├── sample_vision.png
│   ├── gemma_text_ok.json
│   ├── gemma_vision_ok.json
│   └── gemma_ok.json
└── docs/gemma_local_inference.md        # this file
```

---

## 4. Python wrapper 사용

```python
from pdf_vlm.llm.gemma_llama_cpp import Gemma3LlamaCpp

llm = Gemma3LlamaCpp.from_config()

# Text
r = llm.generate_text("Say OK in one word.", max_tokens=16, temperature=0.0)
print(r.text, r.latency_ms)

# Vision (image + question)
r = llm.generate_vision(
    "What do you see?",
    "artifacts/smoke/sample_vision.png",
    max_tokens=64,
)
print(r.text, r.latency_ms)
print("vision_supported:", llm.vision_supported)
```

주요 메서드:
- `generate_text(...)` → `GenerationResult` (`text`, `latency_ms`, RSS/VRAM)
- `generate_vision(prompt, image_path, ...)`
- `generate_multimodal(prompt, image_paths, ...)`
- `vision_supported` property

---

## 5. 실행 예시

```bash
# 샘플 호출
python scripts/sample_gemma_infer.py

# 사용자 이미지로 vision 테스트
python scripts/smoke_gemma.py --image path\to\photo.png --max-tokens 128
```

공식 CLI와 대응 관계:
```bash
# text
llama-cli -hf google/gemma-3-4b-it-qat-q4_0-gguf -p "Write a poem about the Kraken."

# vision
llama-gemma3-cli -hf google/gemma-3-4b-it-qat-q4_0-gguf -p "Describe this image." --image photo.png
```

---

## 6. 문제 해결 체크리스트

### A. `401/403` on download
- [ ] HF에서 Gemma license 수락했는가
- [ ] `huggingface-cli whoami`가 로그인 상태인가
- [ ] `HF_TOKEN`이 gated repo 권한이 있는가

### B. `ModuleNotFoundError: llama_cpp`
- [ ] `pip install llama-cpp-python` 실행했는가
- [ ] 올바른 Python/venv인가 (`where python`)

### C. Vision 실패: mmproj missing
- [ ] `models/mmproj-model-f16-4B.gguf` 존재?
- [ ] `python scripts/download_models.py --with-mmproj`

### D. Vision 실패: chat handler missing
- [ ] `from llama_cpp.llama_chat_format import MTMDChatHandler`
- [ ] Windows: prebuilt wheel index로 재설치
- [ ] `pip install -U llama-cpp-python` (Linux/macOS)

### E. CUDA 빌드 실패 (Windows)
- [ ] 먼저 CPU wheel로 text/vision 기능 검증
- [ ] 이후 CUDA toolkit 버전과 `CMAKE_ARGS=-DGGML_CUDA=on` 재설치
- [ ] `n_gpu_layers: 0`으로 CPU 강제 후 동작 확인

### F. 응답이 비거나 이상함
- [ ] `chat_format: gemma` / vision handler 충돌 여부 (vision ON이면 handler가 format 담당)
- [ ] `n_ctx`가 너무 작은지 (이미지 임베딩 포함 시 8192 권장)
- [ ] temperature=0.0으로 smoke 재실행

### G. 메모리 부족
- [ ] 다른 GPU 프로세스 종료
- [ ] `n_gpu_layers`를 20~30으로 낮추기 또는 `0`(CPU)
- [ ] `n_ctx`를 4096으로 낮추기

### H. “이미지는 받았지만 내용을 못 봄”
- [ ] data-URI(base64)로 전달되는지 확인 (wrapper가 자동 변환)
- [ ] mmproj가 **4B용**(`mmproj-model-f16-4B.gguf`)인지 확인 (1B/12B mmproj와 차원 mismatch)

---

## 7. 이 단계에서 하지 않는 것

- PaddleOCR / PDF parsing
- RAG retrieval / indexing

---

## 8. Inference practicality benchmark

배포 실용성(TTFT / e2e / tok/s / 메모리) 측정:

```bash
python scripts/bench_gemma_inference.py --mock          # no weights
python scripts/bench_gemma_inference.py --repeats 3      # real GGUF
```

자세한 설계: [`docs/inference_benchmark.md`](inference_benchmark.md)
