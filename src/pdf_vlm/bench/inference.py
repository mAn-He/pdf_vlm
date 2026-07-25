"""Gemma 3 4B local quantized inference practicality benchmark.

Focus: deployment practicality (TTFT, e2e latency, tok/s, memory),
NOT answer quality.
"""

from __future__ import annotations

import csv
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from pdf_vlm.eval.system_metrics import get_rss_mb, get_vram_mb
from pdf_vlm.utils.io import ensure_dir, save_json
from pdf_vlm.utils.logging import get_logger

logger = get_logger("bench.inference")


@dataclass
class BenchCase:
    case_id: str
    modality: str  # text | multimodal
    top_k: int
    page_bucket: int
    chars_per_page: int = 800
    n_images: int = 0
    question: str = "What is the key fact in the evidence?"
    max_tokens: int = 64
    warmup: bool = False


@dataclass
class BenchRow:
    run_id: str
    case_id: str
    modality: str
    top_k: int
    page_bucket: int
    n_images: int
    prompt_chars: int
    repeat: int
    ttft_ms: float | None
    e2e_ms: float
    tokens_per_sec: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    peak_rss_mb: float | None
    peak_vram_mb: float | None
    load_rss_mb: float | None = None
    answer_preview: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_flat(self) -> dict[str, Any]:
        d = asdict(self)
        d["meta"] = ""
        return d


class TimedLLM(Protocol):
    def generate_text(self, prompt: str, *, max_tokens: int = 64, temperature: float = 0.0, stream: bool = True, **kwargs: Any): ...
    def generate_multimodal(
        self,
        prompt: str,
        image_paths: list[str],
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        stream: bool = True,
        **kwargs: Any,
    ): ...

    @property
    def vision_supported(self) -> bool: ...


def build_evidence_prompt(
    *,
    question: str,
    top_k: int,
    page_bucket: int,
    chars_per_page: int = 800,
) -> str:
    """Simulate RAG context: top-k evidence pages drawn from a longer document."""
    pages = []
    for i in range(top_k):
        page_id = i % max(page_bucket, 1)
        body = (
            f"[Page {page_id} of {page_bucket}] "
            f"Acme Corp section {page_id}. Founded in 1998. Product VisionX-4. "
            f"Revenue context for retrieval unit {i}. "
        )
        # pad to approximate OCR page length
        pad = max(0, chars_per_page - len(body))
        body = body + ("lorem " * ((pad // 6) + 1))[:pad]
        pages.append(body)
    evidence = "\n\n".join(pages)
    return (
        "You are a document QA assistant. Use ONLY the evidence below.\n\n"
        f"=== EVIDENCE (top-{top_k} of {page_bucket}-page doc) ===\n"
        f"{evidence}\n"
        f"=== QUESTION ===\n{question}\n"
        "Answer briefly."
    )


def make_synthetic_page_images(out_dir: Path, n: int, size: tuple[int, int] = (512, 384)) -> list[str]:
    """Create simple page images for multimodal top-k sweeps (no PDF required)."""
    from PIL import Image, ImageDraw

    ensure_dir(out_dir)
    paths: list[str] = []
    for i in range(n):
        path = out_dir / f"bench_page_{i}.png"
        if not path.exists():
            img = Image.new("RGB", size, color=(245, 245, 240))
            draw = ImageDraw.Draw(img)
            draw.rectangle([20, 20, size[0] - 20, 60], fill=(40, 80, 140))
            draw.text((30, 30), f"PAGE {i}", fill=(255, 255, 255))
            draw.text((30, 100), "Acme Corp founded 1998", fill=(20, 20, 20))
            draw.text((30, 140), "Flagship: VisionX-4", fill=(20, 20, 20))
            img.save(path)
        paths.append(str(path))
    return paths


def default_cases(
    *,
    top_ks: list[int] | None = None,
    page_buckets: list[int] | None = None,
    modalities: list[str] | None = None,
    chars_per_page: int = 800,
    max_tokens: int = 64,
) -> list[BenchCase]:
    top_ks = top_ks or [1, 3, 5]
    page_buckets = page_buckets or [5, 20, 50, 100]
    modalities = modalities or ["text", "multimodal"]
    cases: list[BenchCase] = []
    for modality in modalities:
        for bucket in page_buckets:
            for k in top_ks:
                if k > bucket:
                    continue
                cases.append(
                    BenchCase(
                        case_id=f"{modality}_bucket{bucket}_k{k}",
                        modality=modality,
                        top_k=k,
                        page_bucket=bucket,
                        chars_per_page=chars_per_page,
                        n_images=k if modality == "multimodal" else 0,
                        max_tokens=max_tokens,
                    )
                )
    return cases


def host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "rss_mb_now": round(get_rss_mb(), 2),
        "vram_mb_now": get_vram_mb(),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / (1024**3), 2)
        info["ram_available_gb"] = round(vm.available / (1024**3), 2)
    except Exception:
        pass
    return info


def run_case(
    llm: TimedLLM,
    case: BenchCase,
    *,
    run_id: str,
    image_paths: list[str] | None,
    repeat: int,
    load_rss_mb: float | None,
    temperature: float = 0.0,
) -> BenchRow:
    prompt = build_evidence_prompt(
        question=case.question,
        top_k=case.top_k,
        page_bucket=case.page_bucket,
        chars_per_page=case.chars_per_page,
    )
    if case.modality == "text":
        res = llm.generate_text(
            prompt,
            max_tokens=case.max_tokens,
            temperature=temperature,
            stream=True,
        )
        n_images = 0
    else:
        imgs = (image_paths or [])[: case.top_k]
        if not imgs:
            raise RuntimeError("multimodal case requires image_paths")
        if not getattr(llm, "vision_supported", True):
            raise RuntimeError("LLM does not support vision")
        res = llm.generate_multimodal(
            prompt,
            imgs,
            max_tokens=case.max_tokens,
            temperature=temperature,
            stream=True,
        )
        n_images = len(imgs)

    return BenchRow(
        run_id=run_id,
        case_id=case.case_id,
        modality=case.modality,
        top_k=case.top_k,
        page_bucket=case.page_bucket,
        n_images=n_images,
        prompt_chars=len(prompt),
        repeat=repeat,
        ttft_ms=res.ttft_ms,
        e2e_ms=res.latency_ms,
        tokens_per_sec=res.tokens_per_sec,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        peak_rss_mb=res.peak_rss_mb,
        peak_vram_mb=res.peak_vram_mb,
        load_rss_mb=load_rss_mb,
        answer_preview=(res.text or "")[:160],
        meta=dict(res.meta or {}),
    )


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate_rows(rows: list[BenchRow]) -> list[dict[str, Any]]:
    """Mean metrics per (modality, page_bucket, top_k)."""
    groups: dict[tuple, list[BenchRow]] = {}
    for r in rows:
        key = (r.modality, r.page_bucket, r.top_k)
        groups.setdefault(key, []).append(r)

    out: list[dict[str, Any]] = []
    for (modality, bucket, k), grp in sorted(groups.items()):
        ttfts = [float(r.ttft_ms) for r in grp if r.ttft_ms is not None]
        e2es = [float(r.e2e_ms) for r in grp]
        tps = [float(r.tokens_per_sec) for r in grp if r.tokens_per_sec is not None]
        rss = [float(r.peak_rss_mb) for r in grp if r.peak_rss_mb is not None]
        vram = [float(r.peak_vram_mb) for r in grp if r.peak_vram_mb is not None]
        out.append(
            {
                "modality": modality,
                "page_bucket": bucket,
                "top_k": k,
                "n": len(grp),
                "prompt_chars_mean": mean([float(r.prompt_chars) for r in grp]),
                "ttft_ms_mean": mean(ttfts) if ttfts else None,
                "e2e_ms_mean": mean(e2es),
                "tokens_per_sec_mean": mean(tps) if tps else None,
                "peak_rss_mb_max": max(rss) if rss else None,
                "peak_vram_mb_max": max(vram) if vram else None,
                "n_images": grp[0].n_images,
            }
        )
    return out


def interpret_practicality(
    aggregates: list[dict[str, Any]],
    *,
    host: dict[str, Any],
    model_name: str = "gemma-3-4b-it-qat-q4_0",
) -> dict[str, Any]:
    """Rule-of-thumb practicality summary for local laptop/desktop use."""
    text_rows = [a for a in aggregates if a["modality"] == "text"]
    mm_rows = [a for a in aggregates if a["modality"] == "multimodal"]

    def _best(rows: list[dict[str, Any]], key: str) -> float | None:
        vals = [a[key] for a in rows if a.get(key) is not None]
        return min(vals) if vals else None

    def _worst(rows: list[dict[str, Any]], key: str) -> float | None:
        vals = [a[key] for a in rows if a.get(key) is not None]
        return max(vals) if vals else None

    text_ttft = _best(text_rows, "ttft_ms_mean")
    text_e2e_k3 = next((a["e2e_ms_mean"] for a in text_rows if a["top_k"] == 3), None)
    mm_e2e_k3 = next((a["e2e_ms_mean"] for a in mm_rows if a["top_k"] == 3), None)
    text_tps = next(
        (a["tokens_per_sec_mean"] for a in text_rows if a.get("tokens_per_sec_mean") is not None),
        None,
    )
    if text_tps is None and text_rows:
        tps_vals = [a["tokens_per_sec_mean"] for a in text_rows if a.get("tokens_per_sec_mean") is not None]
        text_tps = mean(tps_vals) if tps_vals else None

    rss_max = _worst(aggregates, "peak_rss_mb_max")
    vram_max = _worst(aggregates, "peak_vram_mb_max")

    # Thresholds (practical local QA, not SOTA chat)
    has_gpu = vram_max is not None and vram_max > 100
    verdicts: list[str] = []

    if text_tps is not None:
        if text_tps >= 20:
            verdicts.append(f"Text decode ~{text_tps:.1f} tok/s → snappy interactive QA.")
        elif text_tps >= 8:
            verdicts.append(f"Text decode ~{text_tps:.1f} tok/s → usable for short answers (64–128 tokens).")
        elif text_tps >= 3:
            verdicts.append(f"Text decode ~{text_tps:.1f} tok/s → acceptable for offline batch / demos.")
        else:
            verdicts.append(f"Text decode ~{text_tps:.1f} tok/s → slow; prefer GPU build or smaller max_tokens.")

    if text_ttft is not None:
        if text_ttft < 500:
            verdicts.append(f"Best text TTFT {text_ttft:.0f} ms → feels interactive.")
        elif text_ttft < 2000:
            verdicts.append(f"Best text TTFT {text_ttft:.0f} ms → OK for research prototypes.")
        else:
            verdicts.append(f"Best text TTFT {text_ttft:.0f} ms → noticeable wait; keep prompts short.")

    if text_e2e_k3 is not None and mm_e2e_k3 is not None and text_e2e_k3 > 0:
        overhead = mm_e2e_k3 / text_e2e_k3
        verdicts.append(
            f"Multimodal top-3 e2e is ~{overhead:.2f}× text "
            f"({mm_e2e_k3:.0f} ms vs {text_e2e_k3:.0f} ms) — expect vision cost on each call."
        )

    # top-k sensitivity
    text_by_k = sorted([a for a in text_rows if a["page_bucket"] == 5], key=lambda x: x["top_k"])
    if len(text_by_k) >= 2:
        slowdown = text_by_k[-1]["e2e_ms_mean"] / max(text_by_k[0]["e2e_ms_mean"], 1e-6)
        verdicts.append(
            f"On 5-page bucket, raising top-k {text_by_k[0]['top_k']}→{text_by_k[-1]['top_k']} "
            f"changes text e2e by ~{slowdown:.2f}× (prefill grows with evidence)."
        )

    if rss_max is not None:
        if rss_max < 6000:
            verdicts.append(f"Peak RSS ~{rss_max:.0f} MB fits typical 16GB laptops alongside OCR/index.")
        elif rss_max < 10000:
            verdicts.append(f"Peak RSS ~{rss_max:.0f} MB — workable on 16–32GB desktops; close other apps.")
        else:
            verdicts.append(f"Peak RSS ~{rss_max:.0f} MB — heavy; reduce n_ctx or avoid concurrent heavy jobs.")

    if has_gpu:
        verdicts.append(f"GPU VRAM peak ~{vram_max:.0f} MB observed — prefer CUDA/Metal llama-cpp builds.")
    else:
        verdicts.append("No VRAM reported (CPU build or pynvml missing) — CPU path is valid but slower.")

    # Overall label
    score = "limited"
    if text_tps is not None and text_tps >= 8 and (text_e2e_k3 is None or text_e2e_k3 < 8000):
        score = "practical_local_qa"
    if text_tps is not None and text_tps >= 20 and (text_ttft is None or text_ttft < 800):
        score = "interactive_local"

    recommendation = (
        f"Keep retrieval top-k small (1–3), max_tokens≤128 for interactive use, "
        f"and use {model_name} Q4_0 so text+vision share a ~3–4GB weight budget on a local machine."
    )

    return {
        "model": model_name,
        "host": host,
        "backend_hint": "GPU" if has_gpu else "CPU_or_unknown",
        "practicality_score": score,
        "highlights": verdicts,
        "recommendation": recommendation,
        "thresholds": {
            "interactive_tok_s": 20,
            "usable_tok_s": 8,
            "batch_tok_s": 3,
            "interactive_ttft_ms": 500,
            "prototype_ttft_ms": 2000,
            "comfortable_e2e_ms_short_answer": 8000,
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.2f}"
    return str(v)


def markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return "_No data._\n"
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def build_markdown_report(
    *,
    run_id: str,
    host: dict[str, Any],
    aggregates: list[dict[str, Any]],
    practicality: dict[str, Any],
    config: dict[str, Any],
) -> str:
    cols = [
        "modality",
        "page_bucket",
        "top_k",
        "ttft_ms_mean",
        "e2e_ms_mean",
        "tokens_per_sec_mean",
        "prompt_chars_mean",
        "peak_rss_mb_max",
        "peak_vram_mb_max",
    ]
    lines = [
        f"# Gemma Local Inference Benchmark: `{run_id}`",
        "",
        "## Goal",
        "",
        "Measure **deployment practicality** of Gemma 3 4B QAT Q4_0 on a local machine "
        "(not answer quality).",
        "",
        "## Environment",
        "",
        f"- Platform: `{host.get('platform')}`",
        f"- Python: `{host.get('python')}`",
        f"- RAM total/available GB: `{host.get('ram_total_gb')}` / `{host.get('ram_available_gb')}`",
        f"- Backend hint: `{practicality.get('backend_hint')}`",
        f"- Seed/config: `{config.get('seed')}`, max_tokens=`{config.get('max_tokens')}`, "
        f"repeats=`{config.get('repeats')}`",
        "",
        "## Aggregate results",
        "",
        markdown_table(aggregates, cols),
        "",
        "## Practicality summary",
        "",
        f"**Score:** `{practicality.get('practicality_score')}`",
        "",
    ]
    for h in practicality.get("highlights") or []:
        lines.append(f"- {h}")
    lines += [
        "",
        f"**Recommendation:** {practicality.get('recommendation')}",
        "",
        "## Interpretation guide",
        "",
        "| Metric | How to read |",
        "| --- | --- |",
        "| TTFT | Time to first token (streaming). <500 ms feels interactive; <2 s OK for prototypes. |",
        "| e2e_ms | Full answer latency. Keep short answers (≤128 tokens) for local UX. |",
        "| tokens/sec | Decode throughput after first token. ≥20 interactive; ≥8 usable; ≥3 batch-only. |",
        "| top_k × page_bucket | Prefill cost grows with evidence size; RAG should retrieve few pages even from 100-page PDFs. |",
        "| multimodal vs text | Extra cost from mmproj/image tokens — only attach top-k page images, never all pages. |",
        "| RSS / VRAM | Q4_0 keeps weights ~3GB so OCR + embeddings can coexist on a laptop. |",
        "",
        "---",
        "",
        "_Generated by `pdf_vlm.bench.inference`._",
        "",
    ]
    return "\n".join(lines)


class MockTimedLLM:
    """Deterministic fake LLM for CI / no-weights dry runs."""

    def __init__(self, base_ms: float = 40.0, ms_per_char: float = 0.002, ms_per_image: float = 25.0):
        self.base_ms = base_ms
        self.ms_per_char = ms_per_char
        self.ms_per_image = ms_per_image
        self.vision_supported = True

    def _fake(self, prompt: str, n_images: int, max_tokens: int):
        from pdf_vlm.schemas import GenerationResult

        prefill = self.base_ms + len(prompt) * self.ms_per_char + n_images * self.ms_per_image
        # simulate decode
        decode = max_tokens * 2.0  # 2 ms/token → 500 tok/s mock is too fast; use 8ms → ~125? use 5ms
        decode = max_tokens * 5.0
        time.sleep(min(0.05, (prefill + decode) / 1000.0))  # tiny real sleep
        ttft = prefill
        e2e = prefill + decode
        return GenerationResult(
            text="1998",
            latency_ms=e2e,
            ttft_ms=ttft,
            tokens_per_sec=(max_tokens / (decode / 1000.0)) if decode else None,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max_tokens,
            peak_rss_mb=get_rss_mb(),
            peak_vram_mb=get_vram_mb(),
            meta={"modality": "text" if n_images == 0 else "multimodal", "mock": True, "streamed": True},
        )

    def generate_text(self, prompt: str, *, max_tokens: int = 64, temperature: float = 0.0, stream: bool = True, **kwargs: Any):
        return self._fake(prompt, 0, max_tokens)

    def generate_multimodal(
        self,
        prompt: str,
        image_paths: list[str],
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        stream: bool = True,
        **kwargs: Any,
    ):
        return self._fake(prompt, len(image_paths), max_tokens)


def run_inference_benchmark(
    *,
    llm: TimedLLM | None = None,
    cases: list[BenchCase] | None = None,
    repeats: int = 2,
    warmup: int = 1,
    out_dir: str | Path | None = None,
    image_dir: str | Path | None = None,
    mock: bool = False,
    max_tokens: int = 64,
    seed: int = 42,
    model_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run full bench and write CSV/JSON/Markdown."""
    from pdf_vlm.eval.seed import set_global_seed

    seed_snap = set_global_seed(seed)
    run_id = f"gemma_infer_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base = Path(out_dir) if out_dir else Path("results/bench")
    out = ensure_dir(base / run_id)
    host = host_info()
    cases = cases or default_cases(max_tokens=max_tokens)

    load_rss = None
    if mock:
        llm = MockTimedLLM()
        load_rss = get_rss_mb()
    elif llm is None:
        from pdf_vlm.llm.gemma_llama_cpp import Gemma3LlamaCpp

        llm = Gemma3LlamaCpp.from_config(model_cfg)
        # Force load to capture baseline RSS
        t_load = time.perf_counter()
        _ = llm.generate_text("Say OK", max_tokens=4, temperature=0.0, stream=True)
        load_ms = (time.perf_counter() - t_load) * 1000.0
        load_rss = get_rss_mb()
        logger.info("Model ready (first-call includes load) first_call_ms=%.1f rss=%.1f", load_ms, load_rss or 0)

    img_dir = ensure_dir(Path(image_dir) if image_dir else out / "images")
    max_images_needed = max((c.n_images for c in cases), default=0)
    images = make_synthetic_page_images(img_dir, max(max_images_needed, 1))

    # Warmup on smallest text case
    warm_case = next((c for c in cases if c.modality == "text"), cases[0])
    for i in range(max(0, warmup)):
        logger.info("warmup %d/%d", i + 1, warmup)
        try:
            run_case(llm, warm_case, run_id=run_id, image_paths=images, repeat=-1 - i, load_rss_mb=load_rss)
        except Exception as e:
            logger.warning("warmup failed: %s", e)

    rows: list[BenchRow] = []
    for case in cases:
        for r in range(repeats):
            try:
                row = run_case(
                    llm,
                    case,
                    run_id=run_id,
                    image_paths=images,
                    repeat=r,
                    load_rss_mb=load_rss,
                )
                rows.append(row)
                logger.info(
                    "%s rep=%d ttft=%.1f e2e=%.1f tok/s=%s",
                    case.case_id,
                    r,
                    row.ttft_ms or -1,
                    row.e2e_ms,
                    f"{row.tokens_per_sec:.1f}" if row.tokens_per_sec else "—",
                )
            except Exception as e:
                logger.warning("Skip %s rep=%d: %s", case.case_id, r, e)

    aggregates = aggregate_rows(rows)
    config = {
        "seed": seed,
        "repeats": repeats,
        "warmup": warmup,
        "max_tokens": max_tokens,
        "mock": mock or bool((rows[0].meta.get("mock") if rows else False)),
        "n_cases": len(cases),
        "n_rows": len(rows),
    }
    practicality = interpret_practicality(aggregates, host=host)

    write_csv(out / "raw_results.csv", [r.to_flat() for r in rows])
    write_csv(out / "aggregate.csv", aggregates)
    save_json(out / "raw_results.json", [asdict(r) for r in rows])
    save_json(out / "aggregate.json", aggregates)
    save_json(out / "host_info.json", host)
    save_json(out / "config_snapshot.json", {**config, "seed_snapshot": seed_snap})
    save_json(out / "practicality.json", practicality)

    md = build_markdown_report(
        run_id=run_id,
        host=host,
        aggregates=aggregates,
        practicality=practicality,
        config=config,
    )
    (out / "report.md").write_text(md, encoding="utf-8")
    report_dir = ensure_dir(Path("results/reports"))
    report_copy = report_dir / f"{run_id}.md"
    report_copy.write_text(md, encoding="utf-8")

    result = {
        "run_id": run_id,
        "out_dir": str(out),
        "report_md": str(out / "report.md"),
        "n_rows": len(rows),
        "aggregates": aggregates,
        "practicality": practicality,
        "host": host,
    }
    save_json(out / "summary.json", {k: v for k, v in result.items() if k not in {"aggregates"}})
    logger.info("Benchmark complete -> %s (score=%s)", out, practicality.get("practicality_score"))
    return result
