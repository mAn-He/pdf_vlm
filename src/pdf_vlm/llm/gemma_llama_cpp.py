"""Gemma 3 4B local inference via llama-cpp-python (llama.cpp).

Primary path (official):
  - Text GGUF: google/gemma-3-4b-it-qat-q4_0-gguf / gemma-3-4b-it-q4_0.gguf
  - Vision mmproj: same repo / mmproj-model-f16-4B.gguf
  - Python API: Llama + Gemma3ChatHandler (or MTMDChatHandler)

Refs:
  - https://ai.google.dev/gemma/docs/core/model_card_3
  - https://ai.google.dev/gemma/docs/integrations/llamacpp
  - https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf
"""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

from pdf_vlm.eval.system_metrics import SystemMonitor
from pdf_vlm.schemas import GenerationResult
from pdf_vlm.utils.io import load_named_config, resolve_path
from pdf_vlm.utils.logging import get_logger

logger = get_logger("llm.gemma")


def _image_to_data_uri(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _extract_stream_delta(chunk: Any) -> str:
    """Extract text delta from a streaming chat-completion chunk."""
    if not isinstance(chunk, dict):
        return ""
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    c0 = choices[0]
    delta = c0.get("delta") or {}
    if isinstance(delta, dict) and delta.get("content"):
        return str(delta["content"])
    # some builds put partial text here
    if c0.get("text"):
        return str(c0["text"])
    msg = c0.get("message") or {}
    if isinstance(msg, dict) and msg.get("content"):
        return str(msg["content"])
    return ""


def _estimate_completion_tokens(text: str, reported: int | None) -> int:
    if reported is not None and reported > 0:
        return int(reported)
    # rough fallback (~4 chars/token) when usage missing
    return max(1, len(text) // 4) if text else 0


def _throughput(completion_tokens: int, e2e_ms: float, ttft_ms: float | None) -> float | None:
    """Decode tokens/sec after first token when possible; else e2e tok/s."""
    if completion_tokens <= 0 or e2e_ms <= 0:
        return None
    if ttft_ms is not None and e2e_ms > ttft_ms and completion_tokens > 1:
        decode_ms = e2e_ms - ttft_ms
        if decode_ms > 0:
            return (completion_tokens - 1) / (decode_ms / 1000.0)
    return completion_tokens / (e2e_ms / 1000.0)


def _extract_text(out: Any) -> tuple[str, int | None, int | None]:
    text = ""
    prompt_tokens = None
    completion_tokens = None
    if isinstance(out, dict):
        choices = out.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            text = msg.get("content") or choices[0].get("text") or ""
        usage = out.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
    return (text or "").strip(), prompt_tokens, completion_tokens


def _resolve_vision_handler(mmproj_path: Path):
    """Return a multimodal chat handler for Gemma 3.

    llama-cpp-python >=0.3.x typically exposes MTMDChatHandler for Gemma 3 mmproj.
    Older forks may expose Gemma3ChatHandler.
    """
    errors: list[str] = []

    try:
        from llama_cpp.llama_chat_format import MTMDChatHandler

        logger.info("Using MTMDChatHandler with mmproj=%s", mmproj_path)
        return MTMDChatHandler(clip_model_path=str(mmproj_path))
    except Exception as exc:
        errors.append(f"MTMDChatHandler: {exc}")

    try:
        from llama_cpp.llama_chat_format import Gemma3ChatHandler

        logger.info("Using Gemma3ChatHandler with mmproj=%s", mmproj_path)
        return Gemma3ChatHandler(clip_model_path=str(mmproj_path))
    except Exception as exc:
        errors.append(f"Gemma3ChatHandler: {exc}")

    raise RuntimeError(
        "Vision requires llama-cpp-python with MTMDChatHandler (preferred) or Gemma3ChatHandler, "
        f"and mmproj at {mmproj_path}. "
        "Windows tip: pip install llama-cpp-python --extra-index-url "
        "https://abetlen.github.io/llama-cpp-python/whl/cpu "
        f"Errors: {' | '.join(errors)}"
    )


class Gemma3LlamaCpp:
    """Python wrapper for local Gemma 3 4B QAT GGUF inference (text + vision)."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        mmproj_path: str | Path | None = None,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        chat_format: str | None = "gemma",
        verbose: bool = False,
        enable_vision: bool = True,
    ):
        self.model_path = Path(model_path)
        self.mmproj_path = Path(mmproj_path) if mmproj_path else None
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.chat_format = chat_format
        self.verbose = verbose
        self.enable_vision = enable_vision and self.mmproj_path is not None
        self._llm = None
        self._vision_enabled = False

    @classmethod
    def from_config(cls, model_cfg: dict[str, Any] | None = None) -> "Gemma3LlamaCpp":
        cfg = model_cfg or load_named_config("models/gemma3_4b_qat.yaml")
        model_path = resolve_path(cfg["local_path"])
        mmproj = cfg.get("mmproj_local_path")
        mmproj_path = resolve_path(mmproj) if mmproj else None
        return cls(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=int(cfg.get("n_ctx", 8192)),
            n_gpu_layers=int(cfg.get("n_gpu_layers", -1)),
            n_batch=int(cfg.get("n_batch", 512)),
            chat_format=cfg.get("chat_format") or "gemma",
            verbose=bool(cfg.get("verbose", False)),
            enable_vision=bool(cfg.get("enable_vision", True)),
        )

    @property
    def vision_supported(self) -> bool:
        self._ensure_loaded()
        return self._vision_enabled

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required. See docs/gemma_local_inference.md"
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"GGUF not found: {self.model_path}. Run: python scripts/download_models.py"
            )

        kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_batch": self.n_batch,
            "verbose": self.verbose,
        }

        if self.enable_vision and self.mmproj_path is not None:
            if not self.mmproj_path.exists():
                raise FileNotFoundError(
                    f"mmproj not found: {self.mmproj_path}. "
                    "Vision needs mmproj-model-f16-4B.gguf from the same Google HF repo. "
                    "Run: python scripts/download_models.py --with-mmproj"
                )
            kwargs["chat_handler"] = _resolve_vision_handler(self.mmproj_path)
            # chat_handler owns formatting; avoid conflicting chat_format
            self._vision_enabled = True
            logger.info("Loading Gemma 3 with vision (mmproj=%s)", self.mmproj_path.name)
        else:
            if self.chat_format:
                kwargs["chat_format"] = self.chat_format
            self._vision_enabled = False
            logger.info("Loading Gemma 3 text-only from %s", self.model_path)

        self._llm = Llama(**kwargs)

    def unload(self) -> None:
        self._llm = None
        self._vision_enabled = False

    def _run_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        track_metrics: bool,
        stream: bool,
        meta: dict[str, Any],
    ) -> GenerationResult:
        """Shared chat path with optional streaming TTFT measurement."""
        assert self._llm is not None
        monitor = SystemMonitor(track_rss=track_metrics, track_vram=track_metrics)
        monitor.start()
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        text = ""
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        if stream:
            try:
                chunks = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                parts: list[str] = []
                last_usage: dict[str, Any] = {}
                for chunk in chunks:
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("usage"):
                        last_usage = chunk["usage"] or {}
                    delta = _extract_stream_delta(chunk)
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000.0
                        parts.append(delta)
                    monitor.sample()
                text = "".join(parts).strip()
                prompt_tokens = last_usage.get("prompt_tokens")
                completion_tokens = last_usage.get("completion_tokens")
            except TypeError:
                # Older builds may reject stream=True
                logger.warning("Streaming unsupported; falling back to non-stream completion")
                stream = False

        if not stream:
            out = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency_partial = (time.perf_counter() - t0) * 1000.0
            text, prompt_tokens, completion_tokens = _extract_text(out)
            # Without streaming, TTFT is undefined; approximate as e2e (upper bound)
            ttft_ms = latency_partial if ttft_ms is None else ttft_ms
            meta = {**meta, "ttft_approx": True}

        e2e_ms = (time.perf_counter() - t0) * 1000.0
        snap = monitor.stop()
        n_comp = _estimate_completion_tokens(text, completion_tokens)
        tps = _throughput(n_comp, e2e_ms, ttft_ms if not meta.get("ttft_approx") else None)

        return GenerationResult(
            text=text,
            latency_ms=e2e_ms,
            ttft_ms=ttft_ms,
            tokens_per_sec=tps,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens if completion_tokens is not None else n_comp,
            peak_rss_mb=snap.get("peak_rss_mb"),
            peak_vram_mb=snap.get("peak_vram_mb"),
            meta={**meta, "streamed": stream and not meta.get("ttft_approx")},
        )

    def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.1,
        system: str | None = None,
        track_metrics: bool = True,
        stream: bool = True,
    ) -> GenerationResult:
        """Text-only generation with wall-clock latency (+ TTFT when stream=True)."""
        self._ensure_loaded()
        assert self._llm is not None

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self._run_chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            track_metrics=track_metrics,
            stream=stream,
            meta={"modality": "text", "vision_enabled": self._vision_enabled},
        )

    def generate_vision(
        self,
        prompt: str,
        image_path: str | Path,
        *,
        max_tokens: int = 256,
        temperature: float = 0.1,
        system: str | None = None,
        track_metrics: bool = True,
        stream: bool = True,
    ) -> GenerationResult:
        """Single-image + text generation (Gemma 3 multimodal)."""
        return self.generate_multimodal(
            prompt,
            [str(image_path)],
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            track_metrics=track_metrics,
            stream=stream,
        )

    def generate_multimodal(
        self,
        prompt: str,
        image_paths: list[str],
        *,
        max_tokens: int = 256,
        temperature: float = 0.1,
        system: str | None = None,
        track_metrics: bool = True,
        stream: bool = True,
    ) -> GenerationResult:
        """Image(+s) + text generation. Requires mmproj and a vision-capable llama-cpp-python."""
        if not image_paths:
            return self.generate_text(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                track_metrics=track_metrics,
                stream=stream,
            )

        self._ensure_loaded()
        assert self._llm is not None

        if not self._vision_enabled:
            raise RuntimeError(
                "Vision is not enabled. Set mmproj_local_path to "
                "models/mmproj-model-f16-4B.gguf and reload. "
                "Official CLI equivalent: llama-gemma3-cli -hf google/gemma-3-4b-it-qat-q4_0-gguf "
                "--image <path> -p '<question>'"
            )

        content: list[dict[str, Any]] = []
        for path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_uri(path)},
                }
            )
        content.append({"type": "text", "text": prompt})

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        return self._run_chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            track_metrics=track_metrics,
            stream=stream,
            meta={
                "modality": "multimodal",
                "vision_enabled": True,
                "n_images": len(image_paths),
                "image_paths": list(image_paths),
            },
        )


# Backward-compatible alias used by RAG pipeline
GemmaLlamaCpp = Gemma3LlamaCpp


def build_llm(model_cfg: dict[str, Any] | None = None) -> Gemma3LlamaCpp:
    return Gemma3LlamaCpp.from_config(model_cfg)
