"""Multimodal RAG pipeline: text retrieval -> top-k page images + OCR -> Gemma 3.

Design (minimum viable, fair vs text-only):
  1) Retrieve with the SAME text index / retriever as text-only RAG
  2) Keep only top-k unique pages (never all document pages)
  3) Load those page images + OCR text
  4) Multimodal generate with Gemma 3 (text+image in, text out)

Interface mirrors TextOnlyRAGPipeline:
  retrieve() / answer(QAExample) / run(examples)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pdf_vlm.llm.gemma_llama_cpp import Gemma3LlamaCpp, GemmaLlamaCpp, build_llm
from pdf_vlm.ocr.paddle_structure import load_artifact
from pdf_vlm.rag.page_images import resolve_multimodal_pages
from pdf_vlm.rag.prompt_builder import SYSTEM_PROMPT_MULTIMODAL, build_multimodal_prompt
from pdf_vlm.rag.text_only import build_text_retriever
from pdf_vlm.retrieve.base import unique_page_ids
from pdf_vlm.schemas import DocumentArtifact, QAExample, QAPrediction, RetrievalHit
from pdf_vlm.utils.io import ensure_dir, load_named_config, resolve_path, save_json
from pdf_vlm.utils.logging import get_logger

logger = get_logger("rag.multimodal")

RetrievalMode = Literal["page", "section", "hierarchical"]


class MultimodalRAGPipeline:
    """Text-retrieve then multimodal-reason over top-k page images + OCR."""

    modality: str = "multimodal"

    def __init__(
        self,
        llm: Gemma3LlamaCpp | GemmaLlamaCpp | None,
        retriever,
        artifact: DocumentArtifact,
        *,
        retrieval_mode: RetrievalMode | str = "page",
        top_k: int = 3,
        max_images: int | None = None,
        generation_cfg: dict[str, Any] | None = None,
        max_prompt_chars: int = 10000,
        dry_run: bool = False,
    ):
        self.llm = llm
        self.retriever = retriever
        self.artifact = artifact
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.max_images = max_images if max_images is not None else top_k
        self.generation_cfg = generation_cfg or {"max_tokens": 256, "temperature": 0.1}
        self.max_prompt_chars = max_prompt_chars
        self.dry_run = dry_run or llm is None

    @classmethod
    def from_paths(
        cls,
        index_dir: str | Path,
        doc_id: str,
        *,
        retrieval_mode: RetrievalMode | str = "page",
        top_k: int = 3,
        max_images: int | None = None,
        device: str = "cpu",
        dry_run: bool = False,
        model_cfg: dict[str, Any] | None = None,
        generation_cfg: dict[str, Any] | None = None,
        processed_dir: str | Path | None = None,
        coarse_k: int = 5,
    ) -> "MultimodalRAGPipeline":
        artifact = load_artifact(doc_id, processed_dir=processed_dir)
        retriever = build_text_retriever(
            index_dir, retrieval=retrieval_mode, device=device, coarse_k=coarse_k
        )
        llm = None
        if not dry_run:
            cfg = model_cfg or load_named_config("models/gemma3_4b_qat.yaml")
            cfg = dict(cfg)
            cfg["local_path"] = str(resolve_path(cfg["local_path"]))
            if cfg.get("mmproj_local_path"):
                cfg["mmproj_local_path"] = str(resolve_path(cfg["mmproj_local_path"]))
            cfg["enable_vision"] = True
            llm = build_llm(cfg)
        return cls(
            llm,
            retriever,
            artifact,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            max_images=max_images,
            generation_cfg=generation_cfg,
            dry_run=dry_run,
        )

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalHit]:
        """Same text retrieval as text-only baseline (no images in retriever)."""
        k = self.top_k if top_k is None else top_k
        hits = self.retriever.retrieve(question, top_k=k)
        for h in hits:
            h.meta = {
                **(h.meta or {}),
                "modality": "multimodal",
                "retrieval": self.retrieval_mode,
                "stage": "text_retrieval",
            }
        return hits

    def answer(self, example: QAExample) -> QAPrediction:
        t0 = time.perf_counter()

        # Stage 1: text retrieval (shared with text-only)
        hits = self.retrieve(example.question, top_k=self.top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        retrieved_pages = unique_page_ids(hits)

        # Stage 2: load ONLY top-k page images + OCR
        mm_pages, image_load_ms = resolve_multimodal_pages(
            self.artifact, hits, max_images=self.max_images
        )
        prompt, image_paths = build_multimodal_prompt(
            example.question, mm_pages, max_chars=self.max_prompt_chars
        )

        if self.dry_run:
            gen_text = ""
            generation_ms = 0.0
            peak_rss = None
            peak_vram = None
            prompt_tokens = None
            completion_tokens = None
            vision_fallback = False
        else:
            assert self.llm is not None
            t1 = time.perf_counter()
            if image_paths:
                gen = self.llm.generate_multimodal(
                    prompt,
                    image_paths,
                    max_tokens=int(self.generation_cfg.get("max_tokens", 256)),
                    temperature=float(self.generation_cfg.get("temperature", 0.1)),
                    system=SYSTEM_PROMPT_MULTIMODAL,
                )
            else:
                # No images available: still call text path but keep multimodal logs
                logger.warning("No usable page images; falling back to text generation")
                gen = self.llm.generate_text(
                    prompt,
                    max_tokens=int(self.generation_cfg.get("max_tokens", 256)),
                    temperature=float(self.generation_cfg.get("temperature", 0.1)),
                    system=SYSTEM_PROMPT_MULTIMODAL,
                )
            generation_ms = gen.latency_ms or ((time.perf_counter() - t1) * 1000.0)
            gen_text = gen.text
            peak_rss = gen.peak_rss_mb
            peak_vram = gen.peak_vram_mb
            prompt_tokens = gen.prompt_tokens
            completion_tokens = gen.completion_tokens
            vision_fallback = bool((gen.meta or {}).get("vision_fallback"))

        e2e_ms = (time.perf_counter() - t0) * 1000.0
        used_images = list(image_paths)

        pred = QAPrediction(
            example_id=example.example_id,
            doc_id=example.doc_id or self.artifact.doc_id,
            question=example.question,
            prediction=gen_text,
            gold_answers=list(example.answers),
            retrieved_page_ids=retrieved_pages,
            evidence_pages=list(example.evidence_pages),
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
            e2e_latency_ms=e2e_ms,
            peak_rss_mb=peak_rss,
            peak_vram_mb=peak_vram,
            hits=hits,
            meta={
                "baseline": "multimodal_rag",
                "modality": "multimodal",
                "retrieval_mode": self.retrieval_mode,
                "top_k": self.top_k,
                "max_images": self.max_images,
                "dry_run": self.dry_run,
                "prompt": prompt,
                "used_images": used_images,
                "mm_pages": [
                    {
                        "page_id": p["page_id"],
                        "score": p["score"],
                        "exists": p["exists"],
                        "image_path": p["image_path"],
                    }
                    for p in mm_pages
                ],
                "latency_breakdown_ms": {
                    "retrieval": retrieval_ms,
                    "image_load": image_load_ms,
                    "generation": generation_ms,
                    "e2e": e2e_ms,
                },
                "image_load_latency_ms": image_load_ms,
                "n_images_used": len(used_images),
                "vision_fallback": vision_fallback if not self.dry_run else False,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "unanswerable": example.unanswerable,
                **{k: v for k, v in example.meta.items() if k not in {"prompt", "used_images"}},
            },
        )
        logger.info(
            "qa_id=%s retrieved=%s used_images=%d image_load_ms=%.1f gen_ms=%.1f e2e_ms=%.1f answer=%r",
            example.example_id,
            retrieved_pages,
            len(used_images),
            image_load_ms,
            generation_ms,
            e2e_ms,
            (gen_text[:120] if gen_text else ""),
        )
        return pred

    def run(
        self,
        examples: list[QAExample],
        *,
        out_dir: str | Path | None = None,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        preds = [self.answer(ex) for ex in examples]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = run_name or f"mm_rag_{self.retrieval_mode}_{stamp}"
        out = ensure_dir(Path(out_dir) if out_dir else resolve_path(f"results/runs/{run_id}"))

        records = [multimodal_log_record(p) for p in preds]
        save_json(out / "predictions.json", [p.model_dump(mode="json") for p in preds])
        save_json(out / "rag_log.json", records)
        summary = {
            "run_id": run_id,
            "baseline": "multimodal_rag",
            "modality": "multimodal",
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k,
            "max_images": self.max_images,
            "n": len(preds),
            "latency_ms_mean": sum(p.e2e_latency_ms for p in preds) / max(len(preds), 1),
            "retrieval_latency_ms_mean": sum(p.retrieval_latency_ms for p in preds) / max(len(preds), 1),
            "image_load_latency_ms_mean": sum(
                float(p.meta.get("image_load_latency_ms") or 0.0) for p in preds
            )
            / max(len(preds), 1),
            "generation_latency_ms_mean": sum(p.generation_latency_ms for p in preds) / max(len(preds), 1),
            "images_used_mean": sum(int(p.meta.get("n_images_used") or 0) for p in preds)
            / max(len(preds), 1),
        }
        save_json(out / "run_summary.json", summary)
        logger.info("Wrote multimodal RAG logs to %s", out)
        return {"out_dir": str(out), "summary": summary, "predictions": preds}


def multimodal_log_record(pred: QAPrediction) -> dict[str, Any]:
    """Eval/log schema for multimodal RAG."""
    return {
        "example_id": pred.example_id,
        "doc_id": pred.doc_id,
        "question": pred.question,
        "final_answer": pred.prediction,
        "answer": pred.prediction,
        "gold_answers": pred.gold_answers,
        "retrieved_pages": pred.retrieved_page_ids,
        "used_images": list(pred.meta.get("used_images") or []),
        "mm_pages": pred.meta.get("mm_pages") or [],
        "evidence_pages": pred.evidence_pages,
        "latency": {
            "retrieval_ms": pred.retrieval_latency_ms,
            "image_load_ms": pred.meta.get("image_load_latency_ms"),
            "generation_ms": pred.generation_latency_ms,
            "e2e_ms": pred.e2e_latency_ms,
            "breakdown_ms": pred.meta.get("latency_breakdown_ms"),
        },
        "retrieval_hits": [
            {
                "chunk_id": h.chunk_id,
                "score": h.score,
                "level": h.level,
                "page_ids": h.page_ids,
                "section_id": h.section_id,
                "text_preview": (h.text or "")[:240],
            }
            for h in pred.hits
        ],
        "meta": {
            "baseline": "multimodal_rag",
            "modality": "multimodal",
            "retrieval_mode": pred.meta.get("retrieval_mode"),
            "n_images_used": pred.meta.get("n_images_used"),
            "peak_rss_mb": pred.peak_rss_mb,
            "peak_vram_mb": pred.peak_vram_mb,
            "vision_fallback": pred.meta.get("vision_fallback"),
        },
    }
