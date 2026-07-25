"""Text-only RAG baseline pipeline (OCR text -> retrieve -> Gemma 3).

Interface mirrors future multimodal RAG:
  - same Retriever.retrieve(query, top_k)
  - same QAExample in / QAPrediction out
  - modality flag fixed to "text"
  - never sends images to the LLM
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pdf_vlm.llm.gemma_llama_cpp import Gemma3LlamaCpp, GemmaLlamaCpp, build_llm
from pdf_vlm.ocr.paddle_structure import load_artifact
from pdf_vlm.rag.prompt_builder import SYSTEM_PROMPT_TEXT_ONLY, build_text_prompt
from pdf_vlm.retrieve.base import unique_page_ids
from pdf_vlm.retrieve.page_retriever import PageRetriever
from pdf_vlm.retrieve.section_retriever import SectionRetriever
from pdf_vlm.schemas import QAExample, QAPrediction, RetrievalHit
from pdf_vlm.utils.io import ensure_dir, load_named_config, resolve_path, save_json
from pdf_vlm.utils.logging import get_logger

logger = get_logger("rag.text_only")

RetrievalMode = Literal["page", "section", "hierarchical"]


def build_text_retriever(
    index_dir: str | Path,
    *,
    retrieval: RetrievalMode | str = "page",
    device: str = "cpu",
    coarse_k: int = 5,
):
    index_dir = Path(index_dir)
    if retrieval == "page":
        return PageRetriever(index_dir, modality="text", device=device)
    if retrieval == "section":
        return SectionRetriever(index_dir, device=device)
    if retrieval in {"hierarchical", "hier"}:
        from pdf_vlm.retrieve.hierarchical_retriever import HierarchicalRetriever

        return HierarchicalRetriever(
            index_dir, modality="text", device=device, coarse_k=coarse_k
        )
    raise ValueError(f"text retriever supports page|section|hierarchical, got {retrieval}")


class TextOnlyRAGPipeline:
    """Strict OCR-text RAG baseline with page- or section-level retrieval."""

    modality: str = "text"

    def __init__(
        self,
        llm: Gemma3LlamaCpp | GemmaLlamaCpp | None,
        retriever,
        *,
        retrieval_mode: RetrievalMode | str = "page",
        top_k: int = 3,
        generation_cfg: dict[str, Any] | None = None,
        max_prompt_chars: int = 12000,
        dry_run: bool = False,
    ):
        self.llm = llm
        self.retriever = retriever
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.generation_cfg = generation_cfg or {"max_tokens": 256, "temperature": 0.1}
        self.max_prompt_chars = max_prompt_chars
        self.dry_run = dry_run or llm is None

    @classmethod
    def from_paths(
        cls,
        index_dir: str | Path,
        *,
        retrieval_mode: RetrievalMode | str = "page",
        top_k: int = 3,
        device: str = "cpu",
        dry_run: bool = False,
        model_cfg: dict[str, Any] | None = None,
        generation_cfg: dict[str, Any] | None = None,
        coarse_k: int = 5,
    ) -> "TextOnlyRAGPipeline":
        retriever = build_text_retriever(
            index_dir, retrieval=retrieval_mode, device=device, coarse_k=coarse_k
        )
        llm = None
        if not dry_run:
            cfg = model_cfg or load_named_config("models/gemma3_4b_qat.yaml")
            cfg = dict(cfg)
            cfg["local_path"] = str(resolve_path(cfg["local_path"]))
            # Text-only: disable vision path even if mmproj exists
            cfg["enable_vision"] = False
            cfg["mmproj_local_path"] = None
            llm = build_llm(cfg)
        return cls(
            llm,
            retriever,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            generation_cfg=generation_cfg,
            dry_run=dry_run,
        )

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalHit]:
        k = self.top_k if top_k is None else top_k
        hits = self.retriever.retrieve(question, top_k=k)
        # Enforce no image leakage in baseline
        for h in hits:
            h.image_path = None
            h.meta = {**(h.meta or {}), "modality": "text", "retrieval": self.retrieval_mode}
        return hits

    def answer(self, example: QAExample) -> QAPrediction:
        t0 = time.perf_counter()
        hits = self.retrieve(example.question, top_k=self.top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        page_ids = unique_page_ids(hits)
        prompt = build_text_prompt(
            example.question, hits, max_chars=self.max_prompt_chars
        )

        if self.dry_run:
            gen_text = ""
            generation_ms = 0.0
            peak_rss = None
            peak_vram = None
            prompt_tokens = None
            completion_tokens = None
        else:
            assert self.llm is not None
            t1 = time.perf_counter()
            gen = self.llm.generate_text(
                prompt,
                max_tokens=int(self.generation_cfg.get("max_tokens", 256)),
                temperature=float(self.generation_cfg.get("temperature", 0.1)),
                system=SYSTEM_PROMPT_TEXT_ONLY,
            )
            generation_ms = gen.latency_ms or ((time.perf_counter() - t1) * 1000.0)
            gen_text = gen.text
            peak_rss = gen.peak_rss_mb
            peak_vram = gen.peak_vram_mb
            prompt_tokens = gen.prompt_tokens
            completion_tokens = gen.completion_tokens

        e2e_ms = (time.perf_counter() - t0) * 1000.0
        pred = QAPrediction(
            example_id=example.example_id,
            doc_id=example.doc_id,
            question=example.question,
            prediction=gen_text,
            gold_answers=list(example.answers),
            retrieved_page_ids=page_ids,
            evidence_pages=list(example.evidence_pages),
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
            e2e_latency_ms=e2e_ms,
            peak_rss_mb=peak_rss,
            peak_vram_mb=peak_vram,
            hits=hits,
            meta={
                "baseline": "text_only_rag",
                "modality": "text",
                "retrieval_mode": self.retrieval_mode,
                "top_k": self.top_k,
                "dry_run": self.dry_run,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "unanswerable": example.unanswerable,
                **{k: v for k, v in example.meta.items() if k != "prompt"},
            },
        )
        logger.info(
            "qa_id=%s retrieval=%s pages=%s latency_ms=ret:%.1f gen:%.1f e2e:%.1f answer=%r",
            example.example_id,
            self.retrieval_mode,
            page_ids,
            retrieval_ms,
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
        """Run batch QA and persist prediction/retrieval logs."""
        preds = [self.answer(ex) for ex in examples]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = run_name or f"text_rag_{self.retrieval_mode}_{stamp}"
        out = ensure_dir(Path(out_dir) if out_dir else resolve_path(f"results/runs/{run_id}"))

        records = [prediction_log_record(p) for p in preds]
        save_json(out / "predictions.json", [p.model_dump(mode="json") for p in preds])
        save_json(out / "rag_log.json", records)
        summary = {
            "run_id": run_id,
            "baseline": "text_only_rag",
            "modality": "text",
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k,
            "n": len(preds),
            "latency_ms_mean": sum(p.e2e_latency_ms for p in preds) / max(len(preds), 1),
            "retrieval_latency_ms_mean": sum(p.retrieval_latency_ms for p in preds) / max(len(preds), 1),
            "generation_latency_ms_mean": sum(p.generation_latency_ms for p in preds) / max(len(preds), 1),
        }
        save_json(out / "run_summary.json", summary)
        logger.info("Wrote text-only RAG logs to %s", out)
        return {"out_dir": str(out), "summary": summary, "predictions": preds}


def prediction_log_record(pred: QAPrediction) -> dict[str, Any]:
    """Compact eval/log schema: answer, retrieved pages, latency, retrieval hits."""
    return {
        "example_id": pred.example_id,
        "doc_id": pred.doc_id,
        "question": pred.question,
        "answer": pred.prediction,
        "gold_answers": pred.gold_answers,
        "retrieved_pages": pred.retrieved_page_ids,
        "evidence_pages": pred.evidence_pages,
        "latency": {
            "retrieval_ms": pred.retrieval_latency_ms,
            "generation_ms": pred.generation_latency_ms,
            "e2e_ms": pred.e2e_latency_ms,
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
            "baseline": pred.meta.get("baseline"),
            "retrieval_mode": pred.meta.get("retrieval_mode"),
            "modality": "text",
            "peak_rss_mb": pred.peak_rss_mb,
            "peak_vram_mb": pred.peak_vram_mb,
        },
    }


# Backward-compatible thin wrapper used by older scripts
class RAGPipeline(TextOnlyRAGPipeline):
    def __init__(self, llm, retriever, *, modality: str = "text", top_k: int = 3, generation_cfg=None):
        if modality != "text":
            logger.warning("RAGPipeline multimodal path moved; forcing text-only for this instance")
        super().__init__(
            llm,
            retriever,
            retrieval_mode="page",
            top_k=top_k,
            generation_cfg=generation_cfg,
            dry_run=llm is None,
        )


def build_retriever(
    index_dir: str | Path,
    *,
    retrieval: str = "page",
    modality: str = "text",
    retrieval_cfg: dict[str, Any] | None = None,
    device: str = "cpu",
):
    """Factory kept for older scripts; text-only ignores multimodal flags."""
    retrieval_cfg = retrieval_cfg or {}
    if modality != "text":
        logger.warning("build_retriever called with modality=%s; text-only baseline uses text", modality)
    if retrieval in {"page"}:
        return build_text_retriever(index_dir, retrieval="page", device=device)
    if retrieval in {"section"}:
        return build_text_retriever(index_dir, retrieval="section", device=device)
    # hierarchical still available for experiments
    from pdf_vlm.retrieve.hierarchical_retriever import HierarchicalRetriever

    return HierarchicalRetriever(
        index_dir,
        modality="text",
        device=device,
        coarse_k=int(retrieval_cfg.get("coarse_k", 5)),
        fine_k=int(retrieval_cfg.get("fine_k", 3)),
    )


def enrich_artifact_from_pdf_text(doc_id: str) -> None:
    """Optional helper: if OCR stub is weak, fill page markdown from PDF text layer."""
    art = load_artifact(doc_id)
    if not art.source_path or not Path(art.source_path).exists():
        return
    try:
        import fitz
    except ImportError:
        return
    pdf = fitz.open(art.source_path)
    changed = False
    for page in art.pages:
        if page.page_id >= len(pdf):
            continue
        text = pdf.load_page(page.page_id).get_text("text").strip()
        if text and (not page.markdown or page.markdown.startswith("Stub OCR")):
            page.markdown = text
            changed = True
    pdf.close()
    if changed:
        from pdf_vlm.ocr.normalize import build_sections
        from pdf_vlm.utils.io import resolve_path, save_model

        art.sections = build_sections(art.pages)
        art.full_markdown = "\n\n".join(
            f"<!-- page {p.page_id} -->\n{p.markdown}" for p in art.pages if p.markdown
        )
        save_model(resolve_path(f"data/processed/{doc_id}/document.json"), art)
        logger.info("Enriched OCR artifact text from PDF text layer: %s", doc_id)
