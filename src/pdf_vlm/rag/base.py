"""Shared RAG pipeline protocol (text-only and multimodal)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pdf_vlm.schemas import QAExample, QAPrediction, RetrievalHit


class RAGPipelineProtocol(Protocol):
    """Common interface so baselines remain comparable."""

    modality: str

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalHit]:
        ...

    def answer(self, example: QAExample) -> QAPrediction:
        ...

    def run(
        self,
        examples: list[QAExample],
        *,
        out_dir: str | Path | None = None,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        ...
