"""End-to-end RAG pipeline exports.

Prefer TextOnlyRAGPipeline for the text-only baseline.
"""

from __future__ import annotations

from pdf_vlm.rag.text_only import (
    RAGPipeline,
    TextOnlyRAGPipeline,
    build_retriever,
    build_text_retriever,
    prediction_log_record,
)

__all__ = [
    "TextOnlyRAGPipeline",
    "RAGPipeline",
    "build_retriever",
    "build_text_retriever",
    "prediction_log_record",
]
