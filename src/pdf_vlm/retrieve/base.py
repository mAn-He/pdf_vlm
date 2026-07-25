"""Common retriever protocol for text-only and multimodal RAG."""

from __future__ import annotations

from typing import Protocol

from pdf_vlm.retrieve.result import RetrievalResult
from pdf_vlm.schemas import RetrievalHit


class Retriever(Protocol):
    """Shared interface so text-only / multimodal RAG stay comparable."""

    mode: str

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        ...


class DetailedRetriever(Protocol):
    """Extended interface exposing coarse-to-fine traces."""

    mode: str

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        ...

    def retrieve_detailed(self, query: str, top_k: int = 3) -> RetrievalResult:
        ...


def unique_page_ids(hits: list[RetrievalHit]) -> list[int]:
    pages: list[int] = []
    for hit in hits:
        for pid in hit.page_ids:
            if pid not in pages:
                pages.append(pid)
    return pages


def build_retriever_for_mode(
    index_dir,
    *,
    mode: str,
    device: str = "cpu",
    coarse_k: int = 5,
    modality: str = "text",
):
    """Factory used by text-only / multimodal pipelines."""
    if mode == "page":
        from pdf_vlm.retrieve.page_retriever import PageRetriever

        return PageRetriever(index_dir, modality=modality, device=device)
    if mode in {"hierarchical", "hier"}:
        from pdf_vlm.retrieve.hierarchical_retriever import HierarchicalRetriever

        return HierarchicalRetriever(
            index_dir, modality=modality, device=device, coarse_k=coarse_k
        )
    if mode == "section":
        from pdf_vlm.retrieve.section_retriever import SectionRetriever

        return SectionRetriever(index_dir, device=device)
    raise ValueError(f"Unknown retrieval mode: {mode}")
