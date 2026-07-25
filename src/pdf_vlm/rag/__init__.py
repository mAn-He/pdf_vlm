"""RAG package exports (lazy to avoid circular imports with eval/llm)."""

from __future__ import annotations

__all__ = [
    "TextOnlyRAGPipeline",
    "MultimodalRAGPipeline",
    "build_text_retriever",
    "SYSTEM_PROMPT_TEXT_ONLY",
    "SYSTEM_PROMPT_MULTIMODAL",
    "build_text_prompt",
    "build_multimodal_prompt",
]


def __getattr__(name: str):
    if name in {"TextOnlyRAGPipeline", "build_text_retriever"}:
        from pdf_vlm.rag.text_only import TextOnlyRAGPipeline, build_text_retriever

        return TextOnlyRAGPipeline if name == "TextOnlyRAGPipeline" else build_text_retriever
    if name == "MultimodalRAGPipeline":
        from pdf_vlm.rag.multimodal import MultimodalRAGPipeline

        return MultimodalRAGPipeline
    if name in {
        "SYSTEM_PROMPT_TEXT_ONLY",
        "SYSTEM_PROMPT_MULTIMODAL",
        "build_text_prompt",
        "build_multimodal_prompt",
    }:
        from pdf_vlm.rag import prompt_builder as pb

        return getattr(pb, name)
    raise AttributeError(name)
