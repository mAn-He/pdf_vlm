"""Per-example evaluation row schema for the harness."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvalRow(BaseModel):
    """One scored QA example under a fixed experiment cell."""

    run_id: str
    dataset: str
    doc_id: str
    question_id: str
    question_type: str = "text"
    page_bucket: int | None = None

    pipeline_type: str  # text | multimodal
    retrieval_type: str  # page | hierarchical | section
    top_k: int = 3

    question: str = ""
    answer: str = ""
    gold_answer: str = ""
    gold_answers: list[str] = Field(default_factory=list)

    # Correctness
    primary_metric: str = "anls"
    correctness: float = 0.0
    anls: float = 0.0
    em: float = 0.0
    token_f1: float = 0.0

    # Latency / memory
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    image_load_latency_ms: float | None = None
    peak_rss_mb: float | None = None
    peak_vram_mb: float | None = None

    # Retrieval
    retrieved_pages: list[int] = Field(default_factory=list)
    evidence_pages: list[int] = Field(default_factory=list)
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    page_hit_at_1: float = 0.0
    page_hit_at_3: float = 0.0
    page_hit_at_5: float = 0.0
    gold_answer_contained: bool = False

    meta: dict[str, Any] = Field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        """CSV-friendly flat dict (lists joined)."""
        d = self.model_dump(mode="json")
        d["gold_answers"] = " | ".join(self.gold_answers)
        d["retrieved_pages"] = ",".join(str(p) for p in self.retrieved_pages)
        d["evidence_pages"] = ",".join(str(p) for p in self.evidence_pages)
        d["meta"] = ""  # keep CSV clean; full meta in JSON
        return d
