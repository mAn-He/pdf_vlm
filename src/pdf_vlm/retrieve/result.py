"""Retrieval result types with coarse-to-fine path traces."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from pdf_vlm.schemas import RetrievalHit


class StageHit(BaseModel):
    """One candidate at a retrieval stage (for visual debugging)."""

    stage: str  # document | section | page | paragraph
    chunk_id: str
    score: float
    page_ids: list[int] = Field(default_factory=list)
    section_id: str | None = None
    title: str | None = None
    text_preview: str = ""
    kept: bool = True


class RetrievalResult(BaseModel):
    """Unified retrieval output for page-level and hierarchical modes."""

    query: str
    mode: str  # page | hierarchical
    final_hits: list[RetrievalHit] = Field(default_factory=list)
    path: list[StageHit] = Field(default_factory=list)
    coarse_hits: list[StageHit] = Field(default_factory=list)
    fine_hits: list[StageHit] = Field(default_factory=list)
    latency_ms: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def page_ids(self) -> list[int]:
        pages: list[int] = []
        for hit in self.final_hits:
            for pid in hit.page_ids:
                if pid not in pages:
                    pages.append(pid)
        return pages

    def format_trace(self) -> str:
        """Human-readable coarse-to-fine search path."""
        lines = [
            f"mode={self.mode} latency_ms={self.latency_ms:.1f}",
            f"query={self.query!r}",
        ]
        if self.mode == "hierarchical":
            lines.append("-- COARSE (section) --")
            if not self.coarse_hits:
                lines.append("  (none)")
            for h in self.coarse_hits:
                mark = "KEEP" if h.kept else "drop"
                lines.append(
                    f"  [{mark}] {h.chunk_id} score={h.score:.4f} "
                    f"pages={h.page_ids} title={h.title!r}"
                )
            lines.append("-- FINE (page/paragraph) --")
            for h in self.fine_hits:
                mark = "KEEP" if h.kept else "drop"
                lines.append(
                    f"  [{mark}] {h.chunk_id} score={h.score:.4f} "
                    f"pages={h.page_ids} preview={h.text_preview[:80]!r}"
                )
        else:
            lines.append("-- PAGE CANDIDATES --")
            for h in self.fine_hits or [
                StageHit(
                    stage="page",
                    chunk_id=hit.chunk_id,
                    score=hit.score,
                    page_ids=hit.page_ids,
                    text_preview=(hit.text or "")[:120],
                    kept=True,
                )
                for hit in self.final_hits
            ]:
                lines.append(
                    f"  [KEEP] {h.chunk_id} score={h.score:.4f} pages={h.page_ids} "
                    f"preview={h.text_preview[:80]!r}"
                )
        lines.append(f"FINAL pages={self.page_ids}")
        return "\n".join(lines)
