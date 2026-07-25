"""Shared Pydantic schemas for artifacts, retrieval, and QA examples."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Modality(str, Enum):
    TEXT = "text"
    MULTIMODAL = "multimodal"


class RetrievalMode(str, Enum):
    PAGE = "page"
    HIERARCHICAL = "hierarchical"


class QuestionType(str, Enum):
    """Canonical question type for cross-dataset evaluation."""

    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    CROSS_PAGE = "cross-page"
    IMAGE = "image"
    LAYOUT = "layout"
    UNANSWERABLE = "unanswerable"
    OTHER = "other"


class BBox(BaseModel):
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0


class Block(BaseModel):
    block_id: str
    page_id: int
    block_type: str = "text"
    text: str = ""
    html: str | None = None
    bbox: BBox | None = None
    section_id: str | None = None


class TableBlock(BaseModel):
    table_id: str
    page_id: int
    html: str = ""
    text: str = ""
    bbox: BBox | None = None


class PageArtifact(BaseModel):
    page_id: int
    image_path: str | None = None
    width: int | None = None
    height: int | None = None
    markdown: str = ""
    blocks: list[Block] = Field(default_factory=list)
    tables: list[TableBlock] = Field(default_factory=list)


class SectionNode(BaseModel):
    section_id: str
    title: str = ""
    page_ids: list[int] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)
    summary_text: str = ""


class DocumentArtifact(BaseModel):
    doc_id: str
    source_path: str
    num_pages: int
    pages: list[PageArtifact] = Field(default_factory=list)
    sections: list[SectionNode] = Field(default_factory=list)
    full_markdown: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)

    def page_markdown(self, page_id: int) -> str:
        for page in self.pages:
            if page.page_id == page_id:
                return page.markdown
        return ""

    def evidence_texts(self, page_ids: list[int]) -> list[str]:
        return [self.page_markdown(pid) for pid in page_ids]


class PageRef(BaseModel):
    """Lightweight page reference used by dataset loaders (pre-OCR)."""

    page_id: int
    image_path: str | None = None
    pdf_path: str | None = None
    width: int | None = None
    height: int | None = None
    text: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class QAPair(BaseModel):
    """One QA item under a document (common evaluation unit)."""

    qa_id: str
    question: str
    answer: str | list[str]
    question_type: QuestionType = QuestionType.TEXT
    evidence_pages: list[int] = Field(default_factory=list)
    unanswerable: bool = False
    answer_type: str | None = None
    evidence_modality: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("answer", mode="before")
    @classmethod
    def _coerce_answer(cls, v: Any) -> str | list[str]:
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        return str(v)

    def answers_list(self) -> list[str]:
        if isinstance(self.answer, list):
            return [str(a) for a in self.answer]
        if self.answer == "":
            return []
        return [str(self.answer)]

    @property
    def primary_answer(self) -> str:
        answers = self.answers_list()
        return answers[0] if answers else ""


class DatasetDocument(BaseModel):
    """Common document record shared by MP-DocVQA / MMLongBench / custom."""

    doc_id: str
    pages: list[PageRef] = Field(default_factory=list)
    qa_pairs: list[QAPair] = Field(default_factory=list)
    source_path: str | None = None
    source_dataset: str = ""
    num_pages: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.num_pages is None:
            self.num_pages = len(self.pages)

    @property
    def page_count(self) -> int:
        if self.num_pages is not None and self.num_pages > 0:
            return int(self.num_pages)
        return len(self.pages)


class DatasetBundle(BaseModel):
    """Loaded dataset ready for evaluation iteration."""

    name: str
    documents: list[DatasetDocument] = Field(default_factory=list)
    split: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    def iter_qa(self) -> list[tuple[DatasetDocument, QAPair]]:
        out: list[tuple[DatasetDocument, QAPair]] = []
        for doc in self.documents:
            for qa in doc.qa_pairs:
                out.append((doc, qa))
        return out

    @property
    def n_docs(self) -> int:
        return len(self.documents)

    @property
    def n_questions(self) -> int:
        return sum(len(d.qa_pairs) for d in self.documents)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    page_ids: list[int] = Field(default_factory=list)
    section_id: str | None = None
    level: str = "page"
    text: str = ""
    image_path: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    chunk_id: str
    score: float
    doc_id: str
    page_ids: list[int] = Field(default_factory=list)
    section_id: str | None = None
    level: str = "page"
    text: str = ""
    image_path: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class QAExample(BaseModel):
    """Flat QA example used by the RAG/eval pipeline (derived from QAPair)."""

    example_id: str
    doc_id: str
    question: str
    answers: list[str] = Field(default_factory=list)
    evidence_pages: list[int] = Field(default_factory=list)
    unanswerable: bool = False
    answer_type: str | None = None
    question_type: QuestionType | None = None
    evidence_modality: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_qa_pair(cls, doc: DatasetDocument, qa: QAPair) -> "QAExample":
        return cls(
            example_id=qa.qa_id,
            doc_id=doc.doc_id,
            question=qa.question,
            answers=qa.answers_list(),
            evidence_pages=list(qa.evidence_pages),
            unanswerable=qa.unanswerable,
            answer_type=qa.answer_type,
            question_type=qa.question_type,
            evidence_modality=list(qa.evidence_modality),
            meta={
                "source_dataset": doc.source_dataset,
                "unanswerable": qa.unanswerable,
                **qa.meta,
            },
        )


class GenerationResult(BaseModel):
    text: str
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    tokens_per_sec: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    peak_rss_mb: float | None = None
    peak_vram_mb: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class QAPrediction(BaseModel):
    example_id: str
    doc_id: str
    question: str
    prediction: str
    gold_answers: list[str] = Field(default_factory=list)
    retrieved_page_ids: list[int] = Field(default_factory=list)
    evidence_pages: list[int] = Field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    peak_rss_mb: float | None = None
    peak_vram_mb: float | None = None
    hits: list[RetrievalHit] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
