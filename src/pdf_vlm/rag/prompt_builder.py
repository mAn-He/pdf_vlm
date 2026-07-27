"""Prompt templates for RAG generators (Gemma 3).

Text-only baseline never includes image placeholders.
Multimodal templates use OCR text + top-k page images only.
"""

from __future__ import annotations

from typing import Any

from pdf_vlm.schemas import RetrievalHit

SYSTEM_PROMPT_TEXT_ONLY = (
    "You are a document question-answering assistant. "
    "Use ONLY the OCR text evidence provided below. "
    "Do not assume information that is not present in the evidence. "
    "If the evidence is insufficient, reply exactly: unanswerable. "
    "Answer in Korean when the question is in Korean. "
    "Return only a short final answer — no explanations, no page citations."
)

SYSTEM_PROMPT_MULTIMODAL = (
    "You are a multimodal document question-answering assistant. "
    "Use the OCR text evidence AND the attached page images together. "
    "Images are especially useful for tables, charts, and captions that OCR may miss. "
    "If the evidence is insufficient, reply exactly: unanswerable. "
    "Answer in Korean when the question is in Korean. "
    "Return only a short final answer — no explanations, no page citations "
    "(do not write [page N] or similar)."
)

SYSTEM_PROMPT = SYSTEM_PROMPT_TEXT_ONLY

TEXT_ONLY_USER_TEMPLATE = """# Document Evidence (OCR text only)
{evidence}

# Question
{question}

# Instructions
- Base the answer strictly on the evidence above.
- If multiple evidence blocks conflict, prefer the higher-ranked (earlier) block.
- If the question is in Korean, answer in Korean.
- Output only the final answer (no page citations, no commentary).
"""

MULTIMODAL_USER_TEMPLATE = """# Question
{question}

# OCR Text Evidence (retrieved top-k pages only)
{ocr_evidence}

# Page Image Evidence
The attached images correspond to these retrieved pages (in order):
{image_manifest}

# Instructions
- Use both OCR text and page images. Prefer images for tables/charts/layout.
- If the question is in Korean, answer in Korean.
- Do NOT include page citations like [page 1] or [pages 1, 3] in the answer.
- If evidence is insufficient, reply exactly: unanswerable.
- Output only the short final answer.
"""

def format_text_evidence(hits: list[RetrievalHit], max_chars: int = 12000) -> str:
    parts: list[str] = []
    used = 0
    for i, hit in enumerate(hits):
        level = hit.level or "chunk"
        sec = f" section={hit.section_id}" if hit.section_id else ""
        header = (
            f"[Evidence {i + 1} | level={level} | pages={hit.page_ids}{sec} "
            f"| score={hit.score:.4f}]"
        )
        body = (hit.text or "").strip() or "(empty)"
        block = f"{header}\n{body}\n"
        if used + len(block) > max_chars:
            remain = max_chars - used
            if remain > 80:
                parts.append(block[:remain] + "\n...[truncated]")
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts) if parts else "(no evidence retrieved)"


def format_ocr_pages(pages: list[dict[str, Any]], max_chars: int = 10000) -> str:
    parts: list[str] = []
    used = 0
    for i, page in enumerate(pages):
        header = (
            f"[OCR page {page.get('page_id')} | rank={i + 1} "
            f"| score={float(page.get('score', 0.0)):.4f}]"
        )
        body = (page.get("ocr_text") or "").strip() or "(empty OCR)"
        block = f"{header}\n{body}\n"
        if used + len(block) > max_chars:
            remain = max_chars - used
            if remain > 80:
                parts.append(block[:remain] + "\n...[truncated]")
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts) if parts else "(no OCR evidence)"


def format_image_manifest(pages: list[dict[str, Any]]) -> str:
    lines = []
    for i, page in enumerate(pages):
        status = "attached" if page.get("exists") else "MISSING"
        lines.append(
            f"- Image {i + 1}: page_id={page.get('page_id')} ({status}) "
            f"path={page.get('image_path')}"
        )
    return "\n".join(lines) if lines else "(no page images selected)"


def build_text_prompt(
    question: str,
    hits: list[RetrievalHit],
    *,
    max_chars: int = 12000,
    template: str = TEXT_ONLY_USER_TEMPLATE,
) -> str:
    """Strict text-only prompt for Gemma 3 chat completion (user turn)."""
    evidence = format_text_evidence(hits, max_chars=max_chars)
    return template.format(evidence=evidence, question=question.strip())


def build_multimodal_prompt(
    question: str,
    pages: list[dict[str, Any]],
    *,
    max_chars: int = 10000,
    template: str = MULTIMODAL_USER_TEMPLATE,
) -> tuple[str, list[str]]:
    """Build multimodal user prompt + ordered image paths (existing files only).

    Only top-k retrieved pages are included — never the full document.
    """
    ocr_evidence = format_ocr_pages(pages, max_chars=max_chars)
    image_manifest = format_image_manifest(pages)
    prompt = template.format(
        question=question.strip(),
        ocr_evidence=ocr_evidence,
        image_manifest=image_manifest,
    )
    image_paths = [str(p["image_path"]) for p in pages if p.get("exists") and p.get("image_path")]
    return prompt, image_paths
