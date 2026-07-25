"""Section / hierarchy construction heuristics for coarse-to-fine retrieval.

Strategies (tried in order when strategy='auto'):
  1) heading  — OCR title/heading blocks
  2) markdown — markdown ATATX headings (# ##)
  3) spacing  — blank-line paragraph groups merged per page windows
  4) page     — one section per page (always available fallback)
  5) toc      — optional TOC entries from artifact.meta['toc']
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pdf_vlm.schemas import DocumentArtifact, PageArtifact, SectionNode
from pdf_vlm.utils.logging import get_logger

logger = get_logger("index.hierarchy")

HierarchyStrategy = Literal["auto", "heading", "markdown", "spacing", "page", "toc"]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_NUMBERED_HEADING_RE = re.compile(
    r"^(\d+(\.\d+)*|[IVXLC]+\.|[A-Z]\.)\s+\S.+$", re.MULTILINE
)


def _clip(text: str, n: int = 2000) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n].rstrip() + "…"


def sections_from_headings(pages: list[PageArtifact]) -> list[SectionNode]:
    sections: list[SectionNode] = []
    current: SectionNode | None = None
    for page in pages:
        titles = [b for b in page.blocks if b.block_type in {"title", "heading"} and b.text.strip()]
        if not titles:
            if current is None:
                current = SectionNode(
                    section_id=f"sec_p{page.page_id}",
                    title=f"Page {page.page_id + 1}",
                    page_ids=[page.page_id],
                    block_ids=[b.block_id for b in page.blocks],
                    summary_text=_clip(page.markdown),
                )
                sections.append(current)
            else:
                if page.page_id not in current.page_ids:
                    current.page_ids.append(page.page_id)
                extra = page.markdown.strip()
                if extra:
                    current.summary_text = _clip(current.summary_text + "\n" + extra, 4000)
            continue
        for title in titles:
            sid = title.section_id or f"sec_{len(sections)}"
            current = SectionNode(
                section_id=sid,
                title=title.text[:200],
                page_ids=[page.page_id],
                block_ids=[title.block_id],
                summary_text=title.text,
            )
            sections.append(current)
        assert current is not None
        extras = [b.text for b in page.blocks if b.block_type not in {"title", "heading"} and b.text]
        if extras:
            current.summary_text = _clip(current.summary_text + "\n" + "\n".join(extras), 4000)
    return sections


def sections_from_markdown(pages: list[PageArtifact]) -> list[SectionNode]:
    sections: list[SectionNode] = []
    current: SectionNode | None = None
    for page in pages:
        md = page.markdown or ""
        matches = list(_HEADING_RE.finditer(md))
        if not matches:
            matches = list(_NUMBERED_HEADING_RE.finditer(md))
            if matches:
                # treat whole match as title
                for m in matches:
                    title = m.group(0).strip()
                    current = SectionNode(
                        section_id=f"sec_md_{len(sections)}",
                        title=title[:200],
                        page_ids=[page.page_id],
                        summary_text=title,
                    )
                    sections.append(current)
                if current is not None:
                    current.summary_text = _clip(md, 4000)
                continue
            if current is None:
                current = SectionNode(
                    section_id=f"sec_p{page.page_id}",
                    title=f"Page {page.page_id + 1}",
                    page_ids=[page.page_id],
                    summary_text=_clip(md),
                )
                sections.append(current)
            else:
                if page.page_id not in current.page_ids:
                    current.page_ids.append(page.page_id)
                current.summary_text = _clip(current.summary_text + "\n" + md, 4000)
            continue

        for i, m in enumerate(matches):
            title = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
            body = md[start:end].strip()
            current = SectionNode(
                section_id=f"sec_md_{len(sections)}",
                title=title[:200],
                page_ids=[page.page_id],
                summary_text=_clip(f"{title}\n{body}"),
            )
            sections.append(current)
    return sections


def sections_from_spacing(pages: list[PageArtifact], max_pages_per_section: int = 3) -> list[SectionNode]:
    """Group consecutive pages into sections using blank-line paragraph density."""
    sections: list[SectionNode] = []
    buf_pages: list[PageArtifact] = []
    buf_text: list[str] = []

    def flush(title_hint: str | None = None) -> None:
        nonlocal buf_pages, buf_text
        if not buf_pages:
            return
        first = buf_pages[0]
        title = title_hint or (first.markdown.strip().split("\n", 1)[0][:80] or f"Page {first.page_id + 1}")
        sections.append(
            SectionNode(
                section_id=f"sec_sp_{len(sections)}",
                title=title,
                page_ids=[p.page_id for p in buf_pages],
                summary_text=_clip("\n\n".join(buf_text), 4000),
            )
        )
        buf_pages, buf_text = [], []

    for page in pages:
        text = (page.markdown or "").strip()
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        # start new section on sparse pages or capacity
        if buf_pages and (
            len(buf_pages) >= max_pages_per_section or (paras and len(paras[0]) < 40 and paras[0].isupper())
        ):
            flush()
        buf_pages.append(page)
        buf_text.append(text)
    flush()
    return sections


def sections_from_pages(pages: list[PageArtifact]) -> list[SectionNode]:
    return [
        SectionNode(
            section_id=f"sec_p{page.page_id}",
            title=f"Page {page.page_id + 1}",
            page_ids=[page.page_id],
            block_ids=[b.block_id for b in page.blocks],
            summary_text=_clip(page.markdown),
        )
        for page in pages
    ]


def sections_from_toc(pages: list[PageArtifact], toc: list[dict[str, Any]]) -> list[SectionNode]:
    """Build sections from TOC entries: [{title, page_id|start_page, end_page?}]."""
    if not toc:
        return []
    page_map = {p.page_id: p for p in pages}
    sections: list[SectionNode] = []
    for i, item in enumerate(toc):
        title = str(item.get("title") or item.get("text") or f"Section {i}")
        start = int(item.get("page_id", item.get("start_page", 0)))
        end = int(item.get("end_page", start))
        pids = [pid for pid in range(start, end + 1) if pid in page_map]
        if not pids:
            continue
        summary = "\n\n".join(page_map[pid].markdown for pid in pids)
        sections.append(
            SectionNode(
                section_id=f"sec_toc_{i}",
                title=title[:200],
                page_ids=pids,
                summary_text=_clip(f"{title}\n{summary}", 4000),
            )
        )
    return sections


def build_document_hierarchy(
    doc: DocumentArtifact,
    *,
    strategy: HierarchyStrategy = "auto",
) -> tuple[list[SectionNode], str]:
    """Return (sections, strategy_used). Mutates nothing; caller may assign to doc.sections."""
    pages = doc.pages
    toc = list((doc.meta or {}).get("toc") or [])

    def try_strategy(name: str) -> list[SectionNode]:
        if name == "toc":
            return sections_from_toc(pages, toc)
        if name == "heading":
            return sections_from_headings(pages)
        if name == "markdown":
            return sections_from_markdown(pages)
        if name == "spacing":
            return sections_from_spacing(pages)
        if name == "page":
            return sections_from_pages(pages)
        raise ValueError(name)

    if strategy != "auto":
        secs = try_strategy(strategy)
        if not secs:
            secs = sections_from_pages(pages)
            return secs, "page"
        return secs, strategy

    # auto: prefer informative structure
    for name in ("toc", "heading", "markdown", "spacing", "page"):
        secs = try_strategy(name)
        if not secs:
            continue
        if name == "toc":
            logger.info("hierarchy strategy=toc sections=%d doc=%s", len(secs), doc.doc_id)
            return secs, "toc"
        if name == "heading":
            # accept only if real title blocks produced non-page section ids
            if any(not s.section_id.startswith("sec_p") for s in secs):
                logger.info("hierarchy strategy=heading sections=%d doc=%s", len(secs), doc.doc_id)
                return secs, "heading"
            continue
        if name == "markdown":
            # accept only ATATX / numbered heading sections
            if any(s.section_id.startswith("sec_md_") for s in secs):
                logger.info("hierarchy strategy=markdown sections=%d doc=%s", len(secs), doc.doc_id)
                return secs, "markdown"
            continue
        if name == "spacing" and len(secs) >= 1:
            logger.info("hierarchy strategy=spacing sections=%d doc=%s", len(secs), doc.doc_id)
            return secs, "spacing"

    secs = sections_from_pages(pages)
    return secs, "page"


def apply_hierarchy(doc: DocumentArtifact, strategy: HierarchyStrategy = "auto") -> DocumentArtifact:
    sections, used = build_document_hierarchy(doc, strategy=strategy)
    meta = dict(doc.meta or {})
    meta["hierarchy_strategy"] = used
    meta["n_sections"] = len(sections)
    return doc.model_copy(update={"sections": sections, "meta": meta})
