"""Normalize raw OCR outputs into DocumentArtifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pdf_vlm.index.hierarchy import build_document_hierarchy
from pdf_vlm.schemas import Block, DocumentArtifact, PageArtifact, SectionNode, TableBlock


def _blocks_from_structure(page_id: int, raw: dict[str, Any]) -> tuple[list[Block], list[TableBlock], str]:
    """Best-effort normalization across PaddleOCR Structure output shapes."""
    blocks: list[Block] = []
    tables: list[TableBlock] = []
    md_parts: list[str] = []

    # PP-StructureV3 style: parsing_res_list / layout / markdown
    if isinstance(raw.get("markdown"), str) and raw["markdown"].strip():
        md_parts.append(raw["markdown"])

    res_list = raw.get("parsing_res_list") or raw.get("layout") or raw.get("res") or []
    if isinstance(res_list, dict):
        res_list = res_list.get("layout") or res_list.get("result") or []

    for i, item in enumerate(res_list if isinstance(res_list, list) else []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("type") or item.get("block_label") or "text")
        text = item.get("content") or item.get("text") or item.get("res") or ""
        if isinstance(text, list):
            text = "\n".join(str(t) for t in text)
        text = str(text).strip()
        html = item.get("html")
        block_id = f"p{page_id}_b{i}"
        block_type = "table" if "table" in label.lower() else ("title" if "title" in label.lower() else "text")
        if block_type == "table" or html:
            tables.append(
                TableBlock(
                    table_id=f"p{page_id}_t{len(tables)}",
                    page_id=page_id,
                    html=str(html or text),
                    text=text,
                )
            )
            md_parts.append(str(html or text))
        else:
            blocks.append(
                Block(
                    block_id=block_id,
                    page_id=page_id,
                    block_type=block_type,
                    text=text,
                    section_id=f"sec_p{page_id}" if block_type == "title" else None,
                )
            )
            if text:
                md_parts.append(text)

    # Fallback: plain OCR lines
    if not md_parts:
        lines = raw.get("ocr_texts") or raw.get("rec_texts") or raw.get("texts") or []
        if isinstance(lines, list):
            joined = "\n".join(str(x) for x in lines if x)
            if joined:
                md_parts.append(joined)
                blocks.append(Block(block_id=f"p{page_id}_b0", page_id=page_id, text=joined))

    # If structure produced little text but ocr_texts is rich, append
    if raw.get("ocr_texts") and isinstance(raw["ocr_texts"], list):
        blob = "\n".join(str(x) for x in raw["ocr_texts"] if x)
        if blob and blob not in "\n\n".join(md_parts):
            if not md_parts:
                md_parts.append(blob)
                blocks.append(Block(block_id=f"p{page_id}_b0", page_id=page_id, text=blob))

    markdown = "\n\n".join(md_parts).strip()
    return blocks, tables, markdown


def build_sections(pages: list[PageArtifact]) -> list[SectionNode]:
    """Heuristic section tree via hierarchy module (heading/markdown/spacing/page)."""
    tmp = DocumentArtifact(doc_id="_tmp", source_path="", num_pages=len(pages), pages=pages)
    sections, _ = build_document_hierarchy(tmp, strategy="auto")
    return sections


def normalize_document(
    doc_id: str,
    source_path: str,
    page_metas: list[dict[str, Any]],
    page_raw_results: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> DocumentArtifact:
    pages: list[PageArtifact] = []
    for meta_page, raw in zip(page_metas, page_raw_results, strict=False):
        page_id = int(meta_page["page_id"])
        blocks, tables, markdown = _blocks_from_structure(page_id, raw)
        pages.append(
            PageArtifact(
                page_id=page_id,
                image_path=meta_page.get("image_path"),
                width=meta_page.get("width"),
                height=meta_page.get("height"),
                markdown=markdown,
                blocks=blocks,
                tables=tables,
            )
        )

    sections = build_sections(pages)
    full_md = "\n\n".join(f"<!-- page {p.page_id} -->\n{p.markdown}" for p in pages if p.markdown)
    return DocumentArtifact(
        doc_id=doc_id,
        source_path=str(Path(source_path)),
        num_pages=len(pages),
        pages=pages,
        sections=sections,
        full_markdown=full_md,
        meta=meta or {},
    )
