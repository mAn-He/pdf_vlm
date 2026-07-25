"""Strict text-only indexing from DocumentArtifact (OCR/parsed JSON).

Modes:
  - page: one chunk per page markdown
  - section: one chunk per section summary_text (no images)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pdf_vlm.index.store import VectorStore
from pdf_vlm.index.text_embedder import TextEmbedder
from pdf_vlm.schemas import Chunk, DocumentArtifact
from pdf_vlm.utils.io import ensure_dir, save_json
from pdf_vlm.utils.logging import get_logger

logger = get_logger("index.text_only")

TextRetrievalMode = Literal["page", "section"]


def build_text_page_chunks(doc: DocumentArtifact) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in doc.pages:
        text = (page.markdown or "").strip()
        # Prefer concatenated block text if markdown empty but blocks exist
        if not text and page.blocks:
            text = "\n".join(b.text for b in page.blocks if b.text.strip())
        if page.tables:
            table_bits = [t.html or t.text for t in page.tables if (t.html or t.text)]
            if table_bits:
                text = (text + "\n" + "\n".join(table_bits)).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::page::{page.page_id}",
                doc_id=doc.doc_id,
                page_ids=[page.page_id],
                level="page",
                text=text,
                image_path=None,  # strict text-only: never index images
                meta={"source": "ocr_text", "modality": "text"},
            )
        )
    return chunks


def build_text_section_chunks(doc: DocumentArtifact) -> list[Chunk]:
    chunks: list[Chunk] = []
    if doc.sections:
        for sec in doc.sections:
            text = (sec.summary_text or sec.title or "").strip()
            if not text:
                # fall back to joining page markdown for section pages
                parts = [doc.page_markdown(pid) for pid in sec.page_ids]
                text = "\n\n".join(p for p in parts if p).strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::section::{sec.section_id}",
                    doc_id=doc.doc_id,
                    page_ids=list(sec.page_ids),
                    section_id=sec.section_id,
                    level="section",
                    text=text,
                    image_path=None,
                    meta={
                        "source": "ocr_text",
                        "modality": "text",
                        "title": sec.title,
                    },
                )
            )
        return chunks

    # No sections in artifact: synthesize one section per page
    for page in doc.pages:
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::section::sec_p{page.page_id}",
                doc_id=doc.doc_id,
                page_ids=[page.page_id],
                section_id=f"sec_p{page.page_id}",
                level="section",
                text=(page.markdown or "").strip(),
                image_path=None,
                meta={"source": "ocr_text", "modality": "text", "synthetic": True},
            )
        )
    return chunks


def build_text_only_index(
    docs: list[DocumentArtifact],
    out_dir: str | Path,
    *,
    mode: TextRetrievalMode = "page",
    retrieval_cfg: dict | None = None,
) -> Path:
    """Build a FAISS/numpy text index. Never embeds images."""
    retrieval_cfg = retrieval_cfg or {}
    out_dir = ensure_dir(Path(out_dir))

    chunks: list[Chunk] = []
    for doc in docs:
        if mode == "page":
            chunks.extend(build_text_page_chunks(doc))
        elif mode == "section":
            chunks.extend(build_text_section_chunks(doc))
        else:
            raise ValueError(f"Unsupported text retrieval mode: {mode}")

    if not chunks:
        raise ValueError("No text chunks to index (empty DocumentArtifact?)")

    emb_cfg = retrieval_cfg.get("embedder", {})
    embedder = TextEmbedder(
        name=emb_cfg.get("name", "BAAI/bge-m3"),
        device=emb_cfg.get("device", "cpu"),
        normalize=bool(emb_cfg.get("normalize", True)),
        dim=int(emb_cfg.get("dim", 1024)),
        backend=emb_cfg.get("backend"),
    )
    vectors = embedder.embed([c.text or "" for c in chunks])
    store = VectorStore(dim=int(vectors.shape[1]), metric=retrieval_cfg.get("index", {}).get("metric", "ip"))
    store.add([c.chunk_id for c in chunks], vectors)
    store.save(out_dir)
    save_json(out_dir / "chunks.json", [c.model_dump(mode="json") for c in chunks])
    save_json(
        out_dir / "manifest.json",
        {
            "modality": "text",
            "retrieval": mode,
            "baseline": "text_only_rag",
            "n_chunks": len(chunks),
            "dim": int(vectors.shape[1]),
            "doc_ids": sorted({c.doc_id for c in chunks}),
        },
    )
    logger.info("Built text-only %s index: %d chunks -> %s", mode, len(chunks), out_dir)
    return out_dir
