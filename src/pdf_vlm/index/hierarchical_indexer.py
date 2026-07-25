"""Build hierarchical (section + fine) indexes with explicit hierarchy strategy."""

from __future__ import annotations

from pathlib import Path

from pdf_vlm.index.hierarchy import apply_hierarchy
from pdf_vlm.index.store import VectorStore
from pdf_vlm.index.text_embedder import TextEmbedder
from pdf_vlm.index.vision_embedder import VisionEmbedder
from pdf_vlm.schemas import Chunk, DocumentArtifact
from pdf_vlm.utils.io import ensure_dir, save_json
from pdf_vlm.utils.logging import get_logger

logger = get_logger("index.hierarchical")


def build_hierarchical_chunks(
    doc: DocumentArtifact,
    modality: str = "text",
    *,
    fine_unit: str = "page",
) -> dict[str, list[Chunk]]:
    """Create coarse section chunks and fine page/paragraph chunks.

    fine_unit:
      - page: one fine chunk per page (default, stable for long PDFs)
      - paragraph/block: prefer OCR blocks when available
    """
    section_chunks: list[Chunk] = []
    fine_chunks: list[Chunk] = []

    # Map page -> owning section id (first wins)
    page_to_section: dict[int, str] = {}
    for sec in doc.sections:
        for pid in sec.page_ids:
            page_to_section.setdefault(pid, sec.section_id)
        section_chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::section::{sec.section_id}",
                doc_id=doc.doc_id,
                page_ids=list(sec.page_ids),
                section_id=sec.section_id,
                level="section",
                text=sec.summary_text or sec.title,
                meta={"title": sec.title, "stage": "coarse"},
            )
        )

    if modality == "text" and fine_unit in {"paragraph", "block"}:
        for page in doc.pages:
            sid = page_to_section.get(page.page_id)
            if page.blocks:
                for block in page.blocks:
                    if not block.text.strip():
                        continue
                    fine_chunks.append(
                        Chunk(
                            chunk_id=f"{doc.doc_id}::block::{block.block_id}",
                            doc_id=doc.doc_id,
                            page_ids=[page.page_id],
                            section_id=block.section_id or sid,
                            level="paragraph",
                            text=block.text,
                            image_path=page.image_path if modality != "text" else None,
                            meta={"stage": "fine"},
                        )
                    )
            else:
                # paragraph split on blank lines
                paras = [p.strip() for p in (page.markdown or "").split("\n\n") if p.strip()]
                if not paras:
                    paras = [page.markdown or ""]
                for j, para in enumerate(paras):
                    fine_chunks.append(
                        Chunk(
                            chunk_id=f"{doc.doc_id}::para::{page.page_id}_{j}",
                            doc_id=doc.doc_id,
                            page_ids=[page.page_id],
                            section_id=sid,
                            level="paragraph",
                            text=para,
                            meta={"stage": "fine"},
                        )
                    )
    else:
        for page in doc.pages:
            sid = page_to_section.get(page.page_id)
            text = page.markdown or ""
            if modality != "text":
                text = text[:500]
            fine_chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::page::{page.page_id}",
                    doc_id=doc.doc_id,
                    page_ids=[page.page_id],
                    section_id=sid,
                    level="page",
                    text=text,
                    image_path=page.image_path if modality != "text" else None,
                    meta={"stage": "fine"},
                )
            )

    # Ensure at least one section
    if not section_chunks:
        section_chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::section::all",
                doc_id=doc.doc_id,
                page_ids=[p.page_id for p in doc.pages],
                section_id="all",
                level="section",
                text="\n\n".join(p.markdown or "" for p in doc.pages)[:4000],
                meta={"title": "Document", "stage": "coarse"},
            )
        )
    return {"section": section_chunks, "fine": fine_chunks}


def build_hierarchical_index(
    docs: list[DocumentArtifact],
    out_dir: str | Path,
    *,
    modality: str = "text",
    retrieval_cfg: dict | None = None,
    hierarchy_strategy: str = "auto",
    fine_unit: str = "page",
) -> Path:
    retrieval_cfg = retrieval_cfg or {}
    out_dir = ensure_dir(Path(out_dir))
    section_chunks: list[Chunk] = []
    fine_chunks: list[Chunk] = []
    strategies: dict[str, str] = {}

    prepared: list[DocumentArtifact] = []
    for doc in docs:
        doc2 = apply_hierarchy(doc, strategy=hierarchy_strategy)  # type: ignore[arg-type]
        strategies[doc2.doc_id] = str(doc2.meta.get("hierarchy_strategy"))
        prepared.append(doc2)
        parts = build_hierarchical_chunks(doc2, modality=modality, fine_unit=fine_unit)
        section_chunks.extend(parts["section"])
        fine_chunks.extend(parts["fine"])

    text_cfg = retrieval_cfg.get("embedder") or retrieval_cfg.get("text_embedder") or {}
    text_embedder = TextEmbedder(
        name=text_cfg.get("name", "BAAI/bge-m3"),
        device=text_cfg.get("device", "cpu"),
        normalize=bool(text_cfg.get("normalize", True)),
        dim=int(text_cfg.get("dim", 1024)),
        backend=text_cfg.get("backend"),
    )
    sec_vecs = text_embedder.embed([c.text or "" for c in section_chunks]) if section_chunks else None

    if modality == "text":
        fine_vecs = text_embedder.embed([c.text or "" for c in fine_chunks])
    else:
        vis_cfg = retrieval_cfg.get("vision_embedder") or retrieval_cfg.get("embedder") or {}
        vision = VisionEmbedder(
            name=vis_cfg.get("name", "google/siglip-base-patch16-224"),
            device=vis_cfg.get("device", "cpu"),
            normalize=bool(vis_cfg.get("normalize", True)),
        )
        paths = [c.image_path for c in fine_chunks]
        if any(not p or not Path(p).exists() for p in paths):
            raise FileNotFoundError("Missing page images for hierarchical multimodal index")
        fine_vecs = vision.embed_images(paths)  # type: ignore[arg-type]

    if sec_vecs is not None and section_chunks:
        sec_store = VectorStore(dim=sec_vecs.shape[1], metric="ip")
        sec_store.add([c.chunk_id for c in section_chunks], sec_vecs)
        sec_store.save(out_dir / "section")
        save_json(out_dir / "section_chunks.json", [c.model_dump(mode="json") for c in section_chunks])

    fine_store = VectorStore(dim=fine_vecs.shape[1], metric="ip")
    fine_store.add([c.chunk_id for c in fine_chunks], fine_vecs)
    fine_store.save(out_dir / "fine")
    save_json(out_dir / "fine_chunks.json", [c.model_dump(mode="json") for c in fine_chunks])
    save_json(
        out_dir / "manifest.json",
        {
            "modality": modality,
            "retrieval": "hierarchical",
            "fine_unit": fine_unit,
            "hierarchy_strategies": strategies,
            "n_section": len(section_chunks),
            "n_fine": len(fine_chunks),
        },
    )
    logger.info(
        "Built hierarchical index (%s): %d sections, %d fine -> %s strategies=%s",
        modality,
        len(section_chunks),
        len(fine_chunks),
        out_dir,
        strategies,
    )
    return out_dir
