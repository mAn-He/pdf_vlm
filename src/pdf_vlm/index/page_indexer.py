"""Page-level indexer for text and multimodal variants."""

from __future__ import annotations

from pathlib import Path

from pdf_vlm.index.store import VectorStore
from pdf_vlm.index.text_embedder import TextEmbedder
from pdf_vlm.index.vision_embedder import VisionEmbedder
from pdf_vlm.schemas import Chunk, DocumentArtifact
from pdf_vlm.utils.io import ensure_dir, save_json
from pdf_vlm.utils.logging import get_logger

logger = get_logger("index.page")


def build_page_chunks(doc: DocumentArtifact, modality: str = "text") -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in doc.pages:
        chunk_id = f"{doc.doc_id}::page::{page.page_id}"
        if modality == "text":
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    page_ids=[page.page_id],
                    level="page",
                    text=page.markdown,
                    image_path=page.image_path,
                )
            )
        else:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    page_ids=[page.page_id],
                    level="page",
                    text=page.markdown[:500],
                    image_path=page.image_path,
                )
            )
    return chunks


def build_page_index(
    docs: list[DocumentArtifact],
    out_dir: str | Path,
    *,
    modality: str = "text",
    retrieval_cfg: dict | None = None,
) -> Path:
    retrieval_cfg = retrieval_cfg or {}
    out_dir = ensure_dir(Path(out_dir))
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(build_page_chunks(doc, modality=modality))

    if modality == "text":
        emb_cfg = retrieval_cfg.get("embedder", {})
        embedder = TextEmbedder(
            name=emb_cfg.get("name", "BAAI/bge-m3"),
            device=emb_cfg.get("device", "cpu"),
            normalize=bool(emb_cfg.get("normalize", True)),
            dim=int(emb_cfg.get("dim", 1024)),
            backend=emb_cfg.get("backend"),
        )
        texts = [c.text or "" for c in chunks]
        vectors = embedder.embed(texts)
    else:
        emb_cfg = retrieval_cfg.get("embedder", {})
        embedder = VisionEmbedder(
            name=emb_cfg.get("name", "google/siglip-base-patch16-224"),
            device=emb_cfg.get("device", "cpu"),
            normalize=bool(emb_cfg.get("normalize", True)),
        )
        paths = []
        for c in chunks:
            if not c.image_path or not Path(c.image_path).exists():
                raise FileNotFoundError(f"Missing page image for chunk {c.chunk_id}")
            paths.append(c.image_path)
        vectors = embedder.embed_images(paths)

    store = VectorStore(dim=vectors.shape[1], metric=retrieval_cfg.get("index", {}).get("metric", "ip"))
    store.add([c.chunk_id for c in chunks], vectors)
    store.save(out_dir)
    save_json(out_dir / "chunks.json", [c.model_dump(mode="json") for c in chunks])
    save_json(
        out_dir / "manifest.json",
        {"modality": modality, "retrieval": "page", "n_chunks": len(chunks), "dim": int(vectors.shape[1])},
    )
    logger.info("Built page index (%s) with %d chunks -> %s", modality, len(chunks), out_dir)
    return out_dir
