"""Section-level text retriever (strict OCR text)."""

from __future__ import annotations

from pathlib import Path

from pdf_vlm.index.store import VectorStore
from pdf_vlm.index.text_embedder import TextEmbedder
from pdf_vlm.schemas import Chunk, RetrievalHit
from pdf_vlm.utils.io import load_json
from pdf_vlm.utils.logging import get_logger

logger = get_logger("retrieve.section")


class SectionRetriever:
    """Retrieve section text chunks; page_ids come from section membership."""

    def __init__(self, index_dir: str | Path, device: str = "cpu"):
        self.index_dir = Path(index_dir)
        manifest = load_json(self.index_dir / "manifest.json")
        if manifest.get("modality") != "text":
            raise ValueError(f"SectionRetriever expects text modality index at {self.index_dir}")
        if manifest.get("retrieval") not in {"section", "hier_text", "hierarchical"}:
            # Allow plain section index; also tolerate older hierarchical section store
            pass
        self.store = VectorStore.load(self.index_dir)
        self.chunks = {
            c["chunk_id"]: Chunk.model_validate(c) for c in load_json(self.index_dir / "chunks.json")
        }
        emb_name = (manifest.get("embedder") or {}).get("name", "BAAI/bge-m3")
        self.embedder = TextEmbedder(name=emb_name, device=device)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        q = self.embedder.embed([query])[0]
        hits_raw = self.store.search(q, top_k=top_k)
        hits: list[RetrievalHit] = []
        for chunk_id, score in hits_raw:
            chunk = self.chunks[chunk_id]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    score=float(score),
                    doc_id=chunk.doc_id,
                    page_ids=list(chunk.page_ids),
                    section_id=chunk.section_id,
                    level="section",
                    text=chunk.text,
                    image_path=None,
                    meta={"retrieval": "section", "modality": "text"},
                )
            )
        return hits
