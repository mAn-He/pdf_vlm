"""Index + retrieve smoke with hashing/histogram fallbacks."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from pdf_vlm.index.hierarchical_indexer import build_hierarchical_index
from pdf_vlm.index.page_indexer import build_page_index
from pdf_vlm.ocr.normalize import normalize_document
from pdf_vlm.retrieve.hierarchical_retriever import HierarchicalRetriever
from pdf_vlm.retrieve.page_retriever import PageRetriever


def _make_doc(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    metas = []
    raws = []
    for i, text in enumerate(["Alpha page about cats", "Beta page about dogs", "Gamma page about birds"]):
        img = pages_dir / f"page_{i}.png"
        Image.new("RGB", (64, 64), color=(i * 40, 80, 120)).save(img)
        metas.append({"page_id": i, "image_path": str(img), "width": 64, "height": 64})
        raws.append({"markdown": text, "ocr_texts": [text]})
    return normalize_document("toy", "toy.pdf", metas, raws)


def test_page_text_index_retrieve(tmp_path: Path):
    doc = _make_doc(tmp_path)
    out = tmp_path / "idx_page_text"
    build_page_index([doc], out, modality="text", retrieval_cfg={"embedder": {"device": "cpu"}})
    ret = PageRetriever(out, modality="text", device="cpu")
    hits = ret.retrieve("dogs", top_k=1)
    assert hits
    assert 1 in hits[0].page_ids


def test_hierarchical_text_index_retrieve(tmp_path: Path):
    doc = _make_doc(tmp_path)
    out = tmp_path / "idx_hier_text"
    build_hierarchical_index([doc], out, modality="text", retrieval_cfg={"embedder": {"device": "cpu"}})
    ret = HierarchicalRetriever(out, modality="text", device="cpu")
    hits = ret.retrieve("birds", top_k=2)
    assert hits
