"""pdf-vlm CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from pdf_vlm.utils.io import load_config, load_named_config, project_root, resolve_path
from pdf_vlm.utils.logging import setup_logging

app = typer.Typer(help="Local document QA experiments (Gemma 3 + PaddleOCR)")
setup_logging()


@app.callback()
def main() -> None:
    """Local Document QA toolkit."""


@app.command("info")
def info() -> None:
    """Show project paths and default config summary."""
    cfg = load_config()
    rprint(
        {
            "project_root": str(project_root()),
            "dataset": cfg.get("dataset"),
            "modality": cfg.get("modality"),
            "retrieval": cfg.get("retrieval"),
            "top_k": cfg.get("top_k"),
        }
    )


@app.command("ingest")
def ingest(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False),
    stub: bool = typer.Option(False, help="Use stub OCR without PaddleOCR"),
    force: bool = typer.Option(False, help="Ignore cache"),
) -> None:
    """Ingest a PDF into data/processed via PP-StructureV3."""
    from pdf_vlm.ocr.paddle_structure import ingest_pdf

    ocr_cfg = load_named_config("ocr/pp_structure_v3.yaml")
    art = ingest_pdf(pdf, ocr_cfg=ocr_cfg, force=force, use_stub=stub)
    rprint({"doc_id": art.doc_id, "num_pages": art.num_pages, "sections": len(art.sections)})


@app.command("build-index")
def build_index_cmd(
    doc_id: str = typer.Option(..., help="Processed document id"),
    modality: str = typer.Option("text"),
    retrieval: str = typer.Option("page"),
) -> None:
    """Build FAISS/numpy index for a processed document."""
    from pdf_vlm.index.hierarchical_indexer import build_hierarchical_index
    from pdf_vlm.index.page_indexer import build_page_index
    from pdf_vlm.ocr.paddle_structure import load_artifact

    art = load_artifact(doc_id)
    variant = f"{'page' if retrieval == 'page' else 'hier'}_{'text' if modality == 'text' else 'mm'}"
    cfg = load_named_config(f"retrieval/{variant}.yaml")
    out = resolve_path(f"indices/{doc_id}/{variant}")
    if retrieval == "page":
        build_page_index([art], out, modality=modality, retrieval_cfg=cfg)
    else:
        build_hierarchical_index([art], out, modality=modality, retrieval_cfg=cfg)
    rprint({"index": str(out), "variant": variant})


@app.command("qa")
def qa(
    doc_id: str = typer.Option(...),
    question: str = typer.Option(...),
    modality: str = typer.Option("text"),
    retrieval: str = typer.Option("page"),
    top_k: int = typer.Option(3),
    dry_run: bool = typer.Option(False, help="Retrieve only; skip LLM"),
) -> None:
    """Run a single question against an indexed document."""
    from pdf_vlm.llm.gemma_llama_cpp import build_llm
    from pdf_vlm.rag.pipeline import RAGPipeline, build_retriever
    from pdf_vlm.schemas import QAExample

    variant = f"{'page' if retrieval == 'page' else 'hier'}_{'text' if modality == 'text' else 'mm'}"
    index_dir = resolve_path(f"indices/{doc_id}/{variant}")
    ret_cfg = load_named_config(f"retrieval/{variant}.yaml")
    retriever = build_retriever(index_dir, retrieval=retrieval, modality=modality, retrieval_cfg=ret_cfg)

    example = QAExample(example_id="cli", doc_id=doc_id, question=question)
    if dry_run:
        hits = retriever.retrieve(question, top_k=top_k)
        rprint([{"pages": h.page_ids, "score": h.score, "text": (h.text or "")[:200]} for h in hits])
        return

    model_cfg = load_named_config("models/gemma3_4b_qat.yaml")
    model_cfg["local_path"] = str(resolve_path(model_cfg["local_path"]))
    llm = build_llm(model_cfg)
    pipe = RAGPipeline(llm, retriever, modality=modality, top_k=top_k)
    pred = pipe.answer(example)
    rprint({"answer": pred.prediction, "pages": pred.retrieved_page_ids, "latency_ms": pred.e2e_latency_ms})


if __name__ == "__main__":
    app()
