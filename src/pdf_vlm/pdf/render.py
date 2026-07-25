"""PDF page rendering via PyMuPDF."""

from __future__ import annotations

from pathlib import Path

from pdf_vlm.utils.io import ensure_dir
from pdf_vlm.utils.logging import get_logger

logger = get_logger("pdf.render")


def render_pdf_pages(
    pdf_path: str | Path,
    out_dir: str | Path,
    dpi: int = 200,
    image_format: str = "png",
) -> list[dict]:
    """Render each PDF page to an image. Returns list of page metadata dicts."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("pymupdf is required: pip install pymupdf") from exc

    pdf_path = Path(pdf_path)
    out_dir = ensure_dir(Path(out_dir))
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[dict] = []

    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image_name = f"page_{page_idx:04d}.{image_format}"
        image_path = out_dir / image_name
        pix.save(str(image_path))
        pages.append(
            {
                "page_id": page_idx,
                "image_path": str(image_path),
                "width": pix.width,
                "height": pix.height,
            }
        )
        logger.debug("Rendered page %s -> %s", page_idx, image_path)

    doc.close()
    logger.info("Rendered %d pages from %s", len(pages), pdf_path.name)
    return pages


def count_pdf_pages(pdf_path: str | Path) -> int:
    import fitz

    doc = fitz.open(pdf_path)
    n = len(doc)
    doc.close()
    return n
