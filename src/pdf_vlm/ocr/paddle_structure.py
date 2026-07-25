"""PaddleOCR PP-StructureV3 document parsing pipeline.

Prefers real PaddleOCR when installed. Stub OCR is opt-in only (--stub).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# Mitigate PaddlePaddle 3.3.x CPU oneDNN PIR crash on Windows:
#   ConvertPirAttribute2RuntimeAttribute not support ArrayAttribute<DoubleAttribute>
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_onnxruntime", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from pdf_vlm.ocr.normalize import normalize_document
from pdf_vlm.pdf.render import render_pdf_pages
from pdf_vlm.schemas import DocumentArtifact
from pdf_vlm.utils.io import ensure_dir, load_model, resolve_path, save_json, save_model
from pdf_vlm.utils.logging import get_logger

logger = get_logger("ocr.paddle")


def _maybe_disable_onednn() -> None:
    try:
        import paddle

        paddle.set_flags({"FLAGS_use_mkldnn": False})
    except Exception:
        pass


def _doc_id_from_path(pdf_path: Path) -> str:
    from pdf_vlm.utils.io import stable_doc_id

    return stable_doc_id(pdf_path)


def _artifact_paths(processed_dir: Path, doc_id: str) -> dict[str, Path]:
    root = processed_dir / doc_id
    return {
        "root": root,
        "pages": root / "pages",
        "raw": root / "ocr_raw.json",
        "artifact": root / "document.json",
        "manifest": root / "manifest.json",
    }


def paddle_available() -> dict[str, bool]:
    """Probe installed PaddleOCR entry points."""
    flags = {"paddleocr": False, "PPStructureV3": False, "PPStructure": False, "PaddleOCR": False}
    try:
        import paddleocr  # noqa: F401

        flags["paddleocr"] = True
    except ImportError:
        return flags
    try:
        from paddleocr import PPStructureV3  # noqa: F401

        flags["PPStructureV3"] = True
    except Exception:
        pass
    try:
        from paddleocr import PPStructure  # noqa: F401

        flags["PPStructure"] = True
    except Exception:
        pass
    try:
        from paddleocr import PaddleOCR  # noqa: F401

        flags["PaddleOCR"] = True
    except Exception:
        pass
    return flags


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    """Convert PPStructureV3 result objects / nested structures to JSON-ish dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # PaddleX / PaddleOCR 3.x result objects
    for attr in ("json", "res", "data"):
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if callable(val):
                try:
                    val = val()
                except TypeError:
                    pass
            if isinstance(val, dict):
                return val
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    return {"raw": val}
    # markdown helpers
    md = None
    if hasattr(obj, "markdown"):
        md = obj.markdown
        if callable(md):
            try:
                md = md()
            except TypeError:
                pass
    if isinstance(md, dict):
        out = {"markdown": md.get("markdown") or md.get("text") or "", "markdown_obj": md}
        return out
    if isinstance(md, str) and md.strip():
        return {"markdown": md}

    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            d = obj.to_dict()
            if isinstance(d, dict):
                return d
        except Exception:
            pass

    # last resort: str
    return {"raw": str(obj)}


def _extract_ocr_lines_from_structure(res_list: list[Any]) -> list[str]:
    lines: list[str] = []
    for item in res_list:
        if not isinstance(item, dict):
            continue
        # classic PPStructure: {'type': 'text', 'res': [[box, (text, conf)], ...]}
        inner = item.get("res")
        if isinstance(inner, list):
            for row in inner:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    txt = row[1][0] if isinstance(row[1], (list, tuple)) else row[1]
                    if txt:
                        lines.append(str(txt))
        text = item.get("text") or item.get("content")
        if isinstance(text, str) and text.strip():
            lines.append(text.strip())
        elif isinstance(text, list):
            lines.extend(str(t) for t in text if t)
    return lines


def _normalize_engine_page_result(raw: Any, source_image: str) -> dict[str, Any]:
    """Unify one page OCR/structure result for normalize_document()."""
    d = _to_plain_dict(raw)
    d.setdefault("source_image", source_image)

    # Flatten nested 'res' wrapper from some paddle builds
    if "res" in d and isinstance(d["res"], dict) and "parsing_res_list" not in d:
        nested = d["res"]
        for k, v in nested.items():
            d.setdefault(k, v)

    # Collect text lines for fallback
    ocr_texts: list[str] = []
    if isinstance(d.get("ocr_texts"), list):
        ocr_texts.extend(str(x) for x in d["ocr_texts"] if x)
    for key in ("rec_texts", "texts"):
        if isinstance(d.get(key), list):
            ocr_texts.extend(str(x) for x in d[key] if x)
    overall = d.get("overall_ocr_res") or d.get("ocr_result")
    if isinstance(overall, dict):
        for key in ("rec_texts", "texts"):
            if isinstance(overall.get(key), list):
                ocr_texts.extend(str(x) for x in overall[key] if x)

    res_list = d.get("parsing_res_list") or d.get("layout") or d.get("res")
    if isinstance(res_list, list):
        ocr_texts.extend(_extract_ocr_lines_from_structure(res_list))
        d["parsing_res_list"] = res_list

    # markdown may be nested
    md = d.get("markdown")
    if isinstance(md, dict):
        d["markdown"] = md.get("markdown") or md.get("text") or ""
        if not ocr_texts and isinstance(md.get("text"), str):
            ocr_texts.append(md["text"])

    if ocr_texts and "ocr_texts" not in d:
        # dedupe preserve order
        seen: set[str] = set()
        uniq: list[str] = []
        for t in ocr_texts:
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                uniq.append(t)
        d["ocr_texts"] = uniq

    if not d.get("markdown") and d.get("ocr_texts"):
        d["markdown"] = "\n".join(d["ocr_texts"])

    return d


def _structure_v3_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build a memory-friendly PPStructureV3 constructor kwargs from YAML."""
    keys = [
        "lang",
        "ocr_version",
        "use_doc_orientation_classify",
        "use_doc_unwarping",
        "use_textline_orientation",
        "use_seal_recognition",
        "use_table_recognition",
        "use_formula_recognition",
        "use_chart_recognition",
        "use_region_detection",
        "layout_detection_model_name",
        "text_detection_model_name",
        "text_recognition_model_name",
        "text_det_limit_side_len",
    ]
    kwargs: dict[str, Any] = {}
    for k in keys:
        if k in cfg and cfg[k] is not None:
            kwargs[k] = cfg[k]
    # Sensible defaults for local machines if not set
    kwargs.setdefault("use_doc_orientation_classify", False)
    kwargs.setdefault("use_doc_unwarping", False)
    kwargs.setdefault("use_textline_orientation", False)
    kwargs.setdefault("use_seal_recognition", False)
    kwargs.setdefault("use_formula_recognition", False)
    kwargs.setdefault("use_chart_recognition", False)
    kwargs.setdefault("use_region_detection", False)
    kwargs.setdefault("use_table_recognition", False)
    kwargs.setdefault("text_det_limit_side_len", 960)
    return kwargs


def _run_pp_structure_v3(image_paths: list[str], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    import gc

    from paddleocr import PPStructureV3

    base = _structure_v3_kwargs(cfg)
    use_gpu = bool(cfg.get("use_gpu", False))
    # Prefer GPU when requested; CPU oneDNN on paddle 3.3.x is buggy on Windows
    devices: list[str] = []
    if use_gpu:
        devices.extend(["gpu:0", "gpu"])
    devices.append("cpu")

    engine = None
    last_err: Exception | None = None
    attempt_list: list[dict[str, Any]] = []
    for device in devices:
        attempt_list.append({**base, "device": device})
    attempt_list.append({**base})
    attempt_list.append(
        {
            "lang": base.get("lang", "korean"),
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_seal_recognition": False,
            "use_formula_recognition": False,
            "use_chart_recognition": False,
            "use_table_recognition": bool(base.get("use_table_recognition", False)),
            "device": devices[0],
        }
    )
    attempt_list.append({})

    for attempt in attempt_list:
        try:
            engine = PPStructureV3(**attempt)
            logger.info("OCR backend=PPStructureV3 init_kwargs=%s", attempt)
            break
        except TypeError as exc:
            last_err = exc
            continue
        except Exception as exc:
            last_err = exc
            logger.warning("PPStructureV3 init failed with %s: %s", attempt, exc)
            continue
    if engine is None:
        raise RuntimeError(f"Could not construct PPStructureV3 ({last_err})")

    results: list[dict[str, Any]] = []
    for i, path in enumerate(image_paths):
        logger.info("PPStructureV3 page %d/%d: %s", i + 1, len(image_paths), Path(path).name)
        out = engine.predict(input=path)
        if isinstance(out, list) and out:
            page = _normalize_engine_page_result(out[0], path)
        else:
            page = _normalize_engine_page_result(out, path)
        page["backend"] = "PPStructureV3"
        results.append(page)
        gc.collect()
    return results


def _run_pp_structure_legacy(image_paths: list[str], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from paddleocr import PPStructure

    lang = cfg.get("lang", "ch")
    # map korean -> korean / korean for older API
    engine = PPStructure(
        lang=lang,
        show_log=False,
        use_gpu=bool(cfg.get("use_gpu", False)),
    )
    logger.info("OCR backend=PPStructure (legacy) lang=%s", lang)
    results: list[dict[str, Any]] = []
    for path in image_paths:
        raw = engine(path)
        # typically list of region dicts
        page = _normalize_engine_page_result({"res": raw}, path)
        page["backend"] = "PPStructure"
        results.append(page)
    return results


def _run_paddleocr_plain(image_paths: list[str], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Last-resort: plain OCR lines (no layout) when Structure APIs missing."""
    from paddleocr import PaddleOCR

    lang = cfg.get("lang", "korean")
    try:
        engine = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        try:
            engine = PaddleOCR(lang=lang, show_log=False, use_gpu=bool(cfg.get("use_gpu", False)))
        except TypeError:
            engine = PaddleOCR(lang=lang)

    logger.info("OCR backend=PaddleOCR (plain text) lang=%s", lang)
    results: list[dict[str, Any]] = []
    for path in image_paths:
        out = None
        if hasattr(engine, "predict"):
            out = engine.predict(input=path)
        else:
            out = engine.ocr(path)

        page: dict[str, Any]
        if isinstance(out, list) and out and not isinstance(out[0], list):
            # 3.x result objects
            page = _normalize_engine_page_result(out[0], path)
        elif isinstance(out, list):
            # classic: [[box, (text, conf)], ...] or [page_results]
            lines: list[str] = []
            rows = out[0] if out and isinstance(out[0], list) and out[0] and isinstance(out[0][0], list) else out
            for row in rows or []:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    txt = row[1][0] if isinstance(row[1], (list, tuple)) else row[1]
                    if txt:
                        lines.append(str(txt))
            page = {"ocr_texts": lines, "markdown": "\n".join(lines), "source_image": path}
        else:
            page = _normalize_engine_page_result(out, path)
        page["backend"] = "PaddleOCR"
        results.append(page)
    return results


def _pdf_text_page(pdf_path: Path, page_id: int, source_image: str) -> dict[str, Any]:
    """Born-digital fallback: extract embedded PDF text for one page."""
    import fitz

    doc = fitz.open(pdf_path)
    try:
        if page_id < 0 or page_id >= len(doc):
            text = ""
        else:
            text = doc[page_id].get_text("text") or ""
    finally:
        doc.close()
    text = text.strip()
    return {
        "markdown": text,
        "ocr_texts": [ln for ln in text.splitlines() if ln.strip()],
        "source_image": source_image,
        "backend": "pdf_text",
        "stub": False,
    }


def _run_paddle_subprocess(
    image_path: str,
    *,
    lang: str = "korean",
    backend: str = "ocr",
    timeout_s: int = 180,
) -> dict[str, Any]:
    """Run OCR in a child process so native segfaults don't kill ingest."""
    import subprocess
    import tempfile

    from pdf_vlm.utils.io import project_root

    worker = project_root() / "scripts" / "_paddle_ocr_worker.py"
    if not worker.exists():
        raise FileNotFoundError(f"OCR worker missing: {worker}")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "page.json"
        cmd = [
            sys.executable,
            str(worker),
            "--image",
            image_path,
            "--out",
            str(out),
            "--lang",
            lang,
            "--backend",
            backend,
        ]
        env = os.environ.copy()
        env["FLAGS_use_mkldnn"] = "0"
        env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        if proc.returncode != 0 or not out.exists():
            err = (proc.stderr or proc.stdout or "")[-500:]
            raise RuntimeError(f"paddle worker exit={proc.returncode}: {err}")
        return json.loads(out.read_text(encoding="utf-8"))


def run_paddle_ocr(
    image_paths: list[str],
    cfg: dict[str, Any] | None = None,
    *,
    pdf_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run best available OCR. Isolates Paddle in subprocess; PDF-text fallback per page."""
    cfg = cfg or {}
    _maybe_disable_onednn()
    flags = paddle_available()
    lang = str(cfg.get("lang") or "korean")
    prefer = str(cfg.get("pipeline") or "PP-StructureV3").lower()
    allow_pdf_fallback = bool(cfg.get("allow_pdf_text_fallback", True))
    use_subprocess = bool(cfg.get("paddle_subprocess", True))

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    paddle_fail_streak = 0
    paddle_disabled = False
    max_fail_before_skip = int(cfg.get("paddle_fail_fast_after", 2))

    for i, path in enumerate(image_paths):
        page: dict[str, Any] | None = None
        if flags.get("paddleocr") and use_subprocess and not paddle_disabled:
            backends = ["structurev3", "ocr"] if "structure" in prefer else ["ocr", "structurev3"]
            for b in backends:
                try:
                    logger.info("Paddle subprocess %s page %d/%d", b, i + 1, len(image_paths))
                    page = _run_paddle_subprocess(path, lang=lang, backend=b)
                    page = _normalize_engine_page_result(page, path)
                    paddle_fail_streak = 0
                    break
                except Exception as exc:
                    errors.append(f"p{i}/{b}: {exc}")
                    logger.warning("Paddle %s failed on page %d: %s", b, i, exc)
                    page = None
            if page is None:
                paddle_fail_streak += 1
                if paddle_fail_streak >= max_fail_before_skip:
                    paddle_disabled = True
                    logger.warning(
                        "Paddle OCR failed %d pages in a row — skipping Paddle for remaining pages "
                        "(likely broken torch/DLL). Falling back to PDF text.",
                        paddle_fail_streak,
                    )

        if page is None and flags.get("paddleocr") and not use_subprocess and not paddle_disabled:
            try:
                batch = _run_pp_structure_v3([path], cfg) if "structure" in prefer else _run_paddleocr_plain([path], cfg)
                page = batch[0]
                paddle_fail_streak = 0
            except Exception as exc:
                errors.append(f"p{i}/inproc: {exc}")
                page = None

        if (page is None or not (page.get("markdown") or page.get("ocr_texts"))) and allow_pdf_fallback and pdf_path:
            if not paddle_disabled:
                logger.warning("Using PDF text fallback for page %d", i)
            page = _pdf_text_page(Path(pdf_path), i, path)

        if page is None:
            page = {"markdown": "", "ocr_texts": [], "source_image": path, "backend": "empty"}

        results.append(page)

    if not any((r.get("markdown") or r.get("ocr_texts")) for r in results):
        raise RuntimeError(
            "OCR produced empty text for all pages. "
            f"paddle={flags} errors={errors[:3]}\n"
            'Install deps: python -m pip install "paddlex[ocr]==3.7.2"\n'
            "Known Windows issue: paddle 3.3.x oneDNN / native crash — "
            "we fall back to PDF text when allow_pdf_text_fallback=true."
        )
    return results


# Backward-compatible name used internally
def _run_pp_structure(image_paths: list[str], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return run_paddle_ocr(image_paths, cfg)


def _stub_ocr_from_images(image_paths: list[str]) -> list[dict[str, Any]]:
    """Deterministic stub for offline unit tests without PaddleOCR."""
    results = []
    for i, path in enumerate(image_paths):
        results.append(
            {
                "markdown": f"Stub OCR text for page {i} from {Path(path).name}.",
                "ocr_texts": [f"Stub OCR text for page {i}."],
                "source_image": path,
                "stub": True,
                "backend": "stub",
            }
        )
    return results


def ingest_pdf(
    pdf_path: str | Path,
    *,
    ocr_cfg: dict[str, Any] | None = None,
    processed_dir: str | Path | None = None,
    doc_id: str | None = None,
    force: bool = False,
    use_stub: bool = False,
) -> DocumentArtifact:
    """Render PDF, run PaddleOCR PP-StructureV3 (default), cache under data/processed."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    ocr_cfg = dict(ocr_cfg or {})
    processed_dir = Path(processed_dir or resolve_path("data/processed"))
    doc_id = doc_id or _doc_id_from_path(pdf_path)
    paths = _artifact_paths(processed_dir, doc_id)

    if not force and paths["artifact"].exists() and ocr_cfg.get("cache", True):
        # Refuse silent reuse of stub artifacts when caller wants real OCR
        try:
            cached = load_model(paths["artifact"], DocumentArtifact)
            was_stub = bool((cached.meta or {}).get("stub"))
            if was_stub and not use_stub:
                logger.warning(
                    "Cached artifact is stub OCR (%s); re-running with PaddleOCR (pass force=True to be explicit)",
                    doc_id,
                )
            else:
                logger.info("Cache hit: %s (stub=%s)", paths["artifact"], was_stub)
                return cached
        except Exception:
            pass

    ensure_dir(paths["pages"])
    dpi = int(ocr_cfg.get("dpi", 200))
    page_metas = render_pdf_pages(pdf_path, paths["pages"], dpi=dpi)
    image_paths = [p["image_path"] for p in page_metas]

    backend = "stub"
    if use_stub:
        logger.warning("Using stub OCR (explicit --stub / use_stub=True)")
        raw_results = _stub_ocr_from_images(image_paths)
    else:
        if not paddle_available().get("paddleocr"):
            raise ImportError(
                "PaddleOCR is not installed in this Python environment.\n"
                "Activate pdf_vlm env and install: python -m pip install paddlepaddle paddleocr "
                "\"paddlex[ocr]==3.7.2\"\n"
                "Or pass --stub for placeholder OCR."
            )
        raw_results = run_paddle_ocr(image_paths, ocr_cfg, pdf_path=pdf_path)
        backends = [str((r or {}).get("backend") or "?") for r in raw_results]
        backend = ",".join(sorted(set(backends)))

    while len(raw_results) < len(page_metas):
        raw_results.append({"markdown": "", "ocr_texts": []})

    save_json(paths["raw"], raw_results)
    artifact = normalize_document(
        doc_id=doc_id,
        source_path=str(pdf_path),
        page_metas=page_metas,
        page_raw_results=raw_results,
        meta={
            "ocr_pipeline": ocr_cfg.get("pipeline", "PP-StructureV3"),
            "dpi": dpi,
            "stub": bool(use_stub),
            "ocr_backend": backend,
            "lang": ocr_cfg.get("lang"),
        },
    )
    save_model(paths["artifact"], artifact)
    save_json(
        paths["manifest"],
        {
            "doc_id": doc_id,
            "source_path": str(pdf_path),
            "num_pages": artifact.num_pages,
            "artifact_path": str(paths["artifact"]),
            "stub": bool(use_stub),
            "ocr_backend": backend,
        },
    )
    logger.info(
        "Ingested %s (%d pages, backend=%s stub=%s) -> %s",
        doc_id,
        artifact.num_pages,
        backend,
        use_stub,
        paths["artifact"],
    )
    return artifact


def load_artifact(doc_id: str, processed_dir: str | Path | None = None) -> DocumentArtifact:
    processed_dir = Path(processed_dir or resolve_path("data/processed"))
    path = processed_dir / doc_id / "document.json"
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return load_model(path, DocumentArtifact)
