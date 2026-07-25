"""Logging setup."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("pdf_vlm")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Prefer UTF-8 on Windows consoles to avoid cp949 encode errors
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    base = logging.getLogger("pdf_vlm")
    if name:
        return base.getChild(name)
    return base
