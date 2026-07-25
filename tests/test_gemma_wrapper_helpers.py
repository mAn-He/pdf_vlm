"""Unit tests for Gemma wrapper helpers (no model weights required)."""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image

from pdf_vlm.llm.gemma_llama_cpp import _extract_text, _image_to_data_uri


def test_image_to_data_uri(tmp_path: Path):
    img_path = tmp_path / "t.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(img_path)
    uri = _image_to_data_uri(img_path)
    assert uri.startswith("data:image/png;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_extract_text_from_chat_completion():
    out = {
        "choices": [{"message": {"content": "  hello  "}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }
    text, pt, ct = _extract_text(out)
    assert text == "hello"
    assert pt == 3
    assert ct == 1
