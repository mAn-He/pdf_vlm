"""Question-type inference helpers shared by dataset loaders."""

from __future__ import annotations

from typing import Any, Sequence

from pdf_vlm.schemas import QuestionType

_MODALITY_MAP = {
    "text": QuestionType.TEXT,
    "txt": QuestionType.TEXT,
    "table": QuestionType.TABLE,
    "tab": QuestionType.TABLE,
    "chart": QuestionType.CHART,
    "cha": QuestionType.CHART,
    "image": QuestionType.IMAGE,
    "img": QuestionType.IMAGE,
    "figure": QuestionType.IMAGE,
    "layout": QuestionType.LAYOUT,
    "lay": QuestionType.LAYOUT,
}


def normalize_modalities(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip().lower() for p in raw.replace("|", ",").split(",") if p.strip()]
        return parts
    if isinstance(raw, (list, tuple, set)):
        out: list[str] = []
        for item in raw:
            out.extend(normalize_modalities(item))
        return out
    return [str(raw).lower()]


def infer_question_type(
    *,
    evidence_pages: Sequence[int] | None = None,
    evidence_modality: Sequence[str] | None = None,
    unanswerable: bool = False,
    explicit: str | None = None,
) -> QuestionType:
    """Map dataset-specific fields to canonical QuestionType.

    Priority:
      1) unanswerable
      2) explicit question_type if valid
      3) cross-page if >=2 evidence pages
      4) first known evidence modality
      5) text (default)
    """
    if unanswerable:
        return QuestionType.UNANSWERABLE

    if explicit:
        key = explicit.strip().lower().replace("_", "-")
        for qt in QuestionType:
            if qt.value == key or qt.name.lower() == key:
                return qt
        if key in _MODALITY_MAP:
            return _MODALITY_MAP[key]

    pages = list(evidence_pages or [])
    if len(set(int(p) for p in pages)) >= 2:
        return QuestionType.CROSS_PAGE

    modalities = normalize_modalities(list(evidence_modality or []))
    for mod in modalities:
        if mod in _MODALITY_MAP:
            mapped = _MODALITY_MAP[mod]
            if mapped != QuestionType.TEXT or len(modalities) == 1:
                return mapped
    if modalities:
        # prefer non-text if mixed
        for mod in modalities:
            if mod in _MODALITY_MAP and _MODALITY_MAP[mod] != QuestionType.TEXT:
                return _MODALITY_MAP[mod]
        if modalities[0] in _MODALITY_MAP:
            return _MODALITY_MAP[modalities[0]]

    return QuestionType.TEXT


def as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if "," in value:
            return [int(x.strip()) for x in value.split(",") if x.strip()]
        return [int(value)]
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            out.extend(as_int_list(item))
        return out
    return [int(value)]
