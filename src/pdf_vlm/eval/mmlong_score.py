"""MMLongBench-Doc style scoring helpers (rule-based short-answer)."""

from __future__ import annotations

import re
from typing import Sequence

from pdf_vlm.eval.anls import anls
from pdf_vlm.eval.em_f1 import exact_match


_EXACT_PATTERNS = [
    re.compile(r"^[\w.+-]+@[\w.-]+\.\w+$"),  # email
    re.compile(r"^https?://"),  # url
    re.compile(r"^\+?\d[\d\s\-()]{5,}$"),  # phone-ish
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),  # date
]


def needs_exact_match(gold: str) -> bool:
    g = gold.strip()
    return any(p.search(g) for p in _EXACT_PATTERNS)


def score_answer(
    prediction: str,
    gold_answers: Sequence[str],
    *,
    unanswerable: bool = False,
    anls_threshold: float = 0.5,
) -> float:
    """Return 0/1 style score for one example (generalized accuracy component)."""
    pred = prediction.strip()
    pred_unans = pred.lower() in {"unanswerable", "none", "n/a", "unknown", ""}

    if unanswerable:
        return 1.0 if pred_unans else 0.0
    if pred_unans:
        return 0.0
    if not gold_answers:
        return 0.0

    # Prefer exact for identifier-like golds
    if any(needs_exact_match(g) for g in gold_answers):
        return exact_match(pred, gold_answers)
    return 1.0 if anls(pred, gold_answers, threshold=anls_threshold) >= anls_threshold else 0.0


def generalized_acc_f1(scores: list[float], unanswerable_flags: list[bool]) -> dict[str, float]:
    """Compute Acc and F1 balancing answerable / unanswerable like MMLongBench-Doc."""
    if not scores:
        return {"accuracy": 0.0, "f1": 0.0, "n": 0}

    acc = sum(scores) / len(scores)

    # Treat unanswerable as negative class for F1 over "answered correctly when answerable"
    # Simplified: binary correctness F1 with positive = answerable examples predicted correctly
    tp = fp = fn = 0
    for score, una in zip(scores, unanswerable_flags, strict=False):
        if not una:
            if score >= 1.0:
                tp += 1
            else:
                fn += 1
        else:
            if score < 1.0:
                fp += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall, "n": len(scores)}
