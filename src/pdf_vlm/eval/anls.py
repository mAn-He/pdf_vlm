"""ANLS metric (DocVQA / MP-DocVQA style)."""

from __future__ import annotations

import re
from typing import Sequence


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def anls_pair(prediction: str, gold: str, threshold: float = 0.5) -> float:
    pred_n = _normalize(prediction)
    gold_n = _normalize(gold)
    if not pred_n and not gold_n:
        return 1.0
    if not pred_n or not gold_n:
        return 0.0
    dist = levenshtein(pred_n, gold_n)
    score = 1.0 - dist / max(len(pred_n), len(gold_n))
    return score if score >= threshold else 0.0


def anls(prediction: str, gold_answers: Sequence[str], threshold: float = 0.5) -> float:
    if not gold_answers:
        return 0.0
    return max(anls_pair(prediction, g, threshold=threshold) for g in gold_answers)
