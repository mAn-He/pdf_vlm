"""Exact match and token F1."""

from __future__ import annotations

import re
from typing import Sequence


def _tokens(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


def exact_match(prediction: str, gold_answers: Sequence[str]) -> float:
    pred = prediction.strip().lower()
    return 1.0 if any(pred == g.strip().lower() for g in gold_answers) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_toks = _tokens(prediction)
    gold_toks = _tokens(gold)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gold_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def best_token_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(token_f1(prediction, g) for g in gold_answers)
