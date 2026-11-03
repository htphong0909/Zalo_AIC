"""Local reimplementation of the LifeMate metric (see docs/overview/Evaluation.md)."""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

EMPTY = (None, "", [], {})


def _tokens(value: Any) -> list[str]:
    return unicodedata.normalize("NFC", str(value)).lower().split()


def token_f1(pred: Any, gold: Any) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p and not g:
        return 1.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def fuzzy_params(tool_spec: dict | None) -> set[str]:
    props = (tool_spec or {}).get("parameters", {}).get("properties", {})
    return {name for name, spec in props.items() if spec.get("x-match") == "fuzzy"}


def _clean(arguments: dict | None) -> dict:
    return {k: v for k, v in (arguments or {}).items() if v not in EMPTY}


def calls_match(pred: dict, gold: dict, catalog: dict[str, dict], threshold: float = 0.6) -> bool:
    if pred.get("name") != gold.get("name"):
        return False
    fuzzy = fuzzy_params(catalog.get(gold["name"]))
    p, g = _clean(pred.get("arguments")), _clean(gold.get("arguments"))
    if set(p) != set(g):
        return False
    for key, gold_value in g.items():
        if key in fuzzy:
            if token_f1(p[key], gold_value) <= threshold:
                return False
        elif p[key] != gold_value:
            return False
    return True


def count_matches(pred_calls: list[dict], gold_calls: list[dict], catalog: dict[str, dict]) -> int:
    """Maximum one-to-one matching (Kuhn's augmenting paths)."""
    adj = [[j for j, g in enumerate(gold_calls) if calls_match(p, g, catalog)] for p in pred_calls]
    owner = [-1] * len(gold_calls)

    def assign(i: int, seen: set[int]) -> bool:
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] == -1 or assign(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(assign(i, set()) for i in range(len(pred_calls)))


@dataclass(frozen=True)
class SampleResult:
    score: float
    state_match: bool
    matched: int
    unmatched_pred: int
    n_gold: int
    precision: float
    recall: float
    f1: float


def score_sample(
    pred_calls: list[dict],
    pred_hash: str,
    gold_calls: list[dict],
    gold_hash: str,
    catalog: dict[str, dict],
) -> SampleResult:
    state_match = pred_hash == gold_hash
    if not gold_calls:
        ok = float(not pred_calls)
        return SampleResult(ok, state_match, 0, len(pred_calls), 0, ok, ok, ok)
    matched = count_matches(pred_calls, gold_calls, catalog)
    unmatched = len(pred_calls) - matched
    precision = matched / len(pred_calls) if pred_calls else 0.0
    recall = matched / len(gold_calls)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return SampleResult(
        f1 if state_match else 0.0,
        state_match,
        matched,
        unmatched,
        len(gold_calls),
        precision,
        recall,
        f1,
    )
