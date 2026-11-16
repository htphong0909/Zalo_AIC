"""Golden parameter scorer and bipartite tool matcher inheriting lfm3_grpo_with_trl.ipynb and BTC metrics."""
from __future__ import annotations

import ast
import itertools
import math
import operator
import re
import unicodedata
from collections import Counter
from datetime import date, datetime, time
from difflib import SequenceMatcher
from typing import Any

from src.eval.metrics import fuzzy_params, token_f1

EMPTY = (None, "", [], {})

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def norm(s: Any) -> str:
    """Normalize text (NFKC, lowercase, strip Vietnamese & Latin diacritics and punctuation)."""
    s = unicodedata.normalize("NFKC", str(s)).casefold().replace("đ", "d")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in unicodedata.normalize("NFC", s))
    return " ".join(s.split())


def contained(a: str, b: str) -> bool:
    short, long_ = sorted((a, b), key=len)
    return bool(short) and short in long_ and len(short) / len(long_) >= 0.5


def name_equal(a: Any, b: Any) -> bool:
    na, nb = norm(a), norm(b)
    return na == nb or contained(na, nb)


def text_similar(a: Any, b: Any) -> bool:
    if token_f1(a, b) > 0.6:
        return True
    na, nb = norm(a), norm(b)
    return na == nb or contained(na, nb) or (bool(na) and bool(nb) and SequenceMatcher(None, na, nb).ratio() >= 0.6)


def parse_when(value: Any, kind: str) -> Any:
    raw = str(value).strip()
    try:
        if kind == "datetime":
            return datetime.fromisoformat(raw)
        if kind == "date":
            return date.fromisoformat(raw)
        if kind == "time":
            padded = raw.zfill(5) if len(raw) == 4 else raw
            return time.fromisoformat(padded)
    except Exception:
        return None
    return None


def calculate(expression: Any) -> float | None:
    """Safely evaluate an arithmetic expression, or return None."""

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            return float(OPERATORS[type(node.op)](walk(node.left), walk(node.right)))
        if isinstance(node, ast.UnaryOp):
            return float(OPERATORS[type(node.op)](walk(node.operand)))
        raise ValueError

    try:
        cleaned = str(expression).replace("^", "**").replace("×", "*").replace("÷", "/")
        return float(walk(ast.parse(cleaned, mode="eval").body))
    except Exception:
        return None


def same_list(gold: list[Any], pred: list[Any], equal: Any) -> bool:
    if len(gold) != len(pred):
        return False
    if len(gold) <= 6:
        return any(all(equal(g, p) for g, p in zip(gold, perm)) for perm in itertools.permutations(pred))
    return sorted(norm(x) for x in gold) == sorted(norm(x) for x in pred)


def infer_param_kind(prop_name: str, prop_spec: dict[str, Any] | None) -> str:
    spec = prop_spec or {}
    if spec.get("x-match") == "fuzzy":
        return "fuzzy"
    if "enum" in spec:
        return "enum"
    fmt = spec.get("format")
    if fmt == "date-time":
        return "datetime"
    if fmt in ("date", "time"):
        return str(fmt)
    ptype = spec.get("type")
    if ptype == "boolean":
        return "bool"
    if ptype in ("integer", "number"):
        return "number"
    if ptype == "array":
        return "list"
    if prop_name in ("phone", "phone_number") or spec.get("x-ref"):
        return "code"
    if prop_name in ("query", "name"):
        return "name"
    return "exact"


def same_value(kind: str, gold: Any, pred: Any) -> bool:
    try:
        if kind == "bool":
            return isinstance(pred, bool) and isinstance(gold, bool) and gold is pred
        if kind in ("int", "number"):
            return (
                isinstance(pred, (int, float))
                and not isinstance(pred, bool)
                and isinstance(gold, (int, float))
                and not isinstance(gold, bool)
                and abs(float(gold) - float(pred)) < 1e-6
            )
        if kind == "enum":
            return isinstance(pred, str) and str(gold).strip().lower() == pred.strip().lower()
        if kind == "code":
            return isinstance(pred, str) and re.sub(r"[\s\-().]", "", str(gold).casefold()) == re.sub(
                r"[\s\-().]", "", pred.casefold()
            )
        if kind in ("datetime", "date", "time"):
            if str(gold).strip() == str(pred).strip():
                return True
            pg, pp = parse_when(gold, kind), parse_when(pred, kind)
            return pg is not None and pp is not None and pg == pp
        if kind == "expr":
            g, p = calculate(gold), calculate(pred)
            return g is not None and p is not None and math.isclose(g, p, rel_tol=1e-9, abs_tol=1e-9)
        if kind == "name":
            return isinstance(pred, str) and (pred == gold or name_equal(gold, pred))
        if kind in ("fuzzy", "text"):
            return isinstance(pred, str) and (token_f1(pred, gold) > 0.6 or text_similar(gold, pred))
        if kind in ("list", "list_enum"):
            return isinstance(pred, list) and isinstance(gold, list) and sorted(str(x).lower() for x in gold) == sorted(
                str(x).lower() for x in pred
            )
        if kind == "list_name":
            return isinstance(pred, list) and isinstance(gold, list) and same_list(gold, [str(x) for x in pred], name_equal)
        if kind == "list_text":
            return isinstance(pred, list) and isinstance(gold, list) and same_list(gold, [str(x) for x in pred], text_similar)
        return pred == gold
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False


def _clean_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Drop empty optional values without injecting schema defaults (aligned with official src/eval/metrics.py)."""
    return {k: v for k, v in (arguments or {}).items() if v not in EMPTY}


def _official_param_match(pred_val: Any, gold_val: Any, *, is_fuzzy: bool, threshold: float = 0.6) -> bool:
    """Exact LifeMate parameter comparison aligned 100% with src/eval/metrics.py."""
    if is_fuzzy:
        return token_f1(pred_val, gold_val) > threshold
    if isinstance(pred_val, bool) or isinstance(gold_val, bool):
        return type(pred_val) is type(gold_val) and pred_val == gold_val
    return pred_val == gold_val


def canonical(
    name: str,
    arguments: dict[str, Any] | None,
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Clean arguments aligned with official LifeMate evaluation (no schema default injection)."""
    _ = (name, catalog)
    return _clean_args(arguments)


def same_call(
    gold: dict[str, Any],
    pred: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> bool:
    name = gold.get("name")
    if not isinstance(name, str) or name != pred.get("name"):
        return False
    g = _clean_args(gold.get("arguments"))
    p = _clean_args(pred.get("arguments"))
    if set(g) != set(p):
        return False
    fuzzy = fuzzy_params(catalog.get(name))
    for k, gv in g.items():
        if not _official_param_match(p[k], gv, is_fuzzy=(k in fuzzy)):
            return False
    return True


def _max_bipartite_matches(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> int:
    adj = [[j for j, g in enumerate(gold) if same_call(g, p, catalog)] for p in pred]
    owner = [-1] * len(gold)

    def assign(i: int, seen: set[int]) -> bool:
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] == -1 or assign(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(assign(i, set()) for i in range(len(pred)))


def calls_match(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> bool:
    """True when the prediction makes the exact gold calls, in any order."""
    if len(gold) != len(pred):
        return False
    if not gold:
        return True
    if any(call.get("name") not in catalog for call in pred):
        return False
    return _max_bipartite_matches(gold, pred, catalog) == len(gold)


def tool_name_f1(gold: list[dict[str, Any]], pred: list[dict[str, Any]]) -> float:
    """Multiset F1 over tool names as in lfm3_grpo_with_trl.ipynb."""
    gold_names = Counter(c.get("name") for c in gold if c.get("name"))
    pred_names = Counter(c.get("name") for c in pred if c.get("name"))
    if not gold_names and not pred_names:
        return 1.0
    overlap = sum((gold_names & pred_names).values())
    return 0.0 if overlap == 0 else 2 * overlap / (sum(gold_names.values()) + sum(pred_names.values()))


def _single_call_arg_f1(
    gold_call: dict[str, Any],
    pred_call: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> float:
    name = gold_call.get("name")
    if not isinstance(name, str) or name != pred_call.get("name"):
        return 0.0
    g = _clean_args(gold_call.get("arguments"))
    p = _clean_args(pred_call.get("arguments"))
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    fuzzy = fuzzy_params(catalog.get(name))
    matched = 0.0
    for k, gv in g.items():
        if k not in p:
            continue
        if _official_param_match(p[k], gv, is_fuzzy=(k in fuzzy)):
            matched += 1.0
        elif k in fuzzy:
            matched += 0.5 * token_f1(p[k], gv)
    return 0.0 if matched <= 0.0 else (2.0 * matched) / (len(g) + len(p))


def arg_overlap_f1(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> float:
    """Continuous argument-level F1 across gold and predicted calls."""
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0

    best_per_gold: list[float] = []
    for g in gold:
        candidates = [_single_call_arg_f1(g, p, catalog) for p in pred if p.get("name") == g.get("name")]
        best_per_gold.append(max(candidates) if candidates else 0.0)
    recall_arg = sum(best_per_gold) / len(gold)

    best_per_pred: list[float] = []
    for p in pred:
        candidates = [_single_call_arg_f1(g, p, catalog) for g in gold if g.get("name") == p.get("name")]
        best_per_pred.append(max(candidates) if candidates else 0.0)
    prec_arg = sum(best_per_pred) / len(pred)

    return 0.0 if (prec_arg + recall_arg) == 0 else (2 * prec_arg * recall_arg) / (prec_arg + recall_arg)
