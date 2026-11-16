"""Fine-grained failure mode and tool/parameter mismatch diagnostics for error analysis."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.core.schemas import ASK_MISSING_INFO, Sample
from src.eval.metrics import SampleResult, _clean, calls_match, fuzzy_params, token_f1


def analyze_errors(
    predictions: dict[str, dict[str, Any]],
    samples: list[Sample],
    results: dict[str, SampleResult],
    catalog: dict[str, dict[str, Any]],
    *,
    max_examples_per_param: int = 3,
) -> dict[str, Any]:
    failure_modes: Counter[str] = Counter()
    missing_tools: Counter[str] = Counter()
    extra_tools: Counter[str] = Counter()
    extra_arg_keys: Counter[str] = Counter()
    missing_arg_keys: Counter[str] = Counter()
    value_mismatches: Counter[str] = Counter()
    value_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for s in samples:
        res = results[s.id]
        if res.score >= 0.9999:
            failure_modes["perfect"] += 1
            continue

        pred_calls = predictions.get(s.id, {}).get("tool_calls", []) or []
        gold_calls = [c.model_dump() for c in (s.label.tool_calls if s.label else [])]

        gold_has_ask = any(c["name"] == ASK_MISSING_INFO for c in gold_calls)
        pred_has_ask = any(c.get("name") == ASK_MISSING_INFO for c in pred_calls)

        if not gold_calls and pred_calls:
            failure_modes["missed_oos"] += 1
            for pc in pred_calls:
                extra_tools[str(pc.get("name"))] += 1
            continue
        if gold_calls and not pred_calls:
            failure_modes["false_oos"] += 1
            for gc in gold_calls:
                missing_tools[gc["name"]] += 1
            continue
        if gold_has_ask and not pred_has_ask:
            failure_modes["missed_ask"] += 1
        elif pred_has_ask and not gold_has_ask:
            failure_modes["false_ask"] += 1

        if res.f1 >= 0.9999 and not res.state_match:
            failure_modes["state_hash_only_mismatch"] += 1
            continue

        pred_names = Counter(str(c.get("name")) for c in pred_calls)
        gold_names = Counter(c["name"] for c in gold_calls)
        for tname, diff in (gold_names - pred_names).items():
            missing_tools[tname] += diff
        for tname, diff in (pred_names - gold_names).items():
            extra_tools[tname] += diff

        if pred_names != gold_names:
            failure_modes["wrong_tool_sequence"] += 1

        # Pairwise inspect same-name unmatched calls to diagnose key vs value errors
        unmatched_g = [g for g in gold_calls if not any(calls_match(p, g, catalog) for p in pred_calls)]
        unmatched_p = [p for p in pred_calls if not any(calls_match(p, g, catalog) for g in gold_calls)]

        had_key_err = False
        had_val_err = False
        used_p: set[int] = set()
        for g in unmatched_g:
            tname = g["name"]
            p_idx = next(
                (i for i, p in enumerate(unmatched_p) if i not in used_p and p.get("name") == tname),
                None,
            )
            if p_idx is None:
                continue
            used_p.add(p_idx)
            p = unmatched_p[p_idx]
            p_args = _clean(p.get("arguments"))
            g_args = _clean(g.get("arguments"))
            for k in set(p_args) - set(g_args):
                extra_arg_keys[f"{tname}.{k}"] += 1
                had_key_err = True
            for k in set(g_args) - set(p_args):
                missing_arg_keys[f"{tname}.{k}"] += 1
                had_key_err = True
            fuzzy = fuzzy_params(catalog.get(tname))
            for k in set(p_args) & set(g_args):
                pval, gval = p_args[k], g_args[k]
                ok = token_f1(pval, gval) > 0.6 if k in fuzzy else (pval == gval)
                if not ok:
                    param_id = f"{tname}.{k}"
                    value_mismatches[param_id] += 1
                    had_val_err = True
                    if len(value_examples[param_id]) < max_examples_per_param:
                        value_examples[param_id].append(
                            {"id": s.id, "pred": pval, "gold": gval}
                        )

        if had_key_err:
            failure_modes["arg_key_mismatch"] += 1
        if had_val_err:
            failure_modes["arg_value_mismatch"] += 1

    return {
        "failure_mode_counts": dict(failure_modes.most_common()),
        "top_missing_tools": dict(missing_tools.most_common(20)),
        "top_extra_tools": dict(extra_tools.most_common(20)),
        "top_extra_arg_keys": dict(extra_arg_keys.most_common(20)),
        "top_missing_arg_keys": dict(missing_arg_keys.most_common(20)),
        "top_value_mismatches": [
            {
                "param": k,
                "count": cnt,
                "examples": value_examples[k],
            }
            for k, cnt in value_mismatches.most_common(20)
        ],
    }
