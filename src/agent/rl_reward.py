"""Composite exact-metric RL rewards combining Tool-N1 exact ceiling, ToolRL bipartite shaping,
fuzzy-field densification, redundancy suppression, and step validity penalties."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.core.schemas import ASK_MISSING_INFO
from src.data_gen.execution_verifier import validate_step_with_reason
from src.eval.metrics import _clean, fuzzy_params, score_sample, token_f1


@dataclass
class RewardBreakdown:
    episode_reward: float
    official_score: float
    state_match: bool
    f1: float
    fuzzy_f1_mean: float
    bipartite_sim: float
    invalid_steps: int
    extra_reads: int
    tier: str


def canonical_state_anchor(
    sample_id: str,
    step_idx: int,
    step_calls: list[dict[str, Any]],
    step_results: list[Any],
    malformed: bool,
) -> str:
    """Hash canonical post-observation GiGPO state within an episode group."""
    payload = {
        "step_idx": int(step_idx),
        "malformed": bool(malformed),
        "calls": step_calls,
        "results": step_results,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return f"{sample_id}|t{step_idx}|{digest}"


def _multiset_jaccard(items_a: list[str], items_b: list[str]) -> float:
    if not items_a and not items_b:
        return 1.0
    if not items_a or not items_b:
        return 0.0
    ca, cb = Counter(items_a), Counter(items_b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / max(1, union)


def _arg_value_similarity(pred_val: Any, gold_val: Any, is_fuzzy: bool) -> float:
    if pred_val == gold_val:
        return 1.0
    if is_fuzzy and pred_val is not None and gold_val is not None:
        return token_f1(pred_val, gold_val)
    return 0.0


def compute_bipartite_call_similarity(
    pred_calls: list[dict[str, Any]],
    gold_calls: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> float:
    """ToolRL greedy bipartite call+key+value similarity enhanced with fuzzy token-F1, in [0, 1]."""
    if not gold_calls and not pred_calls:
        return 1.0
    if not gold_calls or not pred_calls:
        return 0.0

    name_sim = _multiset_jaccard(
        [str(c.get("name") or "") for c in pred_calls],
        [str(c.get("name") or "") for c in gold_calls],
    )
    used_pred: set[int] = set()
    total_units = 1.0
    earned_units = name_sim

    for g in gold_calls:
        g_name = str(g.get("name") or "")
        g_args = _clean(g.get("arguments") or {})
        fuz = fuzzy_params(catalog.get(g_name, {}))
        total_units += 1.0 + len(g_args)

        best_idx = -1
        best_score = -1.0
        for idx, p in enumerate(pred_calls):
            if idx in used_pred or str(p.get("name") or "") != g_name:
                continue
            p_args = _clean(p.get("arguments") or {})
            key_sim = _multiset_jaccard(sorted(p_args.keys()), sorted(g_args.keys()))
            val_sim = sum(
                _arg_value_similarity(p_args.get(k), gv, k in fuz)
                for k, gv in g_args.items()
                if k in p_args
            )
            cand_score = key_sim + val_sim
            if cand_score > best_score:
                best_score = cand_score
                best_idx = idx

        if best_idx >= 0:
            used_pred.add(best_idx)
            earned_units += max(0.0, best_score)

    # Penalize unmatched extra predicted calls smoothly
    over_ratio = min(0.5, 0.08 * max(0, len(pred_calls) - len(used_pred)))
    return max(0.0, min(1.0, (earned_units / max(1.0, total_units)) - over_ratio))


def _mean_matched_fuzzy_f1(
    pred_calls: list[dict[str, Any]],
    gold_calls: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> float:
    sims: list[float] = []
    used: set[int] = set()
    for g in gold_calls:
        g_name = str(g.get("name") or "")
        fuz = fuzzy_params(catalog.get(g_name, {}))
        if not fuz:
            continue
        g_args = _clean(g.get("arguments") or {})
        best_sim = 0.0
        best_i = -1
        for i, p in enumerate(pred_calls):
            if i in used or str(p.get("name") or "") != g_name:
                continue
            p_args = _clean(p.get("arguments") or {})
            pair_sims = [token_f1(p_args.get(k, ""), g_args[k]) for k in g_args if k in fuz]
            score = sum(pair_sims) / max(1, len(pair_sims)) if pair_sims else 1.0
            if score >= best_sim:
                best_sim = score
                best_i = i
        if best_i >= 0:
            used.add(best_i)
        sims.append(best_sim)
    return sum(sims) / len(sims) if sims else 1.0


def compute_step_format_penalty(
    raw_text: str,
    parsed_calls: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    *,
    is_last_turn: bool,
    truncated: bool = False,
) -> tuple[float, bool]:
    """Return (step_penalty <= 0.0, is_valid_step) enforcing clean XML/JSON & JSON-schema adherence."""
    if truncated:
        return -0.20, False
    has_tag = "<tool_call>" in (raw_text or "") or "<function=" in (raw_text or "")
    if has_tag and not parsed_calls:
        return -0.08, False
    if not parsed_calls:
        return (0.0, True) if is_last_turn else (-0.08, False)

    ok, _ = validate_step_with_reason({"step_calls": parsed_calls}, catalog)
    if not ok:
        return -0.08, False
    return 0.0, True


def compute_episode_reward(
    pred_calls: list[dict[str, Any]],
    pred_hash: str,
    gold_calls: list[dict[str, Any]],
    gold_hash: str,
    catalog: dict[str, dict[str, Any]],
    *,
    invalid_steps: int = 0,
    extra_reads: int = 0,
    malformed_terminal: bool = False,
) -> RewardBreakdown:
    """Compute strictly monotonic episode reward respecting exact score_sample() optimum (+1.0)."""
    res = score_sample(pred_calls, pred_hash, gold_calls, gold_hash, catalog)
    bipartite = compute_bipartite_call_similarity(pred_calls, gold_calls, catalog)
    fuzzy_mean = _mean_matched_fuzzy_f1(pred_calls, gold_calls, catalog)
    inv_pen = min(0.24, 0.08 * max(0, int(invalid_steps))) + (0.10 if malformed_terminal else 0.0)

    # Case 1: Out-of-scope (gold_calls == [])
    if not gold_calls:
        if not pred_calls and not malformed_terminal and invalid_steps == 0:
            return RewardBreakdown(1.0, res.score, res.state_match, res.f1, 1.0, 1.0, invalid_steps, extra_reads, "exact_oos")
        has_ask = any(c.get("name") == ASK_MISSING_INFO for c in pred_calls)
        rew = -0.25 if has_ask else (-0.30 if pred_calls else max(-0.30, 0.20 - inv_pen))
        return RewardBreakdown(round(rew, 6), res.score, res.state_match, res.f1, fuzzy_mean, bipartite, invalid_steps, extra_reads, "oos_fail")

    # Case 2: Clarification (gold_calls contains sys.ask_missing_info)
    if any(c.get("name") == ASK_MISSING_INFO for c in gold_calls):
        if res.score >= 1.0 - 1e-6 and invalid_steps == 0 and not malformed_terminal:
            return RewardBreakdown(1.0, res.score, res.state_match, res.f1, 1.0, 1.0, invalid_steps, extra_reads, "exact")
        pred_asks = [c for c in pred_calls if c.get("name") == ASK_MISSING_INFO]
        if pred_asks:
            g_args = gold_calls[0].get("arguments") or {}
            p_args = pred_asks[0].get("arguments") or {}
            tool_hit = 0.25 if p_args.get("tool_name") == g_args.get("tool_name") else 0.0
            gp = [str(x) for x in (g_args.get("params") or [])]
            pp = [str(x) for x in (p_args.get("params") or [])]
            param_sim = 0.30 * _multiset_jaccard(pp, gp) if tool_hit > 0 else 0.0
            rew = max(0.10, min(0.55, 0.10 + tool_hit + param_sim - 0.5 * inv_pen))
            return RewardBreakdown(round(rew, 6), res.score, res.state_match, res.f1, fuzzy_mean, bipartite, invalid_steps, extra_reads, "ask_partial")
        rew = -0.20 if not pred_calls else max(-0.30, -0.25 - 0.5 * inv_pen)
        return RewardBreakdown(round(rew, 6), res.score, res.state_match, res.f1, fuzzy_mean, bipartite, invalid_steps, extra_reads, "ask_missed")

    # Case 3: Normal action episode (act)
    if res.score >= 1.0 - 1e-6 and res.f1 >= 1.0 - 1e-6 and extra_reads == 0 and invalid_steps == 0 and not malformed_terminal:
        return RewardBreakdown(1.0, res.score, res.state_match, res.f1, 1.0, 1.0, invalid_steps, extra_reads, "exact")

    if res.state_match:
        # Strict ceiling 0.94 ensures official F1=1.0 & minimal calls always strictly win
        raw = 0.50 + 0.42 * float(res.f1) + 0.05 * float(fuzzy_mean) - 0.06 * max(0, int(extra_reads)) - 0.3 * inv_pen
        rew = max(0.44, min(0.94, raw))
        return RewardBreakdown(round(rew, 6), res.score, True, res.f1, fuzzy_mean, bipartite, invalid_steps, extra_reads, "sm_partial")

    # State mismatch tier (res.score == 0.0)
    if any(c.get("name") == ASK_MISSING_INFO for c in pred_calls):
        return RewardBreakdown(-0.15, 0.0, False, res.f1, fuzzy_mean, bipartite, invalid_steps, extra_reads, "false_ask")
    if not pred_calls:
        return RewardBreakdown(-0.20, 0.0, False, 0.0, 0.0, 0.0, invalid_steps, extra_reads, "false_oos")

    rew = max(-0.15, min(0.35, 0.35 * float(bipartite) - 0.5 * inv_pen))
    return RewardBreakdown(round(rew, 6), 0.0, False, res.f1, fuzzy_mean, bipartite, invalid_steps, extra_reads, "sm_fail")
