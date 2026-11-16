#!/usr/bin/env python3
"""Evaluate an LLM checkpoint on the 100-sample deterministic Unseen-Schema Probe."""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.tool_caller import parse_tool_calls
from src.core.schemas import ASK_MISSING_INFO, ToolCall


def _norm_val(val: Any) -> Any:
    if isinstance(val, str):
        return unicodedata.normalize("NFC", val).strip()
    if isinstance(val, list):
        return [_norm_val(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _norm_val(v) for k, v in sorted(val.items())}
    return val


def _row_catalog(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cat: dict[str, dict[str, Any]] = {}
    for t in row.get("tools") or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            cat[str(fn["name"])] = {
                "name": str(fn["name"]),
                "parameters": fn.get("parameters") or {},
            }
    return cat


def _calls_sig(calls: list[ToolCall]) -> list[tuple[str, dict[str, Any]]]:
    return [(c.name, _norm_val(c.arguments or {})) for c in calls]


def score_probe_predictions(
    rows: list[dict[str, Any]],
    raw_outputs: list[str],
) -> dict[str, Any]:
    """Compute unseen-schema tool-name EM, exact call+arg match, false-ask rate, and Ask F1."""
    assert len(rows) == len(raw_outputs)
    total = len(rows)
    name_hits = 0
    exact_hits = 0
    ask_tp = 0
    ask_fp = 0
    ask_fn = 0

    cat_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "name_hits": 0, "exact_hits": 0, "pred_asks": 0}
    )

    for row, pred_raw in zip(rows, raw_outputs):
        cat = _row_catalog(row)
        gold_calls = parse_tool_calls(str(row.get("completion") or ""), catalog=cat)
        pred_calls = parse_tool_calls(str(pred_raw or ""), catalog=cat)

        gold_names = [c.name for c in gold_calls]
        pred_names = [c.name for c in pred_calls]
        name_ok = gold_names == pred_names
        exact_ok = _calls_sig(gold_calls) == _calls_sig(pred_calls)

        name_hits += int(name_ok)
        exact_hits += int(exact_ok)

        gold_is_ask = any(n == ASK_MISSING_INFO for n in gold_names)
        pred_is_ask = any(n == ASK_MISSING_INFO for n in pred_names)
        if gold_is_ask and pred_is_ask:
            ask_tp += 1
        elif pred_is_ask and not gold_is_ask:
            ask_fp += 1
        elif gold_is_ask and not pred_is_ask:
            ask_fn += 1

        pcat = str(row.get("probe_category") or row.get("step_category") or "unknown")
        cs = cat_stats[pcat]
        cs["count"] += 1
        cs["name_hits"] += int(name_ok)
        cs["exact_hits"] += int(exact_ok)
        cs["pred_asks"] += int(pred_is_ask)

    ask_prec = ask_tp / max(1, ask_tp + ask_fp)
    ask_rec = ask_tp / max(1, ask_tp + ask_fn)
    ask_f1 = (2 * ask_prec * ask_rec / (ask_prec + ask_rec)) if (ask_prec + ask_rec) > 0 else 0.0

    anti_stats = cat_stats.get("unseen_anti_false_ask") or {"count": 0, "pred_asks": 0}
    false_ask_rate = anti_stats["pred_asks"] / max(1, anti_stats["count"])

    by_category = {
        k: {
            "count": v["count"],
            "name_em": round(v["name_hits"] / max(1, v["count"]), 4),
            "exact_match": round(v["exact_hits"] / max(1, v["count"]), 4),
            "ask_pred_rate": round(v["pred_asks"] / max(1, v["count"]), 4),
        }
        for k, v in sorted(cat_stats.items())
    }

    return {
        "total": total,
        "tool_name_em": round(name_hits / max(1, total), 4),
        "exact_call_match": round(exact_hits / max(1, total), 4),
        "ask_precision": round(ask_prec, 4),
        "ask_recall": round(ask_rec, 4),
        "ask_f1": round(ask_f1, 4),
        "anti_false_ask_trigger_rate": round(false_ask_rate, 4),
        "by_category": by_category,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", default="data/sft/unseen_schema_eval.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    args = parser.parse_args()

    from src.agent.llm_client import VLLMBackend

    rows = [
        json.loads(line)
        for line in Path(args.probe).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    t0 = time.perf_counter()
    llm = VLLMBackend(
        model_id=args.model,
        adapter_path=args.adapter,
        max_model_len=args.max_model_len,
        max_new_tokens=384,
        enable_thinking=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        repetition_penalty=1.05,
        seed=42,
    )
    llm.warmup()

    raw_outputs: list[str] = []
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        reqs = [(r["prompt_messages"], r.get("tools") or None, False) for r in chunk]
        raw_outputs.extend(llm.generate_batch(reqs))

    elapsed_s = round(time.perf_counter() - t0, 2)
    metrics = score_probe_predictions(rows, raw_outputs)
    metrics["model"] = args.model
    metrics["adapter"] = args.adapter
    metrics["elapsed_s"] = elapsed_s

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
