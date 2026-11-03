"""Score predictions (submission-format rows) against train labels."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from src.core.schemas import Sample
from src.eval.gold import replay_gold
from src.eval.metrics import SampleResult, score_sample
from src.eval.splits import stratum


def evaluate(
    predictions: dict[str, dict],
    samples: list[Sample],
    catalog: dict,
    root: str | Path,
) -> dict[str, SampleResult]:
    results = {}
    for s in samples:
        pred = predictions.get(s.id, {"tool_calls": [], "state_hash": ""})
        gold_calls = [c.model_dump() for c in s.label.tool_calls]
        results[s.id] = score_sample(
            pred.get("tool_calls", []),
            pred.get("state_hash", ""),
            gold_calls,
            replay_gold(s, root).state_hash,
            catalog,
        )
    return results


def summarize_results(results: dict[str, SampleResult], samples: list[Sample]) -> dict:
    by_kind: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        kind, _, hist = stratum(s).split("|")
        by_kind[kind].append(results[s.id].score)
        by_kind["history" if hist == "h" else "no_history"].append(results[s.id].score)
    action = [results[s.id] for s in samples if s.label.tool_calls]
    return {
        "n": len(samples),
        "final_score": mean(r.score for r in results.values()),
        "state_match_rate": mean(r.state_match for r in action) if action else None,
        "precision": mean(r.precision for r in action) if action else None,
        "recall": mean(r.recall for r in action) if action else None,
        "f1": mean(r.f1 for r in action) if action else None,
        "by_category": {k: {"n": len(v), "score": mean(v)} for k, v in sorted(by_kind.items())},
    }


def per_sample_rows(results: dict[str, SampleResult]) -> list[dict]:
    return [{"id": sid, **asdict(r)} for sid, r in results.items()]
