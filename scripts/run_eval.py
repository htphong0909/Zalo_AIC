"""Evaluate a submission-format jsonl on (a subset of) train."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.dataset import load_catalog, load_split
from src.eval.evaluator import evaluate, per_sample_rows, summarize_results


def _load_ids(path: str | Path) -> set[str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return set(raw)
    return set(raw.get("dev_ids") or raw.get("ids") or [])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--root", default="data")
    p.add_argument("--ids", default=None, help="split json with dev_ids / ids or list (default: all train)")
    p.add_argument("--out", required=True, help="experiments/exp_YYYYMMDD_<slug>/eval")
    a = p.parse_args()
    samples = load_split("train", a.root)
    if a.ids:
        keep = _load_ids(a.ids)
        samples = [s for s in samples if s.id in keep]
    with open(a.predictions, encoding="utf-8") as f:
        preds = {row["id"]: row for row in map(json.loads, f) if row}
    results = evaluate(preds, samples, load_catalog("train", a.root), a.root)
    summary = summarize_results(results, samples)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "per_sample.jsonl", "w", encoding="utf-8") as f:
        for row in per_sample_rows(results):
            f.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
