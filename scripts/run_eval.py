"""Evaluate a submission-format jsonl on (a subset of) train."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.dataset import load_catalog, load_split
from src.eval.evaluator import evaluate, per_sample_rows, summarize_results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--root", default="data")
    p.add_argument("--ids", default=None, help="split json with dev_ids (default: all train)")
    p.add_argument("--out", required=True, help="experiments/exp_YYYYMMDD_<slug>/eval")
    a = p.parse_args()
    samples = load_split("train", a.root)
    if a.ids:
        keep = set(json.loads(Path(a.ids).read_text(encoding="utf-8"))["dev_ids"])
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
