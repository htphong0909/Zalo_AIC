"""Write configs/splits/dev_v1.json from the train labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.dataset import load_split
from src.eval.splits import make_dev_split

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--size", type=int, default=300)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="configs/splits/dev_v1.json")
    a = p.parse_args()
    samples = load_split("train", a.root)
    dev = make_dev_split(samples, a.size, a.seed)
    dev_set = set(dev)
    out = {"seed": a.seed, "dev_ids": dev, "train_ids": sorted(s.id for s in samples if s.id not in dev_set)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"dev={len(dev)} train={len(out['train_ids'])} -> {a.out}")
