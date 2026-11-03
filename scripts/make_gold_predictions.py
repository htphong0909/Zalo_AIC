"""Write gold-label predictions for train in submission format (runner sanity check)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.dataset import load_split
from src.eval.gold import replay_gold

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for s in load_split("train", a.root):
            row = {
                "id": s.id,
                "tool_calls": [c.model_dump() for c in s.label.tool_calls],
                "state_hash": replay_gold(s, a.root).state_hash,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
