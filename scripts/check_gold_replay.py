"""Replay every train label and report failures (rejected gold calls, final_state mismatches)."""
from __future__ import annotations

import argparse

from src.core.dataset import load_split
from src.eval.gold import replay_gold

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    a = p.parse_args()
    bad_calls, bad_state, samples = [], [], load_split("train", a.root)
    for s in samples:
        g = replay_gold(s, a.root)
        if not all(r.get("ok") for r in g.results):
            bad_calls.append(s.id)
        if not g.changed_ok:
            bad_state.append(s.id)
    print(f"samples={len(samples)} rejected_gold_calls={len(bad_calls)} final_state_mismatch={len(bad_state)}")
    print("examples:", bad_calls[:10], bad_state[:10])
