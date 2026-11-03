"""Deterministic stratified dev split of the train set."""
from __future__ import annotations

import random
from collections import defaultdict

from src.core.schemas import ASK_MISSING_INFO, Sample


def stratum(sample: Sample) -> str:
    calls = sample.label.tool_calls if sample.label else []
    n = len(calls)
    kind = "oos" if n == 0 else "ask" if any(c.name == ASK_MISSING_INFO for c in calls) else "act"
    size = "0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else "4+"
    return f"{kind}|{size}|{'h' if sample.history else 'n'}"


def make_dev_split(samples: list[Sample], dev_size: int = 300, seed: int = 2026) -> list[str]:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        groups[stratum(s)].append(s.id)
    dev: list[str] = []
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        dev.extend(ids[: round(len(ids) * dev_size / len(samples))])
    return sorted(dev)
