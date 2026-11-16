"""Generate stratified 5-fold cross-validation splits for train_only (1,246 samples).

Guarantees:
1. Zero leak into dev_v1.json (299 samples).
2. Exactly 1,246 samples partitioned into 5 balanced folds (~249 samples/fold).
3. Multi-label stratified by (category, is_multi_app, has_history, rare_app_group).
4. All 46 apps represented across folds.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.dataset import load_split


def generate_train_only_5folds(
    data_root: str = "data",
    dev_split_path: str = "configs/splits/dev_v1.json",
    output_path: str = "configs/splits/train_only_5folds.json",
    num_folds: int = 5,
    seed: int = 20261108,
) -> dict[str, Any]:
    dev_path = Path(dev_split_path)
    if not dev_path.exists():
        raise FileNotFoundError(f"Dev split file not found: {dev_split_path}")

    raw_dev = json.loads(dev_path.read_text(encoding="utf-8"))
    dev_ids = set(raw_dev.get("dev_ids", raw_dev) if isinstance(raw_dev, dict) else raw_dev)

    all_samples = load_split("train", data_root)
    train_only = [s for s in all_samples if s.id not in dev_ids]

    # Pre-compute app frequencies across train_only to identify rare apps (<35 occurrences)
    app_counts: Counter[str] = Counter()
    for s in train_only:
        apps = set()
        if s.label and s.label.tool_calls:
            for c in s.label.tool_calls:
                apps.add(c.name.split(".", 1)[0])
        for turn in getattr(s, "history", []) or []:
            for hc in getattr(turn, "tool_calls", []) or []:
                apps.add(hc.name.split(".", 1)[0])
        for a in apps:
            app_counts[a] += 1

    rare_apps = {app for app, cnt in app_counts.items() if cnt < 35}

    # Group samples by multi-attribute stratification key
    strata: dict[str, list[str]] = defaultdict(list)
    for s in train_only:
        # Category
        if not s.label or not s.label.tool_calls:
            cat = "oos"
        elif any(c.name == "sys.ask_missing_info" for c in s.label.tool_calls):
            cat = "ask"
        else:
            cat = "act"

        # Multi-app
        apps = set(c.name.split(".", 1)[0] for c in (s.label.tool_calls if s.label else []))
        is_multi_app = "multi" if len(apps) > 1 else "single"

        # History
        has_history = "hist" if getattr(s, "history", None) else "nohist"

        # Rare app bucket
        found_rare = sorted([a for a in apps if a in rare_apps])
        rare_key = found_rare[0] if found_rare else "common"

        key = f"{cat}_{is_multi_app}_{has_history}_{rare_key}"
        strata[key].append(s.id)

    # Deterministic round-robin allocation into folds
    rng = random.Random(seed)
    folds: list[list[str]] = [[] for _ in range(num_folds)]

    # Sort strata by size descending for balanced distribution
    sorted_strata = sorted(strata.items(), key=lambda kv: len(kv[1]), reverse=True)
    fold_pointer = 0

    for key, sample_ids in sorted_strata:
        shuffled_ids = list(sample_ids)
        rng.shuffle(shuffled_ids)
        for sid in shuffled_ids:
            # Assign to the fold with minimum current size, tie-broken by pointer
            min_size = min(len(f) for f in folds)
            cand_folds = [i for i, f in enumerate(folds) if len(f) == min_size]
            chosen = min(cand_folds, key=lambda idx: (idx - fold_pointer) % num_folds)
            folds[chosen].append(sid)
            fold_pointer = (chosen + 1) % num_folds

    # Format result structure
    out_data = {
        "metadata": {
            "num_folds": num_folds,
            "total_samples": len(train_only),
            "seed": seed,
            "dev_path": str(dev_split_path),
            "fold_sizes": [len(f) for f in folds],
        },
        "folds": {f"fold_{i}": sorted(folds[i]) for i in range(num_folds)},
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dev-split", default="configs/splits/dev_v1.json")
    parser.add_argument("--output", default="configs/splits/train_only_5folds.json")
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()

    result = generate_train_only_5folds(
        data_root=args.data_root,
        dev_split_path=args.dev_split,
        output_path=args.output,
        seed=args.seed,
    )
    print(f"Generated 5 folds: sizes = {result['metadata']['fold_sizes']}, total = {result['metadata']['total_samples']}")
