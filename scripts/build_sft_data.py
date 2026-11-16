"""Generate leak-free multi-turn ReAct SFT dataset from train split (strictly excluding dev_ids)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.sft_dataset import build_sft_trajectory
from src.core.dataset import load_catalog, load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    parser.add_argument("--exclude-ids", default="configs/splits/dev_v1.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    exclude: set[str] = set()
    if args.exclude_ids:
        raw = json.loads(Path(args.exclude_ids).read_text(encoding="utf-8"))
        exclude = set(raw if isinstance(raw, list) else (raw.get("dev_ids") or raw.get("ids") or []))

    catalog = load_catalog("train", args.root)
    samples = [s for s in load_split("train", args.root) if s.id not in exclude]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            row = build_sft_trajectory(s, catalog)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "n_trajectories": len(samples),
                "excluded_dev_ids": len(exclude),
                "out": str(out_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
