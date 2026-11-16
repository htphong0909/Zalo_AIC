"""CLI entrypoint for LFM3 Stage 2 DPO + SFT preference alignment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.lfm3_dpo import run_lfm3_dpo_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lfm3_dpo.yaml")
    args = parser.parse_args()
    out = run_lfm3_dpo_training(args.config)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
