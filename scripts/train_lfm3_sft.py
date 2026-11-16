"""CLI entrypoint for LFM3 Stage 1 Supervised Fine-Tuning (SFT) with TRL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.lfm3_sft import run_lfm3_sft_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lfm3_sft.yaml", help="Path to SFT YAML config")
    args = parser.parse_args()

    result = run_lfm3_sft_training(args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
