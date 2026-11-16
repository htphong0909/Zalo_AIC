"""CLI submission entrypoint generating /result/submission.jsonl."""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from agent import DEFAULT_CONFIG_PATH, run_inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="LifeMate ZAC 2026 E2E inference submission runner.")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/data"))
    parser.add_argument("--output", default=os.environ.get("OUTPUT_PATH", "/result/submission.jsonl"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--split", default=os.environ.get("DATA_SPLIT") or None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    run_inference(
        data_root=args.data_root,
        output_path=args.output,
        config_path=args.config,
        split_override=args.split,
        limit=args.limit,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
