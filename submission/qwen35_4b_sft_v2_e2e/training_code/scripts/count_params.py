"""Count model parameters from safetensors headers on the Hugging Face Hub (no weight download)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

DEFAULT_CONFIG = {
    "asr": {"repo_id": "Qwen/Qwen3-ASR-0.6B", "exclude_prefixes": []},
    "llm": {"repo_id": "Qwen/Qwen3.5-4B", "exclude_prefixes": ["model.visual.", "mtp."]},
    "reference_asr": {"repo_id": "Qwen/Qwen3-ASR-1.7B", "exclude_prefixes": []},
    "limits": {
        "llm_max_params": 5_000_000_000,
        "total_max_params": 6_000_000_000,
        "llm_max_context": 16384,
    },
}


def summarize(tensors: dict[str, int], exclude_prefixes: tuple[str, ...] = (), depth: int = 2) -> dict:
    total = sum(tensors.values())
    excluded = sum(n for name, n in tensors.items() if name.startswith(exclude_prefixes))
    groups: dict[str, int] = defaultdict(int)
    for name, n in tensors.items():
        groups[".".join(name.split(".")[:depth])] += n
    return {
        "total": total,
        "excluded": excluded,
        "kept": total - excluded,
        "groups": dict(sorted(groups.items(), key=lambda kv: -kv[1])),
    }


def check_budget(asr_params: int, llm_params: int, limits: dict) -> list[str]:
    errors = []
    if llm_params >= limits["llm_max_params"]:
        errors.append(f"LLM has {llm_params:,} params (limit < {limits['llm_max_params']:,})")
    if asr_params + llm_params >= limits["total_max_params"]:
        errors.append(f"total {asr_params + llm_params:,} params (limit < {limits['total_max_params']:,})")
    return errors


def fetch_tensor_counts(repo_id: str, revision: str | None = None) -> dict[str, int]:
    from huggingface_hub import HfApi

    meta = HfApi().get_safetensors_metadata(repo_id, revision=revision)
    return {name: info.parameter_count for fm in meta.files_metadata.values() for name, info in fm.tensors.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--out", default=None, help="optional JSON report path")
    args, _ = parser.parse_known_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else DEFAULT_CONFIG
    report = {}
    for role in ("asr", "llm", "reference_asr"):
        entry = cfg[role]
        summary = summarize(fetch_tensor_counts(entry["repo_id"]), tuple(entry.get("exclude_prefixes", [])), args.depth)
        report[role] = {"repo_id": entry["repo_id"], **summary}
        print(f"\n== {role}: {entry['repo_id']}")
        print(f"total={summary['total']:,} excluded={summary['excluded']:,} kept={summary['kept']:,}")
        for group, n in list(summary["groups"].items())[:15]:
            print(f"  {group:<40} {n:>15,}")

    errors = check_budget(report["asr"]["kept"], report["llm"]["kept"], cfg["limits"])
    ref_errors = check_budget(report["reference_asr"]["kept"], report["llm"]["kept"], cfg["limits"])
    print("\nSubmission system:", "OK" if not errors else "; ".join(errors))
    print("With reference ASR (info only):", "OK" if not ref_errors else "; ".join(ref_errors))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()
