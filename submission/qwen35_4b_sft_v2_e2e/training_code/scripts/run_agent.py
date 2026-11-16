"""Run the ReAct agent on a split (with oracle or external ASR transcripts) and write submission JSONL."""
from __future__ import annotations

import os

# Must be set before torch/vllm initialization so vLLM V1 workers do not fork CUDA contexts.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.llm_client import create_llm_backend
from src.agent.react_loop import run_react_loop
from src.core.config import ExperimentConfig, load_config
from src.core.dataset import load_catalog, load_split
from src.env.safe_env import SafeEnv


def _set_seeds(seed: int = 42, *, init_cuda: bool = True) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if init_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _resolve_sample_transcript_inputs(
    transcripts_map: dict[str, dict] | None,
    sample_id: str,
) -> tuple[str | None, dict[int, str] | None, str | None]:
    """Extract current transcript, turn history transcripts, and phonetic diff hint."""
    if transcripts_map is None:
        return None, None, None
    t_entry = transcripts_map.get(sample_id, {}) or {}
    current_text = t_entry.get("audio_text", "")
    raw_hist = t_entry.get("history_texts") or {}
    history_texts = {int(k): v for k, v in raw_hist.items()}
    asr_phonetic_diff = t_entry.get("asr_phonetic_diff") or None
    return current_text, history_texts, asr_phonetic_diff


def _load_ids(path: str | Path) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return set(data)
    return set(data.get("dev_ids") or data.get("ids") or [])


def _load_transcripts_map(path: str | Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row
    return out


def build_effective_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = load_config(args.config) if args.config else ExperimentConfig()
    if args.backend is not None:
        cfg.llm.backend = args.backend
    if args.model_id is not None:
        cfg.llm.model_id = args.model_id
    if args.adapter_path is not None:
        cfg.llm.adapter_path = args.adapter_path
    if getattr(args, "enable_thinking", None) is not None:
        cfg.llm.enable_thinking = args.enable_thinking
    if getattr(args, "prompt_style", None) is not None:
        cfg.agent.prompt_style = args.prompt_style
    if args.max_steps is not None:
        cfg.agent.max_steps = args.max_steps
    if args.max_repairs is not None:
        cfg.agent.max_repairs = args.max_repairs
    if args.timeout_s is not None:
        cfg.agent.timeout_s = args.timeout_s
    if args.no_xref_guard:
        cfg.agent.guardrails.enable_xref_guard = False
    if args.no_arg_sanitizer:
        cfg.agent.guardrails.enable_arg_sanitizer = False
    if args.no_ask_guard:
        cfg.agent.guardrails.enable_ask_guard = False
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to YAML/JSON ExperimentConfig")
    parser.add_argument("--split", default="train")
    parser.add_argument("--root", default="data")
    parser.add_argument("--ids", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--transcripts", default="oracle", help="'oracle' or path to transcripts jsonl")
    parser.add_argument("--backend", choices=("transformers", "vllm", "fake"), default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--adapter-path", default=None, help="Optional PEFT/LoRA adapter path")
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=None,
        help="Force enable_thinking=True in chat template",
    )
    parser.add_argument(
        "--no-enable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Force enable_thinking=False in chat template",
    )
    parser.add_argument("--prompt-style", default=None, help="Override agent prompt style (e.g. v1, qwen35)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-repairs", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--no-xref-guard", action="store_true")
    parser.add_argument("--no-arg-sanitizer", action="store_true")
    parser.add_argument("--no-ask-guard", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace-out", default=None)
    parser.add_argument("--resume", action="store_true")
    args, _ = parser.parse_known_args()

    cfg = build_effective_config(args)
    _set_seeds(cfg.seed, init_cuda=(cfg.llm.backend != "vllm"))

    catalog = load_catalog(args.split, args.root)
    samples = load_split(args.split, args.root)
    if args.ids:
        keep = _load_ids(args.ids)
        samples = [s for s in samples if s.id in keep]
    if args.limit is not None:
        samples = samples[: args.limit]

    transcripts_source = args.transcripts
    if transcripts_source == "oracle" and cfg.asr.transcripts_path:
        transcripts_source = cfg.asr.transcripts_path
    transcripts_map = None if transcripts_source == "oracle" else _load_transcripts_map(transcripts_source)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = Path(args.trace_out) if args.trace_out else None
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])

    llm = create_llm_backend(cfg.llm)

    mode = "a" if args.resume and out_path.exists() else "w"
    t_start = time.perf_counter()
    processed = 0

    with (
        open(out_path, mode, encoding="utf-8") as fout,
        open(trace_path, mode, encoding="utf-8") if trace_path else open(os.devnull, "w") as ftrace,
    ):
        for idx, sample in enumerate(samples, 1):
            if sample.id in done_ids:
                continue
            current_text, history_texts, asr_phonetic_diff = _resolve_sample_transcript_inputs(
                transcripts_map, sample.id
            )

            env = SafeEnv(args.split, sample.id, args.root)
            res = run_react_loop(
                sample,
                env,
                catalog,
                llm,
                current_text=current_text,
                history_texts=history_texts,
                asr_phonetic_diff=asr_phonetic_diff,
                prompt_style=cfg.agent.prompt_style,
                max_steps=cfg.agent.max_steps,
                max_repairs=cfg.agent.max_repairs,
                timeout_s=cfg.agent.timeout_s,
                max_prompt_tokens=cfg.agent.max_prompt_tokens,
                max_result_chars=cfg.agent.max_result_chars,
                guardrails=cfg.agent.guardrails,
            )
            fout.write(json.dumps(res.to_submission_row(), ensure_ascii=False) + "\n")
            fout.flush()
            if trace_path:
                ftrace.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")
                ftrace.flush()

            processed += 1
            if processed % 10 == 0 or idx == len(samples):
                elapsed = time.perf_counter() - t_start
                print(
                    f"[{idx}/{len(samples)}] processed={processed} elapsed={elapsed:.1f}s "
                    f"avg={elapsed / max(1, processed):.2f}s/sample last_id={sample.id} "
                    f"calls={len(res.tool_calls)} steps={res.steps} repairs={res.repairs}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
