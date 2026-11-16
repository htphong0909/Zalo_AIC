#!/usr/bin/env python3
"""Evaluate RL / Voice-OPD / Souped LoRA checkpoints sequentially in one persistent vLLM engine."""
from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROCESSING_METHOD", "spawn")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

import argparse
from dataclasses import asdict
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.llm_client import create_llm_backend  # noqa: E402
from src.agent.react_loop import run_react_loop  # noqa: E402
from src.core.config import load_config  # noqa: E402
from src.core.dataset import load_catalog, load_split  # noqa: E402
from src.core.hf_sync import ENV_TOKEN, HFSyncer, resolve_repo_id  # noqa: E402
from src.env.safe_env import SafeEnv  # noqa: E402
from src.eval.evaluator import evaluate, per_sample_rows, summarize_results  # noqa: E402


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_ids(path: str | Path) -> set[str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return set(raw)
    return set(raw.get("dev_ids") or raw.get("ids") or [])


def _load_transcripts_map(path: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row
    return out


def _resolve_transcript_inputs(
    transcripts_map: dict[str, dict[str, Any]] | None,
    sample_id: str,
) -> tuple[str | None, dict[int, str] | None, str | None]:
    if transcripts_map is None:
        return None, None, None
    entry = transcripts_map.get(sample_id, {}) or {}
    current_text = entry.get("audio_text", "")
    raw_hist = entry.get("history_texts") or {}
    history_texts = {int(k): v for k, v in raw_hist.items()}
    return current_text, history_texts, (entry.get("asr_phonetic_diff") or None)


def _resolve_checkpoint_specs(
    specs: list[str],
    syncer: HFSyncer,
    hf_prefix: str,
    cache_dir: Path,
) -> list[tuple[str, Path | None]]:
    resolved: list[tuple[str, Path | None]] = []
    for spec in specs:
        if spec == "base":
            resolved.append(("base", None))
        elif spec.startswith("local:"):
            raw = spec[len("local:") :]
            if "=" in raw:
                path_str, name = raw.rsplit("=", 1)
            else:
                path_str, name = raw, Path(raw).name
            resolved.append((name, Path(path_str)))
        else:
            ckpt_dir = syncer.download_dir(f"{hf_prefix}/checkpoints/{spec}", cache_dir / "ckpts")
            assert (ckpt_dir / "adapter_config.json").exists(), f"Missing adapter_config.json in {ckpt_dir}"
            resolved.append((spec, ckpt_dir))
    return resolved


def _download_hf_file(repo_file_spec: str, out_dir: Path, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    if ":" not in repo_file_spec:
        return Path(repo_file_spec)
    repo_id, filename = repo_file_spec.split(":", 1)
    dl = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        local_dir=str(out_dir / "asr_cache"),
        token=token,
    )
    return Path(dl)


def _eval_one_run(
    llm: Any,
    cfg: Any,
    samples: list[Any],
    catalog: dict[str, dict[str, Any]],
    *,
    split: str,
    root: str,
    run_dir: Path,
    transcripts_map: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    preds_path = run_dir / "preds.jsonl"
    preds: dict[str, dict[str, Any]] = {}
    latencies: list[float] = []
    _seed_everything(cfg.seed)
    t_all = time.perf_counter()

    with open(preds_path, "w", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples, 1):
            t_s = time.perf_counter()
            current_text, history_texts, asr_diff = _resolve_transcript_inputs(transcripts_map, sample.id)
            env = SafeEnv(split, sample.id, root)
            res = run_react_loop(
                sample,
                env,
                catalog,
                llm,
                current_text=current_text,
                history_texts=history_texts,
                asr_phonetic_diff=asr_diff,
                prompt_style=cfg.agent.prompt_style,
                max_steps=cfg.agent.max_steps,
                max_repairs=cfg.agent.max_repairs,
                timeout_s=cfg.agent.timeout_s,
                max_prompt_tokens=cfg.agent.max_prompt_tokens,
                max_result_chars=cfg.agent.max_result_chars,
                guardrails=cfg.agent.guardrails,
            )
            dt = time.perf_counter() - t_s
            latencies.append(dt)
            row = res.to_submission_row()
            preds[sample.id] = row
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if idx % 50 == 0 or idx == len(samples):
                elapsed = time.perf_counter() - t_all
                print(
                    f"  [{run_dir.name}] {idx}/{len(samples)} avg={elapsed / idx:.2f}s/sample",
                    flush=True,
                )

    results = evaluate(preds, samples, catalog, root)
    summary = summarize_results(results, samples)
    summary["mean_latency_s"] = round(sum(latencies) / max(1, len(latencies)), 3)
    summary["max_latency_s"] = round(max(latencies) if latencies else 0.0, 3)
    summary["total_eval_s"] = round(time.perf_counter() - t_all, 2)

    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(run_dir / "per_sample.jsonl", "w", encoding="utf-8") as f:
        for row in per_sample_rows(results):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/oracle_qwen35_4b_sft_v2.yaml")
    parser.add_argument("--model-id", default="/content/models/qwen35_4b_sft_v2_merged")
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--hf-prefix", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", choices=("oracle", "e2e"), default=["oracle", "e2e"])
    parser.add_argument("--e2e-top-k", type=int, default=0, help="If >0 and oracle runs first, evaluate e2e only on top-k oracle checkpoints.")
    parser.add_argument("--oracle-top-k", type=int, default=0, help="If >0 and e2e runs first, evaluate oracle only on top-k e2e checkpoints.")
    parser.add_argument("--transcripts", default=None, help="Local path or '<hf_repo>:<path_in_repo>' for e2e mode")
    parser.add_argument("--split", default="train")
    parser.add_argument("--root", default="data")
    parser.add_argument("--ids", required=True)
    parser.add_argument("--eval-tag", default="eval_fold0")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import get_token

    repo = resolve_repo_id(args.hf_repo) or "htphong0909/zalo-aic-rl-checkpoints"
    token = os.environ.get(ENV_TOKEN) or get_token()
    syncer = HFSyncer(repo, token=token)
    syncer.preflight()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_ckpts = _resolve_checkpoint_specs(args.checkpoints, syncer, args.hf_prefix, out_dir)

    transcripts_map = None
    if "e2e" in args.modes:
        if not args.transcripts:
            raise ValueError("--transcripts is required when 'e2e' mode is enabled")
        t_path = _download_hf_file(args.transcripts, out_dir, token)
        transcripts_map = _load_transcripts_map(t_path)

    cfg = load_config(args.config)
    cfg.llm.backend = "vllm"
    cfg.llm.model_id = args.model_id
    first_adapter = next((str(p) for _, p in resolved_ckpts if p is not None), None)
    cfg.llm.adapter_path = first_adapter
    cfg.llm.enable_thinking = False
    cfg.llm.gpu_memory_utilization = 0.80

    catalog = load_catalog(args.split, args.root)
    samples = load_split(args.split, args.root)
    keep = _load_ids(args.ids)
    samples = [s for s in samples if s.id in keep]
    print(f"[eval_rl] Loaded {len(samples)} samples (ids={args.ids}); checkpoints={[n for n, _ in resolved_ckpts]}", flush=True)

    llm = create_llm_backend(cfg.llm)
    from vllm.lora.request import LoRARequest  # type: ignore

    summary_rows: list[dict[str, Any]] = []
    oracle_scores: list[tuple[float, str]] = []
    e2e_scores: list[tuple[float, str]] = []
    lora_counter = 0

    def _set_adapter(ckpt_path: Path | None, ckpt_name: str) -> None:
        nonlocal lora_counter
        if ckpt_path is None:
            llm._lora_request = None
        else:
            lora_counter += 1
            llm._lora_request = LoRARequest(f"eval_{ckpt_name}_{lora_counter}", lora_counter, str(ckpt_path))

    if args.oracle_top_k > 0 and "e2e" in args.modes:
        ordered_modes = sorted(args.modes, key=lambda m: 0 if m == "e2e" else 1)
    else:
        ordered_modes = sorted(args.modes, key=lambda m: 0 if m == "oracle" else 1)
    for mode in ordered_modes:
        if mode == "e2e" and args.e2e_top_k > 0 and oracle_scores:
            top_names = {name for _, name in sorted(oracle_scores, reverse=True)[: args.e2e_top_k]}
            targets = [(n, p) for n, p in resolved_ckpts if n in top_names]
        elif mode == "oracle" and args.oracle_top_k > 0 and e2e_scores:
            top_names = {name for _, name in sorted(e2e_scores, reverse=True)[: args.oracle_top_k]}
            targets = [(n, p) for n, p in resolved_ckpts if n in top_names]
        else:
            targets = resolved_ckpts

        t_map = None if mode == "oracle" else transcripts_map
        for ckpt_name, ckpt_path in targets:
            run_name = f"{ckpt_name}_{mode}"
            run_dir = out_dir / args.eval_tag / run_name
            remote_dir = f"{args.hf_prefix}/{args.eval_tag}/{run_name}"
            if not args.overwrite and syncer.exists(f"{remote_dir}/metrics.json"):
                local_m = syncer.download_dir(remote_dir, out_dir / "_hf_existing") / "metrics.json"
                m_data = json.loads(local_m.read_text(encoding="utf-8"))
                print(f"[eval_rl] Reused existing {remote_dir}: final={m_data['final_score']:.4f}", flush=True)
            else:
                _set_adapter(ckpt_path, ckpt_name)
                m_data = _eval_one_run(
                    llm,
                    cfg,
                    samples,
                    catalog,
                    split=args.split,
                    root=args.root,
                    run_dir=run_dir,
                    transcripts_map=t_map,
                )
                syncer.push(run_dir, remote_dir, f"Eval {args.eval_tag} {run_name}", raise_on_fail=True)
            if mode == "oracle":
                oracle_scores.append((float(m_data["final_score"]), ckpt_name))
            elif mode == "e2e":
                e2e_scores.append((float(m_data["final_score"]), ckpt_name))
            row = {
                "checkpoint": ckpt_name,
                "mode": mode,
                "final_score": round(float(m_data["final_score"]), 4),
                "state_match": round(float(m_data["state_match_rate"] or 0.0), 4),
                "f1": round(float(m_data["f1"] or 0.0), 4),
                "act": round(float((m_data["by_category"].get("act") or {}).get("score", 0.0)), 4),
                "ask": round(float((m_data["by_category"].get("ask") or {}).get("score", 0.0)), 4),
                "oos": round(float((m_data["by_category"].get("oos") or {}).get("score", 0.0)), 4),
                "mean_latency_s": m_data.get("mean_latency_s"),
                "max_latency_s": m_data.get("max_latency_s"),
            }
            summary_rows.append(row)
            print(f"[RESULT] {json.dumps(row)}", flush=True)

    summary_file = out_dir / args.eval_tag / "sweep_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    syncer.push(summary_file, f"{args.hf_prefix}/{args.eval_tag}/sweep_summary.json", f"Save {args.eval_tag} summary")
    print("[SWEEP_COMPLETE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
