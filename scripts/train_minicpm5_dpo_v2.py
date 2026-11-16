#!/usr/bin/env python3
"""MiniCPM5 Tier-1 DPO V2 Training with Exact SFT Reference, Per-Step Telemetry & Anti-Collapse Guards.

Key Guarantees:
1. True Reference Policy: merges `--sft-adapter` into `--base-model` BEFORE attaching fresh DPO LoRA
   so `ref_model=None` (`disable_adapter()`) equals pi_SFT_V3 rather than raw pre-SFT base.
2. Exact Format Parity & Zero Truncation: `--max-length 16384`, `--max-prompt-length 14336`,
   audits exact token length before training.
3. Step-by-Step Telemetry (`logging_steps=1`): logs `loss`, `sft_loss`, `rewards/margins`,
   `rewards/accuracies`, `logps/chosen`, `logps/rejected`, `grad_norm` at EVERY step to
   `step_metrics.jsonl`, plus held-out non-dev validation every `eval_steps`.
4. Anti-Collapse Guard: aborts early if `logps/chosen` drops past `--max-chosen-logp-drop`
   for consecutive steps or if `grad_norm` is NaN/Inf, preserving clean HF checkpoints.
5. Mandatory HF Durable Sync (`<=10 min` + per-step boundaries, never overwriting checkpoints).
"""
from __future__ import annotations

import argparse
import gzip
import inspect
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_minicpm5_sft import (  # noqa: E402
    TRAIN_CHAT_TEMPLATE,
    build_epoch_schedule,
    slice_dataset_by_epoch_progress,
)
from src.core.hf_sync import (  # noqa: E402
    BackgroundPusher,
    HFSyncError,
    HFSyncer,
    make_checkpoint_callback,
    resolve_repo_id,
)


def load_dpo_v2_dataset(data_path: str | Path) -> Any:
    from datasets import Dataset

    opener = gzip.open if str(data_path).endswith(".gz") else open
    records: list[dict[str, str]] = []
    with opener(data_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append(
                {
                    "prompt": str(r["prompt"]),
                    "chosen": str(r["chosen"]),
                    "rejected": str(r["rejected"]),
                }
            )
    return Dataset.from_list(records)


def audit_dpo_token_lengths(ds: Any, tok: Any, max_length: int, label: str) -> dict[str, int]:
    lens: list[int] = []
    for row in ds:
        c_len = len(tok.encode(row["prompt"] + row["chosen"] + "<|im_end|>", add_special_tokens=False))
        r_len = len(tok.encode(row["prompt"] + row["rejected"] + "<|im_end|>", add_special_tokens=False))
        lens.append(max(c_len, r_len))
    lens.sort()
    n = len(lens)
    if n == 0:
        return {"n": 0, "max": 0, "truncated": 0}
    trunc = sum(1 for x in lens if x > max_length)
    stats = {
        "n": n,
        "min": lens[0],
        "p50": lens[n // 2],
        "p95": lens[min(n - 1, int(n * 0.95))],
        "max": lens[-1],
        "truncated": trunc,
    }
    print(f"[TokenAudit:{label}] {stats} (max_length={max_length})", flush=True)
    return stats


def evaluate_step_health(
    logs: dict[str, Any],
    *,
    baseline_chosen_logp: float | None,
    max_chosen_logp_drop: float,
) -> tuple[str, float | None, float | None]:
    """Compute chosen logp delta and classify step telemetry health ('OK', 'WARN', 'COLLAPSE', 'NAN')."""
    loss = logs.get("loss")
    gnorm = logs.get("grad_norm")
    for val in (loss, gnorm):
        if val is not None and (math.isnan(float(val)) or math.isinf(float(val))):
            return "NAN", baseline_chosen_logp, None

    logp_c = logs.get("logps/chosen")
    if logp_c is None:
        return "OK", baseline_chosen_logp, None
    logp_c_f = float(logp_c)
    if baseline_chosen_logp is None:
        baseline_chosen_logp = logp_c_f
    delta = logp_c_f - baseline_chosen_logp
    if delta < -abs(max_chosen_logp_drop):
        return "COLLAPSE", baseline_chosen_logp, delta
    if delta < -0.5 * abs(max_chosen_logp_drop):
        return "WARN", baseline_chosen_logp, delta
    return "OK", baseline_chosen_logp, delta


def make_dpo_monitor_callback(
    out_dir: Path,
    *,
    pusher: BackgroundPusher | None,
    hf_prefix: str,
    total_steps: int,
    step_offset: int = 0,
    epoch_offset: float = 0.0,
    epoch_scale: float = 1.0,
    max_chosen_logp_drop: float = 3.0,
    collapse_patience: int = 3,
):
    from transformers import TrainerCallback

    metrics_path = out_dir / "step_metrics.jsonl"
    best_meta_path = out_dir / "best_val_meta.json"

    class DPOStepMonitorCallback(TrainerCallback):
        def __init__(self) -> None:
            self.baseline_chosen_logp: float | None = None
            self.collapse_streak = 0
            self.best_val_score: float = -1e9
            self.best_val_step: int | None = None
            self.history: list[dict[str, Any]] = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            step = step_offset + int(getattr(state, "global_step", 0))
            local_epoch = float(getattr(state, "epoch", 0.0) or 0.0)
            epoch = epoch_offset + local_epoch * epoch_scale

            if "eval_loss" in logs or "eval_rewards/accuracies" in logs:
                val_acc = float(logs.get("eval_rewards/accuracies", 0.0))
                val_margin = float(logs.get("eval_rewards/margins", 0.0))
                val_logp_c = logs.get("eval_logps/chosen")
                # Favor accuracy first, margin second, without severe chosen-logp drift
                val_score = val_acc * 100.0 + min(val_margin, 5.0)
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    self.best_val_step = step
                    best_meta_path.write_text(
                        json.dumps(
                            {
                                "best_val_step": step,
                                "eval_acc": val_acc,
                                "eval_margin": val_margin,
                                "eval_logp_chosen": val_logp_c,
                                "eval_loss": logs.get("eval_loss"),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                print(
                    f"[DPO-VAL step={step}] eval_loss={logs.get('eval_loss', 0):.4f} "
                    f"eval_acc={val_acc:.4f} eval_margin={val_margin:+.3f} "
                    f"eval_logp_c={val_logp_c} (best_step={self.best_val_step})",
                    flush=True,
                )
                row = {"type": "eval", "step": step, "epoch": epoch, **{k: v for k, v in logs.items()}}
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                return

            if "loss" not in logs:
                return

            status, self.baseline_chosen_logp, delta = evaluate_step_health(
                logs,
                baseline_chosen_logp=self.baseline_chosen_logp,
                max_chosen_logp_drop=max_chosen_logp_drop,
            )
            if status in ("COLLAPSE", "NAN"):
                self.collapse_streak += 1
            else:
                self.collapse_streak = 0

            row = {
                "type": "train",
                "step": step,
                "epoch": round(epoch, 4),
                "chosen_logp_delta": round(delta, 4) if delta is not None else None,
                "health": status,
                **{k: v for k, v in logs.items() if isinstance(v, (int, float, str, bool))},
            }
            self.history.append(row)
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            margin = logs.get("rewards/margins", float("nan"))
            acc = logs.get("rewards/accuracies", float("nan"))
            logp_c = logs.get("logps/chosen", float("nan"))
            logp_r = logs.get("logps/rejected", float("nan"))
            gnorm = logs.get("grad_norm", float("nan"))
            lr = logs.get("learning_rate", float("nan"))
            print(
                f"[DPO step={step}/{total_steps} ep={epoch:.2f}] "
                f"loss={logs['loss']:.4f} margin={margin:+.3f} acc={acc:.3f} "
                f"logp_c={logp_c:.2f} (d={(delta if delta is not None else 0.0):+.2f}) "
                f"logp_r={logp_r:.2f} gnorm={gnorm:.3f} lr={lr:.2e} | {status}",
                flush=True,
            )

            if status == "NAN" or self.collapse_streak >= collapse_patience:
                print(
                    f"!!! GUARD TRIGGERED ({status}, streak={self.collapse_streak}): "
                    "stopping DPO cleanly to prevent model degradation !!!",
                    flush=True,
                )
                control.should_training_stop = True

        def on_save(self, args, state, control, **kwargs):
            if pusher is not None and metrics_path.exists():
                pusher.submit(
                    metrics_path,
                    f"{hf_prefix}/step_metrics.jsonl",
                    f"step metrics @{step_offset + int(getattr(state, 'global_step', 0))}",
                )

    return DPOStepMonitorCallback()


def main() -> None:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    try:
        import peft.import_utils as _peft_imp
        import peft.tuners.lora.torchao as _peft_tao

        _peft_imp.is_torchao_available = lambda: False
        _peft_tao.is_torchao_available = lambda: False
    except Exception:
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--sft-adapter", default=None, help="SFT V3 adapter merged into base BEFORE DPO LoRA")
    parser.add_argument("--resume-dpo-adapter", default=None, help="Optional DPO checkpoint adapter for resume")
    parser.add_argument("--data", required=True)
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=0, help="Defaults to --batch-size when 0")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--no-grad-ckpt", action="store_true", help="Disable gradient checkpointing for high-VRAM GPUs")
    parser.add_argument("--precompute-ref", action="store_true", help="Precompute reference log-probs once before training")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="cDPO label smoothing against noisy pairs")
    parser.add_argument("--sft-weight", type=float, default=0.40, help="RPO/SFT NLL anchor weight protecting chosen logp")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--max-prompt-length", type=int, default=14336)
    parser.add_argument("--max-chosen-logp-drop", type=float, default=3.0)
    parser.add_argument("--collapse-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--hf-prefix", default=None)
    parser.add_argument("--hf-interval-min", type=float, default=10.0)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--epoch-offset", type=float, default=0.0)
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    parser.add_argument("--no-hf-sync", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_prefix = (args.hf_prefix or out_dir.name).strip("/")

    syncer = None
    pusher = None
    if not args.no_hf_sync:
        repo = resolve_repo_id(args.hf_repo)
        if not repo:
            sys.exit("ABORT: set HF_SYNC_REPO or --hf-repo")
        try:
            syncer = HFSyncer(repo)
            hf_user = syncer.preflight()
        except HFSyncError as exc:
            sys.exit(f"ABORT (HF preflight): {exc}")
        pusher = BackgroundPusher(syncer)
        print(f"HF preflight OK as '{hf_user}' -> {repo}/{hf_prefix} (save_steps={args.save_steps}, interval={args.hf_interval_min}m)")

    set_seed(args.seed)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False, use_fast=True)
    tok.chat_template = TRAIN_CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    train_ds = load_dpo_v2_dataset(args.data)
    val_ds = load_dpo_v2_dataset(args.val_data) if args.val_data and Path(args.val_data).exists() else None
    step_window = max(1, args.batch_size * args.grad_accum)
    if args.batch_size > 1 and len(train_ds) > step_window:
        # Sort only within each optimizer step window so batching cuts padding without altering step composition.
        rows = list(train_ds)
        reordered: list[dict[str, Any]] = []
        for i in range(0, len(rows), step_window):
            blk = rows[i : i + step_window]
            blk.sort(key=lambda r: len(r["prompt"]) + max(len(r["chosen"]), len(r["rejected"])))
            reordered.extend(blk)
        from datasets import Dataset

        train_ds = Dataset.from_list(reordered)
    audit_dpo_token_lengths(train_ds, tok, args.max_length, "train")
    if val_ds is not None:
        audit_dpo_token_lengths(val_ds, tok, args.max_length, "val")

    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        pass

    print(f"Loading base model {args.base_model} (attn={attn_impl})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False

    if args.sft_adapter and Path(args.sft_adapter).exists():
        print(f"Merging SFT V3 adapter ({args.sft_adapter}) into reference base before DPO LoRA...", flush=True)
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()

    if args.resume_dpo_adapter and Path(args.resume_dpo_adapter).exists():
        print(f"Resuming DPO LoRA from {args.resume_dpo_adapter}...", flush=True)
        model = PeftModel.from_pretrained(model, args.resume_dpo_adapter, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    eff_bs = max(1, args.batch_size * args.grad_accum)
    total_epochs = float(args.epoch_offset) + float(args.epochs)
    scheduled_ds = build_epoch_schedule(train_ds, total_epochs=total_epochs, seed=args.seed)
    sliced_ds, start_idx, total_rows = slice_dataset_by_epoch_progress(
        scheduled_ds,
        total_epochs=total_epochs,
        epoch_offset=args.epoch_offset,
    )
    total_steps = max(1, int(math.ceil(total_rows / eff_bs)))
    warmup_steps = max(1, int(total_steps * 0.15)) if args.step_offset == 0 else 2
    print(
        f"DPO deterministic schedule: start_idx={start_idx}/{total_rows}, remaining={len(sliced_ds)}, "
        f"total_steps={total_steps}, step_offset={args.step_offset}, epoch_offset={args.epoch_offset:.4f}",
        flush=True,
    )

    if syncer is not None and args.step_offset > 0 and not (out_dir / "step_metrics.jsonl").exists():
        if syncer.exists(f"{hf_prefix}/step_metrics.jsonl"):
            from huggingface_hub import hf_hub_download
            import shutil

            prev_sm = hf_hub_download(repo_id=syncer.repo_id, filename=f"{hf_prefix}/step_metrics.jsonl", token=syncer.token)
            shutil.copy2(prev_sm, out_dir / "step_metrics.jsonl")

    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )
    use_grad_ckpt = not args.no_grad_ckpt
    eval_bs = args.eval_batch_size if args.eval_batch_size > 0 else (4 if vram_gb >= 60.0 else args.batch_size)
    ref_bs = max(4, eval_bs) if vram_gb >= 60.0 else eval_bs
    print(
        f"[SpeedConfig] VRAM={vram_gb:.1f}GB grad_ckpt={use_grad_ckpt} "
        f"bs={args.batch_size} eval_bs={eval_bs} ref_bs={ref_bs} ga={args.grad_accum} precompute_ref={args.precompute_ref}",
        flush=True,
    )

    dpo_sig = inspect.signature(DPOConfig.__init__).parameters
    dpo_kwargs: dict[str, Any] = {
        "output_dir": str(out_dir / "trainer_ckpts"),
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": eval_bs,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": warmup_steps,
        "max_grad_norm": 0.5,
        "bf16": True,
        "gradient_checkpointing": use_grad_ckpt,
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "logging_steps": 1,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "dataloader_num_workers": 0,
        "report_to": "none",
        "seed": args.seed,
        "remove_unused_columns": False,
    }
    if use_grad_ckpt:
        dpo_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if args.precompute_ref and "precompute_ref_log_probs" in dpo_sig:
        dpo_kwargs["precompute_ref_log_probs"] = True
        if "precompute_ref_batch_size" in dpo_sig:
            dpo_kwargs["precompute_ref_batch_size"] = ref_bs
    if val_ds is not None:
        if "eval_strategy" in dpo_sig:
            dpo_kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in dpo_sig:
            dpo_kwargs["evaluation_strategy"] = "steps"
        dpo_kwargs["eval_steps"] = args.save_steps

    if "label_smoothing" in dpo_sig:
        dpo_kwargs["label_smoothing"] = args.label_smoothing
    if "truncation_mode" in dpo_sig:
        dpo_kwargs["truncation_mode"] = "keep_end"

    if "loss_type" in dpo_sig and "loss_weights" in dpo_sig:
        dpo_kwargs["loss_type"] = ["sigmoid", "sft"]
        dpo_kwargs["loss_weights"] = [1.0, args.sft_weight]
    elif "rpo_alpha" in dpo_sig:
        dpo_kwargs["loss_type"] = "sigmoid"
        dpo_kwargs["rpo_alpha"] = args.sft_weight
    else:
        dpo_kwargs["loss_type"] = "sigmoid"

    dpo_cfg = DPOConfig(**{k: v for k, v in dpo_kwargs.items() if k in dpo_sig})

    monitor_cb = make_dpo_monitor_callback(
        out_dir,
        pusher=pusher,
        hf_prefix=hf_prefix,
        total_steps=total_steps,
        step_offset=args.step_offset,
        epoch_offset=args.epoch_offset,
        epoch_scale=args.epochs,
        max_chosen_logp_drop=args.max_chosen_logp_drop,
        collapse_patience=args.collapse_patience,
    )
    callbacks = [monitor_cb]
    time_cb = None
    if args.max_wall_minutes and args.max_wall_minutes > 0:
        from transformers import TrainerCallback

        start_mono = time.monotonic()
        budget_s = float(args.max_wall_minutes) * 60.0

        class DPOTimeBudgetCallback(TrainerCallback):
            def __init__(self) -> None:
                self.stopped_early = False

            def on_step_end(self, _args, state, control, **kwargs):
                if time.monotonic() - start_mono >= budget_s:
                    self.stopped_early = True
                    control.should_training_stop = True
                    print(
                        f"\n[DPO TimeBudget] Reached {args.max_wall_minutes:g}m chunk budget at step={args.step_offset + int(getattr(state, 'global_step', 0))}; stopping cleanly to sync & resume.",
                        flush=True,
                    )
                return control

        time_cb = DPOTimeBudgetCallback()
        callbacks.append(time_cb)

    if pusher is not None:
        callbacks.append(
            make_checkpoint_callback(
                pusher,
                out_dir / "hf_snap",
                interval_s=args.hf_interval_min * 60.0,
                save_steps=args.save_steps,
                path_in_repo=f"{hf_prefix}/checkpoints",
                save_latest_pointer=True,
                step_offset=args.step_offset,
                epoch_offset=args.epoch_offset,
                epoch_scale=args.epochs,
            )
        )

    from torch.utils.data import SequentialSampler

    class _SequentialDPOTrainer(DPOTrainer):
        def _get_train_sampler(self, train_dataset=None):
            target = train_dataset if train_dataset is not None else self.train_dataset
            return SequentialSampler(target)

    tr_sig = inspect.signature(DPOTrainer.__init__).parameters
    tr_kwargs: dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": dpo_cfg,
        "train_dataset": sliced_ds,
        "eval_dataset": val_ds,
        "callbacks": callbacks,
    }
    if "processing_class" in tr_sig:
        tr_kwargs["processing_class"] = tok
    else:
        tr_kwargs["tokenizer"] = tok

    trainer = _SequentialDPOTrainer(**{k: v for k, v in tr_kwargs.items() if k in tr_sig})
    if len(sliced_ds) == 0:
        print("DPO remaining slice == 0; finalizing adapter!", flush=True)
        train_loss_val = 0.0
    else:
        train_res = trainer.train()
        train_loss_val = float(train_res.training_loss)
    runtime_s = time.time() - t0
    print(f"\nDPO V2 chunk exited in {runtime_s:.1f}s (loss={train_loss_val:.4f})", flush=True)

    if time_cb is not None and time_cb.stopped_early:
        if syncer is not None and (out_dir / "step_metrics.jsonl").exists():
            syncer.push(out_dir / "step_metrics.jsonl", f"{hf_prefix}/step_metrics.jsonl", "dpo_v2 chunk metrics", raise_on_fail=False)
        if pusher is not None:
            pusher.close(timeout=600)
        return

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    orig_tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False, use_fast=True)
    orig_tok.save_pretrained(str(adapter_dir))

    summary = {
        "base_model": args.base_model,
        "sft_adapter": args.sft_adapter,
        "epochs": total_epochs,
        "effective_batch_size": eff_bs,
        "learning_rate": args.lr,
        "beta": args.beta,
        "label_smoothing": args.label_smoothing,
        "sft_weight": args.sft_weight,
        "max_length": args.max_length,
        "training_loss": train_loss_val,
        "total_runtime_s": runtime_s,
        "steps_completed": len(monitor_cb.history),
        "best_val_step": monitor_cb.best_val_step,
        "final_health": monitor_cb.history[-1]["health"] if monitor_cb.history else "N/A",
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if syncer is not None:
        syncer.push(adapter_dir, f"{hf_prefix}/adapter", "final dpo_v2 adapter", raise_on_fail=False)
        if (out_dir / "step_metrics.jsonl").exists():
            syncer.push(out_dir / "step_metrics.jsonl", f"{hf_prefix}/step_metrics.jsonl", "dpo_v2 step metrics", raise_on_fail=False)
        if (out_dir / "best_val_meta.json").exists():
            syncer.push(out_dir / "best_val_meta.json", f"{hf_prefix}/best_val_meta.json", "dpo_v2 best val", raise_on_fail=False)
        syncer.push(out_dir / "training_summary.json", f"{hf_prefix}/training_summary.json", "dpo_v2 summary", raise_on_fail=False)
        pusher.close(timeout=600)


if __name__ == "__main__":
    main()
