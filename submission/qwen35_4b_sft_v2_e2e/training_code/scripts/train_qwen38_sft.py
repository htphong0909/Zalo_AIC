#!/usr/bin/env python3
"""LoRA SFT trainer for empero-ai/Qwen3.8-4B-Distill (and Qwen3.5-4B) with HF durable checkpoints.

Enforces AGENTS.md 4.3:
- HF preflight before GPU work unless --no-hf-sync.
- Automatic resume from HF `<prefix>/checkpoints/latest` if present.
- Periodic checkpoint sync every <= 10 min + step boundaries + final pure-text Qwen3_5ForCausalLM export.
- Targets hybrid Gated DeltaNet (in_proj_qkv, in_proj_z, out_proj), Full Attention, and MLP modules.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import inspect
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from scripts.train_minicpm5_sft import (
    build_epoch_schedule,
    slice_dataset_by_epoch_progress,
    token_len_from_template_output,
)
from src.agent.qwen35_sft import (
    QWEN35_LORA_TARGET_MODULES,
    render_qwen35_prompt_completion,
    save_qwen35_text_only_checkpoint,
)


def _safe_finite_float(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f


class ModelHealthMonitorCallback:
    """Writes live_metrics.json per log step and aborts training on NaN/explosion/divergence."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        grad_explosion_threshold: float = 50.0,
        grad_explosion_patience: int = 3,
        loss_divergence_threshold: float = 4.5,
        loss_divergence_min_step: int = 60,
        loss_divergence_patience: int = 3,
        step_offset: int = 0,
        epoch_offset: float = 0.0,
        epoch_scale: float = 1.0,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.grad_explosion_threshold = grad_explosion_threshold
        self.grad_explosion_patience = grad_explosion_patience
        self.loss_divergence_threshold = loss_divergence_threshold
        self.loss_divergence_min_step = loss_divergence_min_step
        self.loss_divergence_patience = loss_divergence_patience
        self.step_offset = int(step_offset)
        self.epoch_offset = float(epoch_offset)
        self.epoch_scale = float(epoch_scale)
        self._grad_spikes = 0
        self._high_loss_streak = 0
        self.anomaly: str | None = None
        self.history: list[dict[str, Any]] = []
        self._start_mono = time.monotonic()

    def on_log(self, _args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_kwargs: Any):
        if not getattr(state, "is_world_process_zero", True) or not isinstance(logs, dict):
            return control

        local_step = int(getattr(state, "global_step", 0) or 0)
        local_max_steps = int(getattr(state, "max_steps", 0) or 0)
        step = self.step_offset + local_step
        max_steps = self.step_offset + local_max_steps
        raw_epoch = _safe_finite_float(getattr(state, "epoch", None))
        epoch = (self.epoch_offset + raw_epoch * self.epoch_scale) if raw_epoch is not None else self.epoch_offset
        loss = _safe_finite_float(logs.get("loss"))
        grad_norm = _safe_finite_float(logs.get("grad_norm"))
        lr = _safe_finite_float(logs.get("learning_rate"))

        anomaly: str | None = None
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            anomaly = "NAN_LOSS"
        elif grad_norm is not None and (math.isnan(grad_norm) or math.isinf(grad_norm)):
            anomaly = "GRAD_EXPLOSION"
        else:
            if grad_norm is not None and grad_norm > self.grad_explosion_threshold:
                self._grad_spikes += 1
            elif grad_norm is not None:
                self._grad_spikes = 0
            if self._grad_spikes >= self.grad_explosion_patience:
                anomaly = "GRAD_EXPLOSION"

            if anomaly is None and loss is not None and step >= self.loss_divergence_min_step:
                if loss > self.loss_divergence_threshold:
                    self._high_loss_streak += 1
                else:
                    self._high_loss_streak = 0
                if self._high_loss_streak >= self.loss_divergence_patience:
                    anomaly = "LOSS_DIVERGENCE"

        if anomaly is not None and self.anomaly is None:
            self.anomaly = anomaly
            if control is not None:
                control.should_training_stop = True
            print(f"HEALTH_ANOMALY: {anomaly} detected at step={step} loss={loss} grad_norm={grad_norm}", flush=True)

        elapsed_s = max(1e-3, time.monotonic() - self._start_mono)
        steps_per_sec = (local_step / elapsed_s) if local_step > 0 else 0.0
        rem_steps = max(0, local_max_steps - local_step)
        eta_min = (rem_steps / steps_per_sec / 60.0) if (steps_per_sec > 0 and rem_steps > 0) else None
        vram_used_gb = None
        vram_total_gb = None
        try:
            import torch

            if torch.cuda.is_available():
                vram_used_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        except Exception:
            pass

        entry = {
            "step": step,
            "max_steps": max_steps,
            "epoch": epoch,
            "loss": None if (loss is not None and not math.isfinite(loss)) else loss,
            "grad_norm": None if (grad_norm is not None and not math.isfinite(grad_norm)) else grad_norm,
            "learning_rate": lr,
        }
        if loss is not None or grad_norm is not None:
            self.history.append(entry)
            if len(self.history) > 20:
                self.history = self.history[-20:]

        status = f"ANOMALY:{self.anomaly}" if self.anomaly else "HEALTHY"
        payload = {
            **entry,
            "status": status,
            "anomaly": self.anomaly,
            "grad_spikes": self._grad_spikes,
            "high_loss_streak": self._high_loss_streak,
            "elapsed_s": round(elapsed_s, 1),
            "steps_per_sec": round(steps_per_sec, 4),
            "eta_min": round(eta_min, 1) if eta_min is not None else None,
            "vram_peak_gb": vram_used_gb,
            "vram_total_gb": vram_total_gb,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "history": self.history,
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "live_metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return control


def build_length_bucketed_indices(
    lengths: list[int],
    *,
    total_epochs: float,
    batch_size: int = 4,
    window_batches: int = 16,
    seed: int = 42,
    epoch_offset: float = 0.0,
) -> tuple[list[int], int, int]:
    """Build deterministic length-bucketed epoch schedule aligned on microbatch boundaries.

    Shuffles indices per epoch, groups mega-windows by sequence length to eliminate
    excessive padding inside microbatches without packing across sequence boundaries,
    then shuffles microbatches so sequence lengths remain evenly distributed over training.
    """
    n = len(lengths)
    if n == 0 or total_epochs <= 0:
        return [], 0, 0

    bs = max(1, int(batch_size))
    win_sz = max(bs, bs * max(1, int(window_batches)))
    full_epochs = int(math.floor(total_epochs))
    frac = total_epochs - full_epochs
    num_segments = full_epochs + (1 if frac > 1e-6 else 0)

    scheduled_all: list[int] = []
    for ep in range(num_segments):
        ep_indices = list(range(n))
        random.Random(seed + ep).shuffle(ep_indices)
        if ep == full_epochs:
            rem = min(n, max(1, int(round(frac * n))))
            ep_indices = ep_indices[:rem]

        microbatches: list[list[int]] = []
        for w_start in range(0, len(ep_indices), win_sz):
            win = ep_indices[w_start : w_start + win_sz]
            win.sort(key=lambda idx: lengths[idx])
            for b_start in range(0, len(win), bs):
                microbatches.append(win[b_start : b_start + bs])

        random.Random(seed + 10000 + ep).shuffle(microbatches)
        for mb in microbatches:
            scheduled_all.extend(mb)

    total_rows = len(scheduled_all)
    if epoch_offset <= 0.0:
        return scheduled_all, 0, total_rows
    progress_frac = min(1.0, max(0.0, float(epoch_offset) / float(total_epochs)))
    raw_start = min(total_rows, max(0, int(round(progress_frac * total_rows))))
    aligned_start = (raw_start // bs) * bs
    return scheduled_all[aligned_start:], aligned_start, total_rows


def load_qwen38_sft_rows(data_path: str | Path) -> list[dict[str, Any]]:
    p = Path(data_path)
    opener = gzip.open if str(p).endswith(".gz") else open
    rows: list[dict[str, Any]] = []
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_resume_from_hf(
    syncer: Any | None,
    hf_prefix: str,
    out_dir: Path,
    *,
    init_adapter: str | None = None,
    step_offset: int = 0,
    epoch_offset: float = 0.0,
) -> tuple[str | None, int, float]:
    """Auto-resume from HF `<prefix>/checkpoints/latest` if no explicit init_adapter is given."""
    if init_adapter or syncer is None:
        return init_adapter, int(step_offset), float(epoch_offset)

    latest_repo_path = f"{hf_prefix}/checkpoints/latest"
    if not syncer.exists(f"{latest_repo_path}/progress.json"):
        return None, int(step_offset), float(epoch_offset)

    local_latest = syncer.download_dir(latest_repo_path, out_dir / "hf_resume")
    progress_file = Path(local_latest) / "progress.json"
    if not progress_file.exists():
        return None, int(step_offset), float(epoch_offset)

    prog = json.loads(progress_file.read_text(encoding="utf-8"))
    resumed_step = int(prog.get("global_step") or step_offset)
    resumed_epoch = float(prog.get("epoch") or epoch_offset)
    return str(local_latest), resumed_step, resumed_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen38_sft_v1.yaml")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--hf-prefix", default=None)
    parser.add_argument("--hf-interval-min", type=float, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--init-adapter", default=None)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--epoch-offset", type=float, default=0.0)
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    parser.add_argument("--use-kernels", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-liger-kernel", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--push-merged", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-hf-sync", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    lora_cfg = cfg.get("lora") or {}
    train_cfg = cfg.get("training") or {}

    base_model = args.base_model or str(cfg.get("base_model") or "empero-ai/Qwen3.8-4B-Distill")
    data_path = args.data or str(cfg.get("train_file") or "data/sft/qwen38_train_sft_v1.jsonl.gz")
    out_dir = Path(args.out or str(cfg.get("output_dir") or "experiments/exp_qwen38_sft_v1"))
    target_epochs = float(args.epochs if args.epochs is not None else train_cfg.get("num_train_epochs", 2.0))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("per_device_train_batch_size", 2))
    grad_accum = int(args.grad_accum if args.grad_accum is not None else train_cfg.get("gradient_accumulation_steps", 8))
    lr = float(args.lr if args.lr is not None else train_cfg.get("learning_rate", 1.0e-4))
    max_length = int(args.max_length if args.max_length is not None else train_cfg.get("max_length", 12288))
    hf_interval = float(
        args.hf_interval_min if args.hf_interval_min is not None else train_cfg.get("hf_interval_min", 10.0)
    )
    save_steps = args.save_steps if args.save_steps is not None else train_cfg.get("save_steps", 50)
    seed = int(cfg.get("seed", 42))

    from src.core.hf_sync import (
        BackgroundPusher,
        HFSyncError,
        HFSyncer,
        make_checkpoint_callback,
        resolve_repo_id,
    )

    syncer = None
    pusher = None
    hf_prefix = (args.hf_prefix or out_dir.name).strip("/")
    if not args.no_hf_sync:
        repo = resolve_repo_id(args.hf_repo)
        if not repo:
            sys.exit("ERROR: HF sync required. Pass --hf-repo <user>/<name> or set HF_SYNC_REPO + HF_TOKEN.")
        try:
            syncer = HFSyncer(repo)
            user = syncer.preflight()
            pusher = BackgroundPusher(syncer)
            print(f"HF preflight OK as '{user}': {repo}/{hf_prefix}")
        except HFSyncError as exc:
            sys.exit(f"ERROR (HF preflight): {exc}")

    init_adapter, step_offset, epoch_offset = resolve_resume_from_hf(
        syncer,
        hf_prefix,
        out_dir,
        init_adapter=args.init_adapter,
        step_offset=args.step_offset,
        epoch_offset=args.epoch_offset,
    )
    if init_adapter:
        print(f"Resuming from adapter={init_adapter} (step_offset={step_offset}, epoch_offset={epoch_offset:.4f})")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import SequentialSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(seed)
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | total_vram={props.total_memory / 1e9:.2f}GB | tf32=True", flush=True)

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    raw_rows = load_qwen38_sft_rows(data_path)
    rendered = [render_qwen35_prompt_completion(tok, r) for r in raw_rows]
    ds = Dataset.from_list(rendered)

    lengths: list[int] = []
    for start in range(0, len(rendered), 256):
        batch_texts = [rec["prompt"] + rec["completion"] for rec in rendered[start : start + 256]]
        enc = tok(batch_texts, add_special_tokens=False, truncation=False)
        for ids in enc["input_ids"]:
            lengths.append(len(ids))
    sorted_lens = sorted(lengths) if lengths else [0]
    p50 = sorted_lens[len(sorted_lens) // 2]
    p95 = sorted_lens[min(len(sorted_lens) - 1, int(0.95 * len(sorted_lens)))]
    p99 = sorted_lens[min(len(sorted_lens) - 1, int(0.99 * len(sorted_lens)))]
    max_toks = sorted_lens[-1]
    trunc_cnt = sum(1 for l in lengths if l > max_length)
    print(
        f"Loaded {len(rendered)} examples | p50={p50} p95={p95} p99={p99} max={max_toks} | truncated={trunc_cnt}/{len(rendered)}",
        flush=True,
    )

    use_kernels = bool(args.use_kernels if args.use_kernels is not None else train_cfg.get("use_kernels", False))
    use_liger = bool(
        args.use_liger_kernel if args.use_liger_kernel is not None else train_cfg.get("use_liger_kernel", False)
    )
    use_grad_ckpt = bool(train_cfg.get("gradient_checkpointing", True))
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "device_map": {"": 0},
    }
    if use_kernels:
        load_kwargs["use_kernels"] = True
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, **load_kwargs)
    except (TypeError, ValueError) as exc:
        if "use_kernels" in load_kwargs:
            print(f"WARN: use_kernels=True failed ({exc}); falling back to standard kernels.")
            load_kwargs.pop("use_kernels", None)
            try:
                model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, **load_kwargs)
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, **load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, **load_kwargs)
    model.config.use_cache = False
    if use_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    targets = list(lora_cfg.get("target_modules") or QWEN35_LORA_TARGET_MODULES)
    lora = LoraConfig(
        r=int(lora_cfg.get("r", 32)),
        lora_alpha=int(lora_cfg.get("alpha", 64)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = (
        PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        if init_adapter
        else get_peft_model(model, lora)
    )
    model.print_trainable_parameters()

    total_epochs = max(target_epochs, epoch_offset)
    use_len_buckets = bool(train_cfg.get("length_grouped_windows", True))
    if use_len_buckets and batch_size > 1:
        sched_indices, start_idx, total_rows = build_length_bucketed_indices(
            lengths,
            total_epochs=total_epochs,
            batch_size=batch_size,
            window_batches=int(train_cfg.get("length_window_batches", 16)),
            seed=seed,
            epoch_offset=epoch_offset,
        )
        sliced_ds = ds.select(sched_indices)
    else:
        scheduled = build_epoch_schedule(ds, total_epochs=total_epochs, seed=seed)
        sliced_ds, start_idx, total_rows = slice_dataset_by_epoch_progress(
            scheduled, total_epochs=total_epochs, epoch_offset=epoch_offset
        )
    remaining_epochs = max(0.0, total_epochs - epoch_offset)
    print(f"Schedule ready: start_idx={start_idx}/{total_rows}, remaining={len(sliced_ds)}, len_buckets={use_len_buckets}", flush=True)

    sft_sig = inspect.signature(SFTConfig.__init__).parameters
    default_workers = 4 if os.name != "nt" else 0
    num_workers = int(train_cfg.get("dataloader_num_workers", default_workers))
    sft_kwargs: dict[str, Any] = {
        "output_dir": str(out_dir / "checkpoints"),
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "warmup_steps": int(train_cfg.get("warmup_steps", 20)) if step_offset == 0 else 5,
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "cosine")),
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "max_length": max_length,
        "logging_steps": int(train_cfg.get("logging_steps", 5)),
        "save_strategy": "no",
        "report_to": ["none"],
        "dataloader_num_workers": num_workers,
        "seed": seed,
    }
    if "dataloader_pin_memory" in sft_sig and torch.cuda.is_available():
        sft_kwargs["dataloader_pin_memory"] = True
    if "dataset_num_proc" in sft_sig:
        sft_kwargs["dataset_num_proc"] = int(train_cfg.get("dataset_num_proc", 4 if os.name != "nt" else 1))
    if "max_grad_norm" in sft_sig:
        sft_kwargs["max_grad_norm"] = float(train_cfg.get("max_grad_norm", 1.0))
    if "gradient_checkpointing" in sft_sig:
        sft_kwargs["gradient_checkpointing"] = use_grad_ckpt
    if "optim" in sft_sig:
        default_optim = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"
        sft_kwargs["optim"] = str(train_cfg.get("optim") or default_optim)
    if "warmup_ratio" in sft_sig and train_cfg.get("warmup_ratio") is not None and step_offset == 0:
        sft_kwargs.pop("warmup_steps", None)
        sft_kwargs["warmup_ratio"] = float(train_cfg["warmup_ratio"])
    elif train_cfg.get("warmup_ratio") is not None and step_offset == 0 and train_cfg.get("warmup_steps") is None:
        est_opt_steps = max(1, math.ceil(len(sliced_ds) / max(1, batch_size * grad_accum)))
        sft_kwargs["warmup_steps"] = max(1, round(est_opt_steps * float(train_cfg["warmup_ratio"])))
    if "completion_only_loss" in sft_sig:
        sft_kwargs["completion_only_loss"] = True
    if use_liger and "use_liger_kernel" in sft_sig:
        try:
            import liger_kernel  # noqa: F401
            sft_kwargs["use_liger_kernel"] = True
        except ImportError:
            print("WARN: liger_kernel not installed; continuing without use_liger_kernel.")

    callbacks = []
    health_cb = ModelHealthMonitorCallback(
        out_dir=out_dir,
        step_offset=step_offset,
        epoch_offset=epoch_offset,
        epoch_scale=remaining_epochs,
    )

    class _HealthCB(TrainerCallback):
        def on_log(self, a, s, c, logs=None, **kw):
            return health_cb.on_log(a, s, c, logs=logs, **kw)

    callbacks.append(_HealthCB())

    time_cb = None
    if args.max_wall_minutes and args.max_wall_minutes > 0:
        start_mono = time.monotonic()
        budget_s = float(args.max_wall_minutes) * 60.0

        class _TimeBudgetCB(TrainerCallback):
            def __init__(self) -> None:
                self.stopped_early = False

            def on_step_end(self, _args, state, control, **kwargs):
                if time.monotonic() - start_mono >= budget_s:
                    self.stopped_early = True
                    control.should_training_stop = True
                return control

        time_cb = _TimeBudgetCB()
        callbacks.append(time_cb)

    if pusher is not None:
        raw_ckpt_cb = make_checkpoint_callback(
            pusher,
            out_dir / "hf_snap",
            interval_s=hf_interval * 60.0,
            save_steps=int(save_steps) if save_steps else None,
            path_in_repo=f"{hf_prefix}/checkpoints",
            save_latest_pointer=True,
            step_offset=step_offset,
            epoch_offset=epoch_offset,
            epoch_scale=remaining_epochs,
        )

        class _GuardedCkptCB(TrainerCallback):
            def on_step_end(self, a, s, c, **kw):
                return c if health_cb.anomaly else raw_ckpt_cb.on_step_end(a, s, c, **kw)

            def on_train_end(self, a, s, c, **kw):
                return c if health_cb.anomaly else raw_ckpt_cb.on_train_end(a, s, c, **kw)

        callbacks.append(_GuardedCkptCB())

    class _SequentialSFTTrainer(SFTTrainer):
        def _get_train_sampler(self, train_dataset=None):
            target = train_dataset if train_dataset is not None else self.train_dataset
            return SequentialSampler(target)

    trainer = _SequentialSFTTrainer(
        model=model,
        args=SFTConfig(**sft_kwargs),
        train_dataset=sliced_ds,
        processing_class=tok,
        callbacks=callbacks or None,
    )
    train_loss = 0.0
    if len(trainer.train_dataset) > 0:
        res = trainer.train()
        train_loss = float(res.training_loss)

    if health_cb.anomaly:
        print(f"ABORT: training halted on {health_cb.anomaly}; skipping HF adapter/merged push.", flush=True)
        if pusher is not None:
            pusher.close(timeout=120)
        sys.exit(2)

    if time_cb is not None and time_cb.stopped_early:
        if pusher is not None:
            pusher.close(timeout=600)
        return

    adapter_dir = out_dir / "adapter"
    merged_dir = out_dir / "merged"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    if syncer is not None:
        syncer.push(adapter_dir, f"{hf_prefix}/adapter", "final adapter", raise_on_fail=True)

    # Free optimizer and cached activations before merging LoRA to avoid peak VRAM OOM
    trainer.optimizer = None
    trainer.lr_scheduler = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged_model = trainer.model.merge_and_unload()
    save_qwen35_text_only_checkpoint(merged_model, tok, merged_dir)
    (merged_dir / ".complete").write_text("ok", encoding="utf-8")

    push_merged = bool(args.push_merged if args.push_merged is not None else train_cfg.get("push_merged", False))
    if syncer is not None and push_merged:
        syncer.push(merged_dir, f"{hf_prefix}/merged", "merged pure-text Qwen3_5ForCausalLM", raise_on_fail=True)

    summary = {
        "base_model": base_model,
        "examples": len(rendered),
        "target_epochs": target_epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": lr,
        "training_loss": train_loss,
        "max_tokens": max_toks,
        "p95_tokens": p95,
        "p99_tokens": p99,
        "truncated": trunc_cnt,
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0,
        "runtime_s": time.time() - t0,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if syncer is not None:
        syncer.push(out_dir / "training_summary.json", f"{hf_prefix}/training_summary.json", "training summary", raise_on_fail=True)
        pusher.close(timeout=600)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
