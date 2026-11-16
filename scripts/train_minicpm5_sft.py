#!/usr/bin/env python3
"""Fine-tune MiniCPM5 (2B-SFT or 2B-Instruct) with TRL + PEFT LoRA.

Implements the official OpenBMB recipe:
- Assistant-only loss masking via TRAIN_CHAT_TEMPLATE
- Standard 7-module Llama projection LoRA (r=64, alpha=128)
- Clean adapter & merged model export preserving original tokenizer
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

TRAIN_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'tool' %}"
    "{%- if loop.first or (messages[loop.index0 - 1]['role'] != 'tool') %}"
    "{{- '<|im_start|>user' }}"
    "{%- endif %}"
    "{{- '\\n<tool_response>\\n' + message['content'] + '\\n</tool_response>' }}"
    "{%- if loop.last or (messages[loop.index0 + 1]['role'] != 'tool') %}"
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if loop.last %}"
    "{%- generation %}"
    "{{- message['content'] + '<|im_end|>\\n' }}"
    "{%- endgeneration %}"
    "{%- else %}"
    "{{- message['content'] + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is true %}"
    "{{- '<think>\\n' }}"
    "{%- elif enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
)


def token_len_from_template_output(encoded: Any) -> int:
    """Return token count whether apply_chat_template returns list[int] or BatchEncoding/dict."""
    if isinstance(encoded, Mapping) or hasattr(encoded, "input_ids"):
        ids = encoded["input_ids"]
    else:
        ids = encoded
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        return len(ids[0])
    return len(ids)


def build_epoch_schedule(ds: Any, total_epochs: float, seed: int = 42) -> Any:
    """Deterministically materialize ``total_epochs`` epochs of ``ds`` in epoch-shuffled order."""
    from datasets import concatenate_datasets

    n = len(ds)
    if n == 0 or total_epochs <= 0:
        return ds.select([])
    full_epochs = int(math.floor(total_epochs))
    frac = total_epochs - full_epochs
    parts = [ds.shuffle(seed=seed + ep_idx) for ep_idx in range(full_epochs)]
    if frac > 1e-6:
        rem = min(n, max(1, int(round(frac * n))))
        parts.append(ds.shuffle(seed=seed + full_epochs).select(range(rem)))
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def slice_dataset_by_epoch_progress(
    ds_prepared: Any,
    *,
    total_epochs: float,
    epoch_offset: float = 0.0,
) -> tuple[Any, int, int]:
    """Skip already-trained prefix of a deterministic schedule/packed dataset cleanly."""
    total_rows = len(ds_prepared)
    if total_rows == 0 or epoch_offset <= 0.0 or total_epochs <= 0.0:
        return ds_prepared, 0, total_rows
    frac = min(1.0, max(0.0, float(epoch_offset) / float(total_epochs)))
    start_idx = min(total_rows, max(0, int(round(frac * total_rows))))
    if start_idx == 0:
        return ds_prepared, 0, total_rows
    return ds_prepared.select(range(start_idx, total_rows)), start_idx, total_rows


def load_sft_dataset(data_path: str | Path) -> Any:
    from datasets import Dataset

    p = Path(data_path)
    rows: list[dict] = []
    if str(p).endswith(".gz"):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return Dataset.from_list([{"messages": r["messages"]} for r in rows])


def main():
    # Fix Colab PEFT incompatible torchao conflict
    try:
        import torchao  # noqa: F401
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
    except ImportError:
        pass

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="openbmb/MiniCPM5-2B-SFT or openbmb/MiniCPM5-2B")
    parser.add_argument("--data", required=True, help="Path to jsonl or jsonl.gz SFT dataset")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hf-repo",
        default=None,
        help="HF repo '<user>/<name>' for durable checkpoints (default: $HF_SYNC_REPO). REQUIRED unless --no-hf-sync.",
    )
    parser.add_argument("--hf-prefix", default=None, help="Folder inside the HF repo (default: basename of --out)")
    parser.add_argument("--hf-interval-min", type=float, default=10.0, help="Checkpoint push period in minutes")
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Optional step interval to snapshot and sync checkpoints in addition to timer",
    )
    parser.add_argument(
        "--no-hf-sync",
        action="store_true",
        help="DANGEROUS: disable HF checkpoint sync. Only for throwaway smoke tests.",
    )
    parser.add_argument("--init-adapter", default=None, help="Warm-start from a saved adapter dir (resume after a crash)")
    parser.add_argument("--step-offset", type=int, default=0, help="Cumulative global_step offset when resuming")
    parser.add_argument("--epoch-offset", type=float, default=0.0, help="Cumulative epoch offset when resuming")
    parser.add_argument("--max-wall-minutes", type=float, default=0.0, help="Graceful stop budget in minutes (0 = disabled)")
    parser.add_argument("--packing", action="store_true", help="Enable TRL sequence packing (eliminates padding waste)")
    parser.add_argument("--padding-free", action="store_true", help="Enable TRL padding_free flattening if supported")
    args = parser.parse_args()

    # Durable-checkpoint gate (AGENTS.md 4.3): fail BEFORE spending any GPU time if HF sync is not usable.
    from src.core.hf_sync import (
        BackgroundPusher,
        HFSyncer,
        HFSyncError,
        make_checkpoint_callback,
        resolve_repo_id,
    )

    syncer = None
    pusher = None
    hf_prefix = (args.hf_prefix or Path(args.out).name).strip("/")
    if args.no_hf_sync:
        print("!" * 80)
        print("WARNING: --no-hf-sync set. Checkpoints live ONLY on this VM and are lost if it dies.")
        print("!" * 80)
    else:
        repo = resolve_repo_id(args.hf_repo)
        if not repo:
            sys.exit("ERROR: HF sync is mandatory. Pass --hf-repo <user>/<name> or set HF_SYNC_REPO (and HF_TOKEN).")
        try:
            syncer = HFSyncer(repo)
            hf_user = syncer.preflight()
        except HFSyncError as exc:
            sys.exit(f"ERROR (HF preflight): {exc}")
        pusher = BackgroundPusher(syncer)
        print(f"HF sync OK as '{hf_user}': {repo}/{hf_prefix} every {args.hf_interval_min:g} min")

    set_seed(args.seed)
    t0 = time.time()

    print("=" * 80)
    print(f"STARTING MINICPM5 SFT TRAINING")
    print(f"Base Model: {args.base_model}")
    print(f"Dataset:    {args.data}")
    print(f"Output Dir: {args.out}")
    print(f"Device:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM: {vram_gb:.2f} GB")
    print("=" * 80)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tokenizer setup
    print("\n[1/5] Loading and patching tokenizer (in-memory only)...")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.chat_template = TRAIN_CHAT_TEMPLATE

    # 2. Dataset loading & truncation pre-audit
    print(f"\n[2/5] Loading training dataset from {args.data}...")
    ds = load_sft_dataset(args.data)
    print(f"Loaded {len(ds)} training samples.")
    trunc_cnt = 0
    max_toks_seen = 0
    for rec in ds:
        ids = tok.apply_chat_template(rec["messages"], tokenize=True)
        n_tok = token_len_from_template_output(ids)
        if n_tok > max_toks_seen:
            max_toks_seen = n_tok
        if n_tok > args.max_length:
            trunc_cnt += 1
    print(
        f"Token length audit vs --max-length={args.max_length}: "
        f"max_seen={max_toks_seen}, truncated={trunc_cnt}/{len(ds)}"
    )

    # 3. Model & LoRA setup
    print(f"\n[3/5] Loading {args.base_model} backbone in bfloat16...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print(f"Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha}, 7 Llama projections)...")
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if args.init_adapter:
        from peft import PeftModel

        print(f"Warm-starting from adapter {args.init_adapter} (optimizer/scheduler restart)...")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # 4. SFTTrainer (deterministic epoch-aware schedule for exact multi-chunk resume)
    total_epochs = float(args.epoch_offset) + float(args.epochs)
    scheduled_ds = build_epoch_schedule(ds, total_epochs=total_epochs, seed=args.seed)
    print(
        f"\n[4/5] Initializing SFTTrainer with assistant_only_loss=True "
        f"(max_length={args.max_length}, packing={args.packing}, padding_free={args.padding_free}, "
        f"total_epochs={total_epochs:.4f}, epoch_offset={args.epoch_offset:.4f}, raw_scheduled={len(scheduled_ds)})..."
    )
    import inspect

    sft_sig = inspect.signature(SFTConfig.__init__).parameters
    sft_kwargs = dict(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=1.0,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=15 if args.step_offset == 0 else 5,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        max_length=args.max_length,
        packing=bool(args.packing),
        assistant_only_loss=True,
        logging_steps=5,
        save_strategy="no",
        report_to=["none"],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=args.seed,
    )
    if args.padding_free and "padding_free" in sft_sig:
        sft_kwargs["padding_free"] = True
    if not args.packing and not args.padding_free and "group_by_length" in sft_sig:
        sft_kwargs["group_by_length"] = True
    sft_cfg = SFTConfig(**sft_kwargs)

    callbacks = []
    time_cb = None
    if args.max_wall_minutes and args.max_wall_minutes > 0:
        from transformers import TrainerCallback

        start_mono = time.monotonic()
        budget_s = float(args.max_wall_minutes) * 60.0

        class TimeBudgetCallback(TrainerCallback):
            def __init__(self) -> None:
                self.stopped_early = False

            def on_step_end(self, _args, state, control, **kwargs):
                if time.monotonic() - start_mono >= budget_s:
                    self.stopped_early = True
                    control.should_training_stop = True
                    abs_step = args.step_offset + int(getattr(state, "global_step", 0))
                    print(
                        f"\n[TimeBudget] Reached {args.max_wall_minutes:g}m chunk budget at step={abs_step}; stopping cleanly to checkpoint & resume.",
                        flush=True,
                    )
                return control

        time_cb = TimeBudgetCallback()
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

    class _SequentialSFTTrainer(SFTTrainer):
        def _get_train_sampler(self, train_dataset=None):
            target = train_dataset if train_dataset is not None else self.train_dataset
            return SequentialSampler(target)

    trainer = _SequentialSFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=scheduled_ds,
        processing_class=tok,
        callbacks=callbacks or None,
    )
    if args.packing:
        trainer.train_dataset = trainer.train_dataset.shuffle(seed=args.seed + 999)
    sliced_ds, start_idx, total_rows = slice_dataset_by_epoch_progress(
        trainer.train_dataset,
        total_epochs=total_epochs,
        epoch_offset=args.epoch_offset,
    )
    trainer.train_dataset = sliced_ds
    print(
        f"Deterministic schedule ready: start_idx={start_idx}/{total_rows}, "
        f"remaining_rows={len(trainer.train_dataset)}",
        flush=True,
    )

    if len(trainer.train_dataset) == 0:
        print("Remaining scheduled rows == 0 (all requested epochs already trained); finalizing adapter!", flush=True)
        train_loss_val = 0.0
    else:
        print("\nStarting training loop...")
        train_res = trainer.train()
        train_loss_val = float(train_res.training_loss)
        print(f"Training loop exited! Loss: {train_loss_val:.4f} in {time.time() - t0:.2f}s")

    if time_cb is not None and time_cb.stopped_early:
        print("[TimeBudget] Chunk finished cleanly; latest checkpoint queued/flushed to HF. Skipping final adapter push.", flush=True)
        if pusher is not None:
            pusher.close(timeout=600)
        return

    # 5. Save Adapter & Merged Model
    adapter_dir = out_dir / "adapter"
    merged_dir = out_dir / "merged"
    print(f"\n[5/5] Saving adapter to {adapter_dir}...")
    trainer.model.save_pretrained(str(adapter_dir))

    # Reload clean original tokenizer (with full tool-calling template) to save alongside adapter
    orig_tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False, use_fast=True)
    orig_tok.save_pretrained(str(adapter_dir))

    # Push the finished adapter to the Hub BEFORE the slow merge, so a crash during merge loses nothing.
    if syncer is not None:
        ok = syncer.push(adapter_dir, f"{hf_prefix}/adapter", "final adapter", raise_on_fail=False)
        print(f"Final adapter pushed to HF: {ok}")

    print(f"Merging LoRA into base model and saving to {merged_dir}...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    orig_tok.save_pretrained(str(merged_dir))
    (merged_dir / ".complete").write_text("ok", encoding="utf-8")  # marks a fully written merged model

    # Save summary stats
    summary = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "training_loss": train_loss_val,
        "total_runtime_s": time.time() - t0,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if syncer is not None:
        syncer.push(out_dir / "training_summary.json", f"{hf_prefix}/training_summary.json", "training summary")
        pusher.close(timeout=600)
    print(f"\nAll models and artifacts saved successfully to {out_dir}!")


if __name__ == "__main__":
    main()
