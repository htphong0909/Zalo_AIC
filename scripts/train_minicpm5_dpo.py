#!/usr/bin/env python3
"""MiniCPM5 Targeted DPO Training with TRL DPOTrainer and HF Durable Checkpoints.

Adheres strictly to AGENTS.md 4.3 (durable checkpoints to HF).
Base Model: openbmb/MiniCPM5-2B-SFT (or warm-start from SFT V2 adapter)
Loss: Sigmoid DPO + SFT Token Retention Anchor (Bradley-Terry margin + alpha*SFT)
Target Modules: 7 Llama projections (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
Format: Native MiniCPM5 XML (<think> + <function name="..."><param name="...">...</param></function>)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_dpo_dataset(data_path: str | Path) -> Any:
    from datasets import Dataset

    p = Path(data_path)
    rows: list[dict[str, Any]] = []
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

    # TRL DPO expects: prompt, chosen, rejected
    clean_rows = []
    for r in rows:
        clean_rows.append({
            "prompt": str(r["prompt"]),
            "chosen": str(r["chosen"]),
            "rejected": str(r["rejected"]),
        })
    return Dataset.from_list(clean_rows)


def main():
    try:
        import torchao  # noqa: F401
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
    except ImportError:
        pass

    import torch
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="openbmb/MiniCPM5-2B-SFT or openbmb/MiniCPM5-2B")
    parser.add_argument("--data", required=True, help="Path to jsonl or jsonl.gz DPO dataset")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--init-adapter", default=None, help="Warm-start adapter directory (e.g. SFT V2 adapter)")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-prompt-length", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hf-repo",
        default=None,
        help="HF repo '<user>/<name>' for durable checkpoints (default: $HF_SYNC_REPO).",
    )
    parser.add_argument("--hf-prefix", default=None, help="Folder inside HF repo (default: basename of --out)")
    parser.add_argument("--hf-interval-min", type=float, default=10.0, help="Checkpoint push period in minutes")
    parser.add_argument("--save-steps", type=int, default=None, help="Optional step interval for snapshots")
    parser.add_argument("--no-hf-sync", action="store_true", help="Disable HF sync (only for smoke tests)")
    args = parser.parse_args()

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
        print("WARNING: --no-hf-sync set. Checkpoints live ONLY on this VM.")
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
    print("STARTING MINICPM5 TARGETED DPO TRAINING")
    print(f"Base Model:    {args.base_model}")
    print(f"Init Adapter:  {args.init_adapter}")
    print(f"Dataset:       {args.data}")
    print(f"Output Dir:    {args.out}")
    print(f"Beta:          {args.beta}")
    print(f"Learning Rate: {args.lr}")
    print(f"Device:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM:    {vram_gb:.2f} GB")
    print("=" * 80)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tokenizer
    print("\n[1/5] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # 2. Dataset
    print(f"\n[2/5] Loading DPO preference dataset from {args.data}...")
    ds = load_dpo_dataset(args.data)
    print(f"Loaded {len(ds)} preference pairs.")

    # 3. Model
    print(f"\n[3/5] Loading base causal LM in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    if args.init_adapter and Path(args.init_adapter).exists():
        print(f"Warm-starting from adapter {args.init_adapter}...")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        print(f"Configuring new LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. DPOConfig & Trainer
    print(f"\n[4/5] Initializing DPOTrainer (beta={args.beta}, lr={args.lr})...")
    total_steps = max(1, int((len(ds) / (args.batch_size * args.grad_accum)) * args.epochs))
    warmup_steps = max(1, int(total_steps * 0.1))

    dpo_kwargs = {
        "output_dir": str(out_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": warmup_steps,
        "beta": args.beta,
        "max_length": args.max_length,
        "bf16": True,
        "fp16": False,
        "logging_steps": 5,
        "save_strategy": "no",
        "report_to": "none",
        "seed": args.seed,
    }

    # Try hybrid loss_type=["sigmoid", "sft"] first; fall back to "sigmoid" if unsupported
    try:
        dpo_cfg = DPOConfig(**dpo_kwargs, loss_type=["sigmoid", "sft"], loss_weights=[1.0, 0.2])
    except Exception as e:
        print(f"Hybrid loss_type not supported by this TRL version ({e}); using sigmoid loss_type.")
        dpo_cfg = DPOConfig(**dpo_kwargs, loss_type="sigmoid")

    callbacks = []
    if pusher is not None:
        callbacks.append(
            make_checkpoint_callback(
                pusher,
                out_dir / "hf_snap",
                interval_s=args.hf_interval_min * 60.0,
                save_steps=args.save_steps,
                path_in_repo=f"{hf_prefix}/checkpoints",
                save_latest_pointer=True,
            )
        )

    import inspect
    sig = inspect.signature(DPOTrainer.__init__)
    trainer_kwargs = {
        "model": model,
        "ref_model": None,  # Automatically uses base model with adapter disabled
        "args": dpo_cfg,
        "train_dataset": ds,
        "callbacks": callbacks or None,
    }
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok

    trainer = DPOTrainer(**trainer_kwargs)

    print("\nStarting DPO training loop...")
    train_res = trainer.train()
    print(f"DPO Training completed! Loss: {train_res.training_loss:.4f} in {time.time() - t0:.2f}s")

    # 5. Save Adapter & Merged Model
    adapter_dir = out_dir / "adapter"
    merged_dir = out_dir / "merged"
    print(f"\n[5/5] Saving adapter to {adapter_dir}...")
    trainer.model.save_pretrained(str(adapter_dir))

    # Reload clean original tokenizer (with full tool-calling template)
    orig_tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False, use_fast=True)
    orig_tok.save_pretrained(str(adapter_dir))

    if syncer is not None:
        ok = syncer.push(adapter_dir, f"{hf_prefix}/adapter", "final dpo adapter", raise_on_fail=False)
        print(f"Final DPO adapter pushed to HF: {ok}")

    print(f"Merging DPO LoRA into base model and saving to {merged_dir}...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    orig_tok.save_pretrained(str(merged_dir))
    (merged_dir / ".complete").write_text("ok", encoding="utf-8")

    # Save summary stats
    summary = {
        "base_model": args.base_model,
        "init_adapter": args.init_adapter,
        "epochs": args.epochs,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "beta": args.beta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "training_loss": train_res.training_loss,
        "total_runtime_s": time.time() - t0,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if syncer is not None:
        syncer.push(out_dir / "training_summary.json", f"{hf_prefix}/training_summary.json", "dpo training summary")
        pusher.close(timeout=600)

    print(f"\nAll DPO models and artifacts saved successfully to {out_dir}!")


if __name__ == "__main__":
    main()
