"""LFM3 Stage 2 Direct Preference Optimization (DPO + SFT anchor) adhering to lfm3_dpo_with_trl.ipynb."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agent.lfm3_sft import (
    LFM3_TARGET_MODULES,
    build_lfm3_lora_config,
    load_yaml_config,
)


@dataclass
class _FallbackDPOConfig:
    output_dir: str = "./experiments/exp_20261107_lfm3_dpo"
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict[str, Any] | None = None
    learning_rate: float = 2e-5
    loss_type: list[str] = field(default_factory=lambda: ["sigmoid", "sft"])
    loss_weights: list[float] = field(default_factory=lambda: [1.0, 0.5])
    lr_scheduler_type: str = "cosine"
    warmup_steps: float = 0.1
    beta: float = 0.1
    max_length: int = 10240
    logging_steps: int = 10
    save_strategy: str = "epoch"
    bf16: bool = True
    fp16: bool = False
    report_to: str = "none"
    seed: int = 42


def build_lfm3_dpo_lora_config(
    r: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> Any:
    """Build PEFT LoraConfig for LFM3 DPO stage (including gate_proj) as in lfm3_dpo_with_trl.ipynb."""
    return build_lfm3_lora_config(
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules or list(LFM3_TARGET_MODULES),
    )


def build_lfm3_dpo_config(cfg: dict[str, Any], *, use_bf16: bool = True) -> Any:
    """Build TRL DPOConfig combining sigmoid DPO + SFT chosen loss from declarative YAML."""
    train_cfg = cfg.get("training") or {}
    grad_ckpt = bool(train_cfg.get("gradient_checkpointing", True))
    raw_loss_type = train_cfg.get("loss_type", ["sigmoid", "sft"])
    loss_type = list(raw_loss_type) if isinstance(raw_loss_type, list) else [str(raw_loss_type)]
    raw_weights = train_cfg.get("loss_weights", [1.0, 0.5])
    loss_weights = [float(x) for x in raw_weights] if isinstance(raw_weights, list) else [1.0]

    kwargs: dict[str, Any] = {
        "output_dir": str(cfg.get("output_dir") or "./experiments/exp_20261107_lfm3_dpo"),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 1.0)),
        "max_steps": int(train_cfg.get("max_steps", -1)),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 2)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 8)),
        "gradient_checkpointing": grad_ckpt,
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if grad_ckpt else None,
        "learning_rate": float(train_cfg.get("learning_rate", 2e-5)),
        "loss_type": loss_type,
        "loss_weights": loss_weights,
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "cosine")),
        "warmup_steps": float(train_cfg.get("warmup_steps", 0.1)),
        "beta": float(train_cfg.get("beta", 0.1)),
        "max_length": int(train_cfg.get("max_length", 10240)),
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "save_strategy": str(train_cfg.get("save_strategy", "epoch")),
        "bf16": bool(use_bf16),
        "fp16": not bool(use_bf16),
        "report_to": "none",
        "seed": int(cfg.get("seed", 42)),
    }
    try:
        from trl import DPOConfig

        return DPOConfig(**kwargs)
    except ImportError:
        return _FallbackDPOConfig(**kwargs)


def run_lfm3_dpo_training(config_path: str | Path) -> dict[str, str]:
    """Execute Stage 2 DPO + SFT calibration on top of Stage 1 merged checkpoint and save adapter + merged weights."""
    os.environ["WANDB_DISABLED"] = "true"
    if "HF_TOKEN" not in os.environ:
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer

    cfg = load_yaml_config(config_path)
    model_id = str(cfg.get("model_id") or "experiments/exp_20261107_lfm3_sft/merged")
    train_file = str(cfg.get("train_file") or "data/sft/lfm3_train_dpo.jsonl")
    output_dir = Path(cfg.get("output_dir") or "./experiments/exp_20261107_lfm3_dpo")
    output_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_dict = cfg.get("lora") or {}
    peft_config = build_lfm3_dpo_lora_config(
        r=int(lora_dict.get("r", 32)),
        alpha=int(lora_dict.get("alpha", 64)),
        dropout=float(lora_dict.get("dropout", 0.05)),
        target_modules=lora_dict.get("target_modules"),
    )
    dpo_config = build_lfm3_dpo_config(cfg, use_bf16=use_bf16)

    raw_ds = load_dataset("json", data_files=train_file, split="train")
    keep_cols = {"prompt", "chosen", "rejected"}
    drop_cols = [c for c in raw_ds.column_names if c not in keep_cols]
    train_dataset = raw_ds.remove_columns(drop_cols) if drop_cols else raw_ds

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    return {"adapter_dir": str(adapter_dir), "merged_dir": str(merged_dir)}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="LFM3 Stage 2 DPO Trainer")
    parser.add_argument("--config", default="configs/lfm3_dpo.yaml", help="Path to YAML config")
    args = parser.parse_args()
    results = run_lfm3_dpo_training(args.config)
    print(json.dumps(results, indent=2))

