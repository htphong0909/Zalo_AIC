"""LFM3 Stage 1 Supervised Fine-Tuning (SFT) configuration and trainer strictly adhering to lfm3_sft_with_trl.ipynb."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LFM3_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "in_proj",
    "out_proj",
    "w1",
    "w2",
    "w3",
]


@dataclass
class _FallbackLoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(LFM3_TARGET_MODULES))
    task_type: str = "CAUSAL_LM"


@dataclass
class _FallbackSFTConfig:
    output_dir: str = "./experiments/exp_20261104_lfm3_sft"
    num_train_epochs: float = 2.0
    max_steps: int = -1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict[str, Any] | None = None
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: float = 0.05
    max_length: int = 9216
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_steps: int = 50
    bf16: bool = True
    fp16: bool = False
    report_to: str = "none"
    seed: int = 42


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def build_lfm3_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
    use_dora: bool = False,
) -> Any:
    """Build PEFT LoraConfig for LFM3 Tri-Hybrid architecture as in lfm3_sft_with_trl.ipynb."""
    modules = list(target_modules) if target_modules is not None else list(LFM3_TARGET_MODULES)
    try:
        from peft import LoraConfig

        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=modules,
            task_type="CAUSAL_LM",
            use_dora=use_dora,
        )
    except ImportError:
        return _FallbackLoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=modules,
            task_type="CAUSAL_LM",
        )


def build_lfm3_sft_config(cfg: dict[str, Any], *, use_bf16: bool = True) -> Any:
    """Build TRL SFTConfig from declarative YAML dictionary."""
    train_cfg = cfg.get("training") or {}
    grad_ckpt = bool(train_cfg.get("gradient_checkpointing", True))
    kwargs: dict[str, Any] = {
        "output_dir": str(cfg.get("output_dir") or "./experiments/exp_20261104_lfm3_sft"),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 2)),
        "max_steps": int(train_cfg.get("max_steps", -1)),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 4)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 4)),
        "gradient_checkpointing": grad_ckpt,
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if grad_ckpt else None,
        "learning_rate": float(train_cfg.get("learning_rate", 2e-4)),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "cosine")),
        "warmup_steps": float(train_cfg.get("warmup_steps", 0.05)),
        "max_length": int(train_cfg.get("max_length", 9216)),
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "save_strategy": str(train_cfg.get("save_strategy", "epoch")),
        "save_steps": int(train_cfg.get("save_steps", 50)),
        "bf16": bool(use_bf16),
        "fp16": not bool(use_bf16),
        "report_to": "none",
        "seed": int(cfg.get("seed", 42)),
    }
    try:
        from trl import SFTConfig

        return SFTConfig(**kwargs)
    except ImportError:
        return _FallbackSFTConfig(**kwargs)


def run_lfm3_sft_training(config_path: str | Path) -> dict[str, str]:
    """Execute Stage 1 SFT Warmup and merge_and_unload() into a standalone reference checkpoint."""
    os.environ["WANDB_DISABLED"] = "true"
    if "HF_TOKEN" not in os.environ:
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer
    cfg = load_yaml_config(config_path)
    model_id = str(cfg.get("model_id") or "LiquidAI/LFM3-1.7B")
    train_file = str(cfg.get("train_file") or "data/sft/lfm3_train_sft.jsonl")
    output_dir = Path(cfg.get("output_dir") or "./experiments/exp_20261104_lfm3_sft")
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
    peft_config = build_lfm3_lora_config(
        r=int(lora_dict.get("r", 16)),
        alpha=int(lora_dict.get("alpha", 32)),
        dropout=float(lora_dict.get("dropout", 0.05)),
        target_modules=lora_dict.get("target_modules"),
        use_dora=bool(lora_dict.get("use_dora", False)),
    )
    sft_config = build_lfm3_sft_config(cfg, use_bf16=use_bf16)

    raw_ds = load_dataset("json", data_files=train_file, split="train")
    keep_cols = {"prompt", "completion"}
    drop_cols = [c for c in raw_ds.column_names if c not in keep_cols]
    train_dataset = raw_ds.remove_columns(drop_cols) if drop_cols else raw_ds

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
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

    parser = argparse.ArgumentParser(description="LFM3 Stage 1 SFT Trainer")
    parser.add_argument("--config", default="configs/lfm3_sft_max.yaml", help="Path to YAML config")
    args = parser.parse_args()
    results = run_lfm3_sft_training(args.config)
    print(json.dumps(results, indent=2))

