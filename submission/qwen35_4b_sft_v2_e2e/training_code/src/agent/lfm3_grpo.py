"""LFM3 Stage 2 Group Relative Policy Optimization (GRPO) configuration and trainer adhering to lfm3_grpo_with_trl.ipynb."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agent.lfm3_reward import create_lfm3_reward_func
from src.agent.lfm3_sft import (
    LFM3_TARGET_MODULES,
    build_lfm3_lora_config,
    load_yaml_config,
)
from src.core.dataset import load_catalog


@dataclass
class _FallbackGRPOConfig:
    output_dir: str = "./experiments/exp_20261104_lfm3_grpo"
    max_steps: int = 150
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict[str, Any] | None = None
    num_generations: int = 8
    generation_batch_size: int = 8
    max_completion_length: int = 640
    temperature: float = 1.0
    learning_rate: float = 1e-5
    beta: float = 0.04
    lr_scheduler_type: str = "constant_with_warmup"
    warmup_steps: int = 5
    logging_steps: int = 5
    save_strategy: str = "steps"
    save_steps: int = 50
    bf16: bool = True
    fp16: bool = False
    report_to: str = "none"
    seed: int = 42


def build_lfm3_grpo_lora_config(
    r: int = 32,
    alpha: int = 64,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> Any:
    """Build PEFT LoraConfig for LFM3 GRPO stage as in lfm3_grpo_with_trl.ipynb."""
    return build_lfm3_lora_config(
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules or list(LFM3_TARGET_MODULES),
    )


def build_lfm3_grpo_config(cfg: dict[str, Any], *, use_bf16: bool = True) -> Any:
    """Build TRL GRPOConfig from declarative YAML dictionary."""
    train_cfg = cfg.get("training") or {}
    grad_ckpt = bool(train_cfg.get("gradient_checkpointing", True))
    kwargs: dict[str, Any] = {
        "output_dir": str(cfg.get("output_dir") or "./experiments/exp_20261104_lfm3_grpo"),
        "max_steps": int(train_cfg.get("max_steps", 150)),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 2)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 16)),
        "gradient_checkpointing": grad_ckpt,
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if grad_ckpt else None,
        "num_generations": int(train_cfg.get("num_generations", 8)),
        "generation_batch_size": int(train_cfg.get("generation_batch_size", 8)),
        "max_completion_length": int(train_cfg.get("max_completion_length", 256)),
        "temperature": float(train_cfg.get("temperature", 1.0)),
        "learning_rate": float(train_cfg.get("learning_rate", 1e-5)),
        "beta": float(train_cfg.get("beta", 0.04)),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "constant_with_warmup")),
        "warmup_steps": int(train_cfg.get("warmup_steps", 5)),
        "logging_steps": int(train_cfg.get("logging_steps", 5)),
        "save_strategy": str(train_cfg.get("save_strategy", "steps")),
        "save_steps": int(train_cfg.get("save_steps", 50)),
        "bf16": bool(use_bf16),
        "fp16": not bool(use_bf16),
        "report_to": "none",
        "seed": int(cfg.get("seed", 42)),
    }
    try:
        from trl import GRPOConfig

        return GRPOConfig(**kwargs)
    except ImportError:
        return _FallbackGRPOConfig(**kwargs)


def run_lfm3_grpo_training(config_path: str | Path) -> dict[str, str]:
    """Execute Stage 2 GRPO on top of SFT-merged reference model and save adapter + merged weights."""
    os.environ["WANDB_DISABLED"] = "true"
    if "HF_TOKEN" not in os.environ:
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer
    cfg = load_yaml_config(config_path)
    model_id = str(cfg.get("model_id") or "experiments/exp_20261104_lfm3_sft/merged")
    train_file = str(cfg.get("train_file") or "data/sft/lfm3_train_grpo.jsonl")
    output_dir = Path(cfg.get("output_dir") or "./experiments/exp_20261104_lfm3_grpo")
    output_dir.mkdir(parents=True, exist_ok=True)
    split = str(cfg.get("split") or "train")
    root = str(cfg.get("root") or "data")

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
    peft_config = build_lfm3_grpo_lora_config(
        r=int(lora_dict.get("r", 32)),
        alpha=int(lora_dict.get("alpha", 64)),
        dropout=float(lora_dict.get("dropout", 0.0)),
        target_modules=lora_dict.get("target_modules"),
    )
    grpo_config = build_lfm3_grpo_config(cfg, use_bf16=use_bf16)

    catalog = load_catalog(split, root)
    reward_fn = create_lfm3_reward_func(
        catalog=catalog,
        tokenizer=tokenizer,
        split=split,
        root=root,
        verify_env=bool(cfg.get("verify_env", True)),
    )

    train_dataset = load_dataset("json", data_files=train_file, split="train")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    resume_ckpt = cfg.get("resume_from_checkpoint")
    if resume_ckpt and Path(resume_ckpt).exists():
        print(f"Resuming GRPO training from checkpoint: {resume_ckpt}", flush=True)
        trainer.train(resume_from_checkpoint=str(resume_ckpt))
    else:
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

    parser = argparse.ArgumentParser(description="LFM3 Stage 3 GRPO Trainer")
    parser.add_argument("--config", default="configs/lfm3_grpo.yaml", help="Path to YAML config")
    args = parser.parse_args()
    results = run_lfm3_grpo_training(args.config)
    print(json.dumps(results, indent=2))

