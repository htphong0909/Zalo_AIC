"""LFM3-1.7B Stage 1 SFT Training Worker on NVIDIA L4 (Colab) - Verified Standard LoRA"""
import os
import sys
import time

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_TOKEN"] = ""
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

def main():
    start_time = time.time()
    print("=" * 60)
    print("STARTING APEX LFM3-1.7B STAGE 1 SFT ON NVIDIA L4 (VERIFIED LORA)")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")
    print("=" * 60)

    model_dir = "/content/models/lfm3_base"
    train_file = "/content/data/sft/lfm3_train_sft.jsonl"
    output_dir = Path("/content/experiments/exp_lfm3_sft_apex")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/5] Loading model backbone in bfloat16 on cuda:0...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.enable_input_require_grads()
    print(f"Base model loaded 100% on GPU. VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print(f"[3/5] Configuring PEFT LoRA (r=64, alpha=128, 10 target modules, use_dora=False)...")
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
        "in_proj", "out_proj", "w1", "w2", "w3"
    ]
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        use_dora=False,
    )

    print(f"[4/5] Loading dataset from {train_file}...")
    raw_ds = load_dataset("json", data_files=train_file, split="train")
    keep_cols = {"prompt", "completion"}
    drop_cols = [c for c in raw_ds.column_names if c not in keep_cols]
    train_dataset = raw_ds.remove_columns(drop_cols) if drop_cols else raw_ds
    print(f"Dataset loaded: {len(train_dataset)} samples.")

    # Effective batch size = 1 * 16 = 16 (identical to 2 * 8 = 16, but uses 50% less activation VRAM)
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        max_length=8192,
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=3,
        bf16=True,
        fp16=False,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()

    print("[5/5] Launching training loop...")
    train_result = trainer.train()
    elapsed = time.time() - start_time
    print(f"\n>>> Training finished in {elapsed/60:.2f} minutes!")
    print(f"Final Train Loss: {train_result.training_loss:.4f}")

    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving LoRA adapter checkpoint to {adapter_dir}...")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print(f"Merging and unloading model to {merged_dir}...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"STAGE 1 SFT COMPLETED SUCCESSFULLY IN {total_time/60:.2f} MINUTES!")
    print("=" * 60)

if __name__ == "__main__":
    main()
