#!/usr/bin/env python3
"""Audio-Tower-Only LoRA Adaptation for PhoWhisper Large on LifeMate Speech (CLAnSR ICASSP 2025).

Trains parameter-efficient LoRA adapters (r=4, alpha=8, lr=2e-5, cosine schedule)
strictly targeting acoustic encoder projections (encoder self-attention q_proj and v_proj)
while keeping 100% of the decoder frozen.

Strict Invariants:
- Zero Dev Leakage: The 299 dev samples are strictly sealed and NEVER accessed.
- Frozen Language Model Prior: The decoder remains 100% frozen to preserve general Vietnamese
  generative syntax, proper names, casing, and number representations.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    set_seed,
)
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.asr.audio import load_wav_16k
from src.asr.whisper_engine import preprocess_waveform

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phowhisper_lora")


def load_processor(model_id: str):
    try:
        return AutoProcessor.from_pretrained(model_id)
    except (AttributeError, Exception):
        from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizerFast

        try:
            fe = WhisperFeatureExtractor.from_pretrained(model_id)
        except Exception:
            fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
        tok = WhisperTokenizerFast.from_pretrained(model_id, extra_special_tokens={})
        return WhisperProcessor(feature_extractor=fe, tokenizer=tok)


class LifeMateASRDataset(Dataset):
    """Dataset loading strictly Train split audio clips with aligned acoustic preprocessing.

    Applies exact inference front-end: conservative silence trimming (-35 dB, 150ms margin),
    peak amplitude normalization to 0.95, and bilateral 60ms collar padding.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        data_root: Path,
        processor: Any,
        pre_roll_pad_ms: float = 60.0,
        post_roll_pad_ms: float = 60.0,
        peak_normalize: bool = True,
        trim_silence: bool = True,
        vad_db_threshold: float = -35.0,
        vad_margin_ms: float = 150.0,
    ):
        self.samples = samples
        self.data_root = Path(data_root)
        self.processor = processor
        self.pre_roll_pad_ms = pre_roll_pad_ms
        self.post_roll_pad_ms = post_roll_pad_ms
        self.peak_normalize = peak_normalize
        self.trim_silence = trim_silence
        self.vad_db_threshold = vad_db_threshold
        self.vad_margin_ms = vad_margin_ms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        audio_rel = item.get("audio", "")
        candidates = [
            self.data_root / audio_rel,
            self.data_root / "data" / "train" / audio_rel,
            self.data_root / "train" / audio_rel,
            Path("/content/eval_data/data/train") / audio_rel,
            Path("/content/ZALO_AIC/data/train") / audio_rel,
            ROOT / "data" / "train" / audio_rel,
        ]
        audio_path = None
        for c in candidates:
            if c.exists():
                audio_path = c
                break
        if audio_path is None:
            raise FileNotFoundError(f"Audio file not found: {audio_rel} in {[str(c) for c in candidates]}")

        # Load 16kHz mono audio
        waveform, sr = load_wav_16k(audio_path, target_sr=16000)

        # Acoustically aligned waveform preprocessing (silence trim, peak norm, bilateral collar)
        waveform = preprocess_waveform(
            waveform,
            sr=sr,
            pre_roll_pad_ms=self.pre_roll_pad_ms,
            post_roll_pad_ms=self.post_roll_pad_ms,
            peak_normalize=self.peak_normalize,
            trim_silence=self.trim_silence,
            vad_db_threshold=self.vad_db_threshold,
            vad_margin_ms=self.vad_margin_ms,
        )

        # Extract 80-channel log-mel features
        input_features = self.processor.feature_extractor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
        ).input_features[0]

        # Tokenize target transcript
        target_text = item.get("audio_text", "").strip()
        labels = self.processor.tokenizer(target_text).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].float()

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss in CrossEntropyLoss
        attention_mask = (
            labels_batch["attention_mask"]
            if isinstance(labels_batch, dict) or "attention_mask" in labels_batch
            else getattr(labels_batch, "attention_mask", None)
        )
        if attention_mask is not None:
            labels = labels_batch["input_ids"].masked_fill(
                attention_mask.ne(1), -100
            )
        else:
            labels = labels_batch["input_ids"]

        # If decoder_start_token_id is present at onset, slice it off as model generate/decoder adds it
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Audio-Tower-Only PhoWhisper LoRA on LifeMate Train Audio")
    parser.add_argument("--model-id", type=str, default="kzap201/phowhisper-large-zaloai")
    parser.add_argument("--train-jsonl", type=str, default="data/train/train.jsonl")
    parser.add_argument("--splits-json", type=str, default="configs/splits/dev_v1.json")
    parser.add_argument("--folds-json", type=str, default="configs/splits/train_only_5folds.json")
    parser.add_argument("--val-fold", type=int, default=None, help="Optional fold index 0-4 for validation, or None for full train")
    parser.add_argument("--include-history-audio", action="store_true", default=True)
    parser.add_argument("--no-history-audio", dest="include_history_audio", action="store_false")
    parser.add_argument("--data-root", type=str, default="data/train")
    parser.add_argument("--output-dir", type=str, default="experiments/phowhisper_lora_loop42")
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=15)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--pre-roll-pad-ms", type=float, default=20.0)
    parser.add_argument("--post-roll-pad-ms", type=float, default=60.0)
    parser.add_argument("--target-mode", type=str, choices=["encoder_only", "encoder_cross_attn"], default="encoder_cross_attn")
    parser.add_argument("--peak-normalize", action="store_true", default=True)
    parser.add_argument("--no-peak-normalize", dest="peak_normalize", action="store_false")
    parser.add_argument("--trim-silence", action="store_true", default=True)
    parser.add_argument("--no-trim-silence", dest="trim_silence", action="store_false")
    parser.add_argument("--vad-db-threshold", type=float, default=-35.0)
    parser.add_argument("--vad-margin-ms", type=float, default=170.0)
    parser.add_argument("--config", type=str, default=None, help="Optional path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 step dry-run test")
    args = parser.parse_args()

    if args.config and Path(args.config).exists():
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if "model_id" in cfg and parser.get_default("model_id") == args.model_id:
            args.model_id = cfg["model_id"]
        if "train_jsonl" in cfg and parser.get_default("train_jsonl") == args.train_jsonl:
            args.train_jsonl = cfg["train_jsonl"]
        if "splits_json" in cfg and parser.get_default("splits_json") == args.splits_json:
            args.splits_json = cfg["splits_json"]
        if "folds_json" in cfg and parser.get_default("folds_json") == args.folds_json:
            args.folds_json = cfg["folds_json"]
        if "output_dir" in cfg and parser.get_default("output_dir") == args.output_dir:
            args.output_dir = cfg["output_dir"]
        lora_cfg = cfg.get("lora") or {}
        if "r" in lora_cfg:
            args.lora_r = int(lora_cfg["r"])
        if "alpha" in lora_cfg:
            args.lora_alpha = int(lora_cfg["alpha"])
        if "dropout" in lora_cfg:
            args.lora_dropout = float(lora_cfg["dropout"])
        if "target_mode" in lora_cfg:
            args.target_mode = str(lora_cfg["target_mode"])
        train_cfg = cfg.get("training") or {}
        if "epochs" in train_cfg:
            args.epochs = int(train_cfg["epochs"])
        if "learning_rate" in train_cfg:
            args.learning_rate = float(train_cfg["learning_rate"])
        if "batch_size" in train_cfg:
            args.batch_size = int(train_cfg["batch_size"])
        if "gradient_accumulation_steps" in train_cfg:
            args.gradient_accumulation_steps = int(train_cfg["gradient_accumulation_steps"])
        if "weight_decay" in train_cfg:
            args.weight_decay = float(train_cfg["weight_decay"])
        if "warmup_steps" in train_cfg:
            args.warmup_steps = int(train_cfg["warmup_steps"])
        if "fp16" in train_cfg:
            args.fp16 = bool(train_cfg["fp16"])
        if "include_history_audio" in train_cfg:
            args.include_history_audio = bool(train_cfg["include_history_audio"])

    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Dev Split IDs strictly (Zero Dev Leakage Invariant)
    splits_path = Path(args.splits_json)
    dev_ids = set()
    if splits_path.exists():
        with open(splits_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            dev_ids = set(raw.get("dev_ids", raw) if isinstance(raw, dict) else raw)
    logger.info(f"Loaded {len(dev_ids)} sealed Dev IDs to exclude from training")

    val_ids = set()
    if args.val_fold is not None and args.folds_json and Path(args.folds_json).exists():
        with open(args.folds_json, "r", encoding="utf-8") as f:
            folds_data = json.load(f)
            val_ids = set(folds_data.get("folds", {}).get(f"fold_{args.val_fold}", []))
        logger.info(f"Held out Fold {args.val_fold} with {len(val_ids)} samples for OOF validation")

    # 2. Filter train.jsonl and extract all in-domain audio utterances (main + history)
    train_jsonl_path = Path(args.train_jsonl)
    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []

    with open(train_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sid = item.get("id")
            if sid in dev_ids:
                continue

            target_list = val_samples if sid in val_ids else train_samples

            # Main turn audio
            main_audio = item.get("audio", "")
            main_text = item.get("audio_text", "").strip()
            if main_audio and main_text:
                target_list.append({
                    "id": sid,
                    "audio": main_audio,
                    "audio_text": main_text,
                })

            # History user audio turns
            if args.include_history_audio:
                for h_idx, turn in enumerate(item.get("history") or []):
                    h_audio = turn.get("audio")
                    h_text = (turn.get("audio_text") or turn.get("text") or "").strip()
                    if h_audio and h_text:
                        target_list.append({
                            "id": f"{sid}_h{h_idx}",
                            "audio": h_audio,
                            "audio_text": h_text,
                        })

    if args.max_train_samples:
        train_samples = train_samples[: args.max_train_samples]

    logger.info(f"Selected {len(train_samples)} Train audio utterances (and {len(val_samples)} Val audio utterances) for PhoWhisper LoRA")

    # 3. Load Processor and Model
    logger.info(f"Loading processor for {args.model_id}")
    processor = load_processor(args.model_id)

    logger.info(f"Loading Whisper backbone: {args.model_id}")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # Bulletproof: ensure conv1/conv2 in audio tower are float32 to match feature_extractor output
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        if hasattr(model.model.encoder, "conv1"):
            model.model.encoder.conv1 = model.model.encoder.conv1.float()
        if hasattr(model.model.encoder, "conv2"):
            model.model.encoder.conv2 = model.model.encoder.conv2.float()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Configure Whisper for Vietnamese generation
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # Required when gradient checkpointing is enabled

    # 4. Explicitly freeze 100% of the Decoder to preserve language model prior
    logger.info("Freezing 100% of the Decoder parameters to prevent overfitting...")
    decoder = getattr(model, "model", model).decoder if hasattr(getattr(model, "model", model), "decoder") else None
    if decoder is not None:
        for param in decoder.parameters():
            param.requires_grad = False

    # 4. Target module selection for LoRA adaptation
    target_modules: List[str] = []
    for name, _ in model.named_modules():
        if name.endswith(("q_proj", "v_proj")):
            if "encoder.layers" in name and "self_attn" in name:
                target_modules.append(name)
            elif (
                args.target_mode == "encoder_cross_attn"
                and "decoder.layers" in name
                and "encoder_attn" in name
            ):
                target_modules.append(name)

    logger.info(f"Selected {len(target_modules)} explicit LoRA target modules for mode '{args.target_mode}'")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    # Freeze decoder autoregressive self-attention and FFN layers to preserve language model prior
    decoder = getattr(model, "model", model).decoder if hasattr(getattr(model, "model", model), "decoder") else None
    if decoder is not None:
        for name, param in decoder.named_parameters():
            if "encoder_attn" not in name:
                param.requires_grad = False

    # Verify trainable parameters
    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    logger.info(f"Verified {len(trainable_params)} trainable LoRA tensors:")
    for name in trainable_params:
        assert ("encoder" in name or "encoder_attn" in name), f"Trainable parameter found outside acoustic path: {name}"
        assert ("decoder.layers" not in name or "encoder_attn" in name), f"Decoder self-attention or FFN unconstitutionally unfrozen: {name}"

    model.print_trainable_parameters()

    # 5. Build Dataset and Collator with aligned acoustic preprocessing
    data_root = Path(args.data_root)
    train_dataset = LifeMateASRDataset(
        train_samples,
        data_root,
        processor,
        pre_roll_pad_ms=args.pre_roll_pad_ms,
        post_roll_pad_ms=args.post_roll_pad_ms,
        peak_normalize=args.peak_normalize,
        trim_silence=args.trim_silence,
        vad_db_threshold=args.vad_db_threshold,
        vad_margin_ms=args.vad_margin_ms,
    )
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # 6. Training Arguments
    use_fp16 = torch.cuda.is_available() and args.fp16
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size if not args.dry_run else 1,
        gradient_accumulation_steps=args.gradient_accumulation_steps if not args.dry_run else 1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs if not args.dry_run else 1,
        max_steps=1 if args.dry_run else -1,
        gradient_checkpointing=True if not args.dry_run else False,
        fp16=use_fp16,
        logging_steps=10 if not args.dry_run else 1,
        save_strategy="epoch" if not args.dry_run else "no",
        save_total_limit=2,
        lr_scheduler_type="cosine",
        dataloader_num_workers=0,
        seed=args.seed,
        report_to=["none"],
        remove_unused_columns=False,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": collator,
    }
    try:
        trainer = Seq2SeqTrainer(**trainer_kwargs, processing_class=processor.feature_extractor)
    except TypeError:
        trainer = Seq2SeqTrainer(**trainer_kwargs, tokenizer=processor.feature_extractor)

    logger.info("Starting PhoWhisper LoRA training...")
    trainer.train()

    logger.info(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info("PhoWhisper LoRA training complete.")


if __name__ == "__main__":
    main()
