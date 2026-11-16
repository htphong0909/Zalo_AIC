#!/usr/bin/env python3
"""Download and verify the two fine-tuned offline models into /code/saved_models/.

Only fine-tuned / inference weights are stored inside the Docker image (base model Qwen/Qwen3.5-4B
and vinai/PhoWhisper-large are intentionally excluded per submission guidelines).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

CODE_ROOT = Path(__file__).resolve().parent
LLM_REPO = "htphong0909/zalo-aic-sft-checkpoints"
LLM_PREFIX = "exp_qwen35_v2r_general_lora/merged"
ASR_REPO = "kzap201/phowhisper-large-zaloai"


def count_safetensors_params(model_dir: Path) -> int:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")
    total = 0
    for sf in files:
        with open(sf, "rb") as f:
            hdr_len = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hdr_len).decode("utf-8"))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            n = 1
            for dim in v["shape"]:
                n *= int(dim)
            total += n
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-dir", default=str(CODE_ROOT / "saved_models"))
    parser.add_argument("--skip-offline-load", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    saved_dir = Path(args.saved_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)

    llm_dest = saved_dir / "qwen35_sft_v2_merged"
    asr_dest = saved_dir / "phowhisper-large-zaloai"

    if not (llm_dest / ".complete").exists() or not (llm_dest / "model.safetensors").exists():
        tmp_dl = saved_dir / "_tmp_llm_dl"
        shutil.rmtree(tmp_dl, ignore_errors=True)
        print(f"[prepare_models] Downloading {LLM_REPO}/{LLM_PREFIX} -> {llm_dest} ...", flush=True)
        snapshot_download(
            repo_id=LLM_REPO,
            repo_type="model",
            allow_patterns=f"{LLM_PREFIX}/*",
            local_dir=str(tmp_dl),
            token=token,
        )
        shutil.rmtree(llm_dest, ignore_errors=True)
        shutil.move(str(tmp_dl / LLM_PREFIX), str(llm_dest))
        shutil.rmtree(tmp_dl, ignore_errors=True)
    print(f"[prepare_models] LLM ready: {llm_dest}", flush=True)

    if not (asr_dest / "model.safetensors").exists() or not (asr_dest / "config.json").exists():
        print(f"[prepare_models] Downloading {ASR_REPO} -> {asr_dest} ...", flush=True)
        shutil.rmtree(asr_dest, ignore_errors=True)
        snapshot_download(
            repo_id=ASR_REPO,
            repo_type="model",
            local_dir=str(asr_dest),
            token=token,
        )
    print(f"[prepare_models] ASR ready: {asr_dest}", flush=True)

    llm_params = count_safetensors_params(llm_dest)
    asr_params = count_safetensors_params(asr_dest)
    total_params = llm_params + asr_params
    print(f"[params] LLM (Qwen3.5-4B SFT V2 merged): {llm_params:,} ({llm_params / 1e9:.4f}B)")
    print(f"[params] ASR (PhoWhisper-Large-ZaloAI): {asr_params:,} ({asr_params / 1e9:.4f}B)")
    print(f"[params] TOTAL E2E:                     {total_params:,} ({total_params / 1e9:.4f}B)")
    assert llm_params < 5_000_000_000, f"LLM params {llm_params} exceeds 5B!"
    assert total_params < 6_000_000_000, f"Total params {total_params} exceeds 6B!"

    if not args.skip_offline_load:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(llm_dest), trust_remote_code=True, local_files_only=True)
        cfg = AutoConfig.from_pretrained(str(llm_dest), trust_remote_code=True, local_files_only=True)
        proc = AutoProcessor.from_pretrained(str(asr_dest), local_files_only=True)
        print(
            f"[offline-verify] Tokenizer={type(tok).__name__}, Arch={cfg.architectures}, "
            f"Processor={type(proc).__name__} OK",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
