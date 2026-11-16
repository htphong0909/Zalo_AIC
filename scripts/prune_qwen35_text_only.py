#!/usr/bin/env python3
"""Export a Qwen3.5 / Qwen3.8-4B-Distill checkpoint as pure text-only Qwen3_5ForCausalLM weights.

Drops unused vision tower (`model.visual.*`, 333.5M) and MTP head (`mtp.*`, 120.6M) so the
on-disk checkpoint contains only `model.language_model` (~4.21B params), keeping total E2E
system size strictly < 6B when paired with PhoWhisper-Large (~1.54B).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.qwen35_sft import save_qwen35_text_only_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="empero-ai/Qwen3.8-4B-Distill")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    torch_dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    out = save_qwen35_text_only_checkpoint(args.model_id, tok, args.out, dtype=torch_dtype)
    print(f"Exported pure text-only Qwen3_5ForCausalLM checkpoint -> {out}")


if __name__ == "__main__":
    main()
