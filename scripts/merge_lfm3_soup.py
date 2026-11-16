"""Model Soup (ICML 2022 / TIES) weight averaging across 5-fold LFM3-1.7B checkpoints.

Merges fine-tuned LoRA checkpoints from multiple folds into a single standalone 1.73B model:
W_soup = (1 / K) * sum_{k=1}^K W_k
Preserves exact 1.73B parameter size and zero inference latency overhead.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("merge_lfm3_soup")


def average_safetensors_weights(
    checkpoint_dirs: Sequence[str | Path],
    output_dir: str | Path,
    weights_filename: str = "adapter_model.safetensors",
) -> Path:
    """Compute uniform linear average of safetensors weights across checkpoints."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / weights_filename

    K = len(checkpoint_dirs)
    if K == 0:
        raise ValueError("checkpoint_dirs must not be empty")

    logger.info(f"Averaging {K} checkpoints into {out_file}")

    # Load first checkpoint to determine parameter keys and shapes
    first_file = Path(checkpoint_dirs[0]) / weights_filename
    if not first_file.exists():
        raise FileNotFoundError(f"Weight file not found: {first_file}")

    base_tensors = load_file(str(first_file))
    averaged: dict[str, torch.Tensor] = {}

    for k, tensor in base_tensors.items():
        averaged[k] = tensor.clone().float() / K

    # Accumulate subsequent checkpoints
    for c_dir in checkpoint_dirs[1:]:
        c_file = Path(c_dir) / weights_filename
        if not c_file.exists():
            raise FileNotFoundError(f"Weight file not found: {c_file}")
        tensors = load_file(str(c_file))
        for k in averaged.keys():
            averaged[k] += tensors[k].float() / K

    # Convert back to original dtypes
    final_tensors: dict[str, torch.Tensor] = {}
    for k, tensor in averaged.items():
        orig_dtype = base_tensors[k].dtype
        final_tensors[k] = tensor.to(orig_dtype)

    save_file(final_tensors, str(out_file))
    logger.info(f"Successfully saved averaged soup weights ({len(final_tensors)} tensors) to {out_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint directories to merge")
    parser.add_argument("--output-dir", required=True, help="Directory to save merged soup checkpoint")
    parser.add_argument("--weights-filename", default="adapter_model.safetensors")
    args = parser.parse_args()

    average_safetensors_weights(
        checkpoint_dirs=args.checkpoints,
        output_dir=args.output_dir,
        weights_filename=args.weights_filename,
    )


if __name__ == "__main__":
    main()
