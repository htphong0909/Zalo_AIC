#!/bin/bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PATH="/opt/venv/bin:${PATH}"

mkdir -p /result
python3 /code/predict.py "$@"
