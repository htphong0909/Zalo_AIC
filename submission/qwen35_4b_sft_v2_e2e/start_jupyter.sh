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

cd /code
exec jupyter lab \
  --port=9777 \
  --ip=0.0.0.0 \
  --ServerApp.root_dir=/code \
  --IdentityProvider.token=zac2026 \
  --ServerApp.allow_remote_access=True \
  --allow-root \
  --no-browser
