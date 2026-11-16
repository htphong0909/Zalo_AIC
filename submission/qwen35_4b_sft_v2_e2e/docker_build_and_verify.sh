#!/bin/bash
set -euo pipefail

# End-to-end Docker build, offline verification, Jupyter notebook verification, and tarball export
# Usage: ./docker_build_and_verify.sh <TEAM_NAME> <PATH_TO_DATA_DIR>
# Example: ./docker_build_and_verify.sh LifeMate /abs/path/to/ZALO_AIC/data

TEAM_NAME="${1:-67Cat}"
DATA_DIR="${2:?Usage: $0 <TEAM_NAME> <PATH_TO_DATA_DIR>}"
IMAGE_TAG="zac2026:$(echo "${TEAM_NAME}" | tr '[:upper:]' '[:lower:]')"
TAR_NAME="zac2026_${TEAM_NAME}.tar.gz"
RESULT_DIR="$(pwd)/_docker_verify_result"

echo "=== [1/7] Prepare offline fine-tuned models (NO base models bundled) ==="
python3 prepare_models.py --saved-dir saved_models

echo "=== [2/7] Build Docker image ${IMAGE_TAG} (CUDA 13.2.0) ==="
docker build --network=host -t "${IMAGE_TAG}" .

echo "=== [3/7] Test 10.1: Offline predict.sh inference (--network none, /data:ro) ==="
rm -rf "${RESULT_DIR}" && mkdir -p "${RESULT_DIR}"
docker run --rm \
  --network none \
  --gpus '"device=0"' \
  -v "${DATA_DIR}:/data:ro" \
  -v "${RESULT_DIR}:/result" \
  "${IMAGE_TAG}" /bin/bash /code/predict.sh

echo "=== [4/7] Validate /result/submission.jsonl with split zaic_tools ==="
SPLIT_NAME="$(python3 -c "import pathlib; r=pathlib.Path('${DATA_DIR}'); print(next(p.name for p in sorted(r.iterdir()) if p.is_dir() and (p / f'{p.name}_tools_catalog.json').exists()))")"
PYTHONPATH="${DATA_DIR}/${SPLIT_NAME}" python3 -m zaic_tools \
  --split "${SPLIT_NAME}" \
  --root "${DATA_DIR}" \
  --validate "${RESULT_DIR}/submission.jsonl"

echo "=== [5/7] Test 10.2 & Appendix: Execute predict_notebook.ipynb offline + verify start_jupyter.sh ==="
docker run --rm \
  --network none \
  --gpus '"device=0"' \
  -v "${DATA_DIR}:/data:ro" \
  -v "${RESULT_DIR}:/result" \
  "${IMAGE_TAG}" /bin/bash -c "
    jupyter nbconvert --to notebook --execute /code/predict_notebook.ipynb --output /tmp/executed_predict.ipynb &&
    python3 -c '
import json
cli = {r[\"id\"]: r for r in map(json.loads, open(\"/result/submission.jsonl\")) if r}
nb = {r[\"id\"]: r for r in map(json.loads, open(\"/result/jupyter_submission.jsonl\")) if r}
assert set(cli) == set(nb), f\"ID mismatch: {len(cli)} vs {len(nb)}\"
diffs = [k for k in cli if cli[k][\"tool_calls\"] != nb[k][\"tool_calls\"] or cli[k][\"state_hash\"] != nb[k][\"state_hash\"]]
assert not diffs, f\"Parity diffs on {len(diffs)} samples: {diffs[:5]}\"
print(f\"PARITY OK: {len(cli)}/{len(cli)} samples 100% identical between predict.py and predict_notebook.ipynb\")
'
  "

echo "=== [6/7] Verify start_jupyter.sh port 9777 with token zac2026 ==="
CID="$(docker run -d --rm --network none --gpus '"device=0"' -p 9777:9777 "${IMAGE_TAG}" /bin/bash /code/start_jupyter.sh)"
trap 'docker stop "${CID}" >/dev/null 2>&1 || true' EXIT
sleep 5
curl -fsS "http://127.0.0.1:9777/api?token=zac2026"
echo ""
docker stop "${CID}" >/dev/null 2>&1 || true
trap - EXIT

echo "=== [7/7] Save compressed tarball ${TAR_NAME}, test docker load, compute MD5 ==="
docker save "${IMAGE_TAG}" | gzip > "${TAR_NAME}"
md5sum "${TAR_NAME}" | tee "${TAR_NAME}.md5"
echo "To re-verify docker load per Step 11.1: docker load -i ${TAR_NAME}"
