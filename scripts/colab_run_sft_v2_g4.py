#!/usr/bin/env python3
"""
Orchestrate SFT V2 Training (3 Epochs, 4,594 verified records) and Evaluation on Colab G4 (NVIDIA RTX PRO 6000 Blackwell 96GB VRAM).
Integrates directly with scripts/colab_runner.py for real-time streaming and self-healing.
Checkpoints pushed every ~10 minutes to Hugging Face (htphong0909/zalo-aic-sft-checkpoints).
"""
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.colab_runner import (
    package_codebase,
    ping_session,
    run_command,
    restart_kernel,
)
from src.core.hf_sync import ENV_REPO, ENV_TOKEN, HFSyncer, HFSyncError, resolve_repo_id

SESSION = os.environ.get("COLAB_SESSION", "minicpm5-g4")
GPU = "G4"

SCRATCH = ROOT / "scripts" / "scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)

UPLOADS = [
    (SCRATCH / "code_latest.zip", "/content/code_latest.zip"),
    (ROOT / "data" / "sft" / "minicpm5_train_sft_v2.jsonl.gz", "/content/minicpm5_train_sft_v2.jsonl.gz"),
    (ROOT / "data" / "train" / "train_tools_catalog.json", "/content/train_tools_catalog.json"),
    (SCRATCH / "train.jsonl.gz", "/content/train.jsonl.gz"),
    (SCRATCH / "train_environment.zip", "/content/train_environment.zip"),
    (SCRATCH / "zaic_tools.zip", "/content/zaic_tools.zip"),
    (ROOT / "configs" / "splits" / "dev_v1.json", "/content/dev_v1.json"),
]


def exec_local_cmd(cmd: list[str], check: bool = True, timeout: int = 180) -> subprocess.CompletedProcess:
    print(f"[colab_g4] Running: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if res.stdout:
        print(res.stdout, flush=True)
    if res.stderr:
        print(res.stderr, file=sys.stderr, flush=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {' '.join(cmd)}")
    return res


DEFAULT_HF_REPO = "htphong0909/zalo-aic-sft-checkpoints"


def hf_preflight_or_die() -> tuple[str, str]:
    """AGENTS.md 4.3: never start a GPU job unless durable HF checkpoint sync is proven to work."""
    from huggingface_hub import get_token

    repo = resolve_repo_id() or DEFAULT_HF_REPO
    token = os.environ.get(ENV_TOKEN) or get_token()
    if not repo or not token:
        sys.exit(
            f"ABORT: set {ENV_REPO}=<user>/<repo> and {ENV_TOKEN}=<write token> (or run `hf auth login`) before running. "
            "Without them checkpoints would only live on the Colab VM and be lost on a crash."
        )
    try:
        user = HFSyncer(repo, token=token, retries=5).preflight()
    except HFSyncError as exc:
        sys.exit(f"ABORT (HF preflight): {exc}")
    print(f"HF preflight OK as '{user}' -> {repo}", flush=True)
    return repo, token


def main():
    print("=" * 80, flush=True)
    print("ORCHESTRATING SFT V2 EXPERIMENT ON COLAB G4 (RTX PRO 6000 BLACKWELL 96GB VRAM)", flush=True)
    print("Model: openbmb/MiniCPM5-2B-SFT (Branch A Winner, 3.0 Epochs, lr=1.5e-4)")
    print("Dataset: minicpm5_train_sft_v2.jsonl.gz (4,594 records, zero dev leakage)")
    print("Self-healing streaming runner enabled via scripts/colab_runner.py", flush=True)
    print("=" * 80, flush=True)

    hf_repo, hf_token = hf_preflight_or_die()

    # 1. Package latest code cleanly
    print("\n[1/7] Packaging codebase via colab_runner...", flush=True)
    pkg_zip = SCRATCH / "code_latest.zip"
    package_codebase(pkg_zip)

    # 2. Check active sessions or create G4 session
    print(f"\n[2/7] Checking Colab session '{SESSION}' (GPU={GPU})...", flush=True)
    check_res = exec_local_cmd(["colab", "sessions"], check=False)
    if SESSION not in check_res.stdout:
        print(f"Creating new Colab session '{SESSION}' with GPU '{GPU}'...", flush=True)
        exec_local_cmd(["colab", "new", "-s", SESSION, "--gpu", GPU])
        time.sleep(8)
    else:
        print(f"Using existing session '{SESSION}'.", flush=True)

    # 3. Ping session to ensure kernel is responsive
    print(f"\n[3/7] Pinging Colab session '{SESSION}'...", flush=True)
    alive = ping_session(SESSION, timeout=20)
    if not alive:
        print("[colab_g4] Kernel not responsive, restarting kernel...", flush=True)
        restart_kernel(SESSION)
        alive = ping_session(SESSION, timeout=20)
        if not alive:
            raise RuntimeError(f"Colab session '{SESSION}' failed to respond to ping!")
    print(f"Colab session '{SESSION}' is healthy and responsive.", flush=True)

    # 4. Upload assets
    print(f"\n[4/7] Uploading assets to Colab G4 session '{SESSION}'...", flush=True)
    for local_p, remote_p in UPLOADS:
        if not local_p.exists():
            raise FileNotFoundError(f"Required asset not found: {local_p}")
        size_mb = local_p.stat().st_size / (1024 * 1024)
        print(f"  Uploading {local_p.name} ({size_mb:.2f} MB) -> {remote_p}...", flush=True)
        exec_local_cmd(["colab", "upload", "-s", SESSION, str(local_p), remote_p])

    # 5. Execute remote worker
    worker_script = SCRATCH / "worker_sft_v2_g4.py"
    print(f"\n[5/7] Executing SFT V2 on Colab G4 96GB (real-time stream, ~10m HF sync)...", flush=True)
    t0 = time.time()
    exec_res = run_command(
        str(worker_script),
        session=SESSION,
        is_file=True,
        timeout=7200,
        max_retries=2,
        env=[f"{ENV_TOKEN}={hf_token}", f"{ENV_REPO}={hf_repo}"],
        stream=True,
        gpu=GPU,
    )
    elapsed = time.time() - t0
    print(f"\nSFT V2 completed on G4 in {elapsed:.1f}s ({elapsed / 60:.2f} min).", flush=True)
    if exec_res.returncode != 0:
        raise RuntimeError(f"Remote SFT V2 worker failed with code {exec_res.returncode}!")

    # 6. Download results
    print("\n[6/7] Downloading SFT V2 results...", flush=True)
    out_dir = ROOT / "experiments" / "exp_20261111_minicpm5_sft_v2_g4"
    out_dir.mkdir(parents=True, exist_ok=True)
    local_zip = out_dir / "sft_v2_results.zip"

    exec_local_cmd(["colab", "download", "-s", SESSION, "/content/sft_v2_results.zip", str(local_zip)])
    if local_zip.exists():
        with zipfile.ZipFile(local_zip, "r") as z:
            z.extractall(str(out_dir))
        print(f"Extracted results to: {out_dir}", flush=True)

    # 7. Stop session cleanly to prevent CU burn
    print(f"\n[7/7] Stopping session '{SESSION}' to preserve compute units...", flush=True)
    exec_local_cmd(["colab", "stop", "-s", SESSION], check=False)
    print("=" * 80, flush=True)
    print("STAGE 1.5 SFT V2 EXPERIMENT ON G4 COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
