#!/usr/bin/env python3
"""
Run step-by-step MiniCPM5 inspection on 2 Dev samples (L4 GPU).
Prints all raw thinking tokens, XML calls, and turn latencies, then stops session.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.colab_runner import (
    package_codebase,
    ping_session,
    run_command,
    restart_kernel,
)

SESSION = "minicpm5-inspect"
GPU = "L4"
SCRATCH = ROOT / "scripts" / "scratch"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

UPLOADS = [
    (SCRATCH / "code_latest.zip", "/content/code_latest.zip"),
    (ROOT / "data" / "train" / "train_tools_catalog.json", "/content/train_tools_catalog.json"),
    (SCRATCH / "train.jsonl.gz", "/content/train.jsonl.gz"),
    (SCRATCH / "train_environment.zip", "/content/train_environment.zip"),
    (SCRATCH / "zaic_tools.zip", "/content/zaic_tools.zip"),
]


def exec_local_cmd(cmd: list[str], check: bool = True, timeout: int = 180) -> subprocess.CompletedProcess:
    print(f"[colab_inspect] Running: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if res.stdout:
        print(res.stdout, flush=True)
    if res.stderr:
        print(res.stderr, file=sys.stderr, flush=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {' '.join(cmd)}")
    return res


def main():
    print("=" * 80, flush=True)
    print("LAUNCHING MINICPM5 STEP-BY-STEP INSPECTION ON COLAB L4", flush=True)
    print("=" * 80, flush=True)

    # 1. Package codebase
    pkg_zip = SCRATCH / "code_latest.zip"
    package_codebase(pkg_zip)

    # 2. Create Colab session
    print(f"\nCreating Colab session '{SESSION}' (GPU={GPU})...", flush=True)
    exec_local_cmd(["colab", "new", "-s", SESSION, "--gpu", GPU])
    time.sleep(10)

    # 3. Ping
    print(f"\nPinging session '{SESSION}'...", flush=True)
    alive = ping_session(SESSION, timeout=25)
    if not alive:
        restart_kernel(SESSION)
        alive = ping_session(SESSION, timeout=25)
        if not alive:
            raise RuntimeError("Colab session failed to respond to ping!")
    print(f"Session '{SESSION}' is healthy.", flush=True)

    # 4. Upload assets
    print("\nUploading assets...", flush=True)
    for local_p, remote_p in UPLOADS:
        if not local_p.exists():
            raise FileNotFoundError(f"Missing {local_p}")
        exec_local_cmd(["colab", "upload", "-s", SESSION, str(local_p), remote_p])

    # 5. Run inspection script
    print("\nExecuting inspection script...", flush=True)
    script_p = SCRATCH / "inspect_minicpm5_sample.py"
    t0 = time.time()
    try:
        res = run_command(
            str(script_p),
            session=SESSION,
            is_file=True,
            timeout=600,
            env=[f"HF_TOKEN={HF_TOKEN}"],
        )
        print(f"\nInspection finished in {time.time() - t0:.1f}s.", flush=True)
        if res.stdout:
            print("\n" + "#" * 80)
            print("INSPECTION OUTPUT:")
            print("#" * 80)
            print(res.stdout)
        if res.stderr:
            print(res.stderr, file=sys.stderr)
    finally:
        # 6. Stop session to save CUs
        print(f"\nStopping session '{SESSION}'...", flush=True)
        exec_local_cmd(["colab", "stop", "-s", SESSION], check=False)


if __name__ == "__main__":
    main()
