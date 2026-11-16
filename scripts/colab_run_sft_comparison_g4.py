#!/usr/bin/env python3
"""Orchestrate Head-to-Head MiniCPM5 SFT Comparison (2B-SFT vs 2B-Instruct) on Colab G4 (96GB)."""
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SESSION = "minicpm5-g4"
GPU = "G4"

uploads = [
    (ROOT / "scripts" / "scratch" / "code_latest.zip", "/content/code_latest.zip"),
    (ROOT / "data" / "sft" / "minicpm5_train_sft.jsonl.gz", "/content/minicpm5_train_sft.jsonl.gz"),
    (ROOT / "data" / "train" / "train_tools_catalog.json", "/content/train_tools_catalog.json"),
    (ROOT / "scripts" / "scratch" / "train.jsonl.gz", "/content/train.jsonl.gz"),
    (ROOT / "scripts" / "scratch" / "train_environment.zip", "/content/train_environment.zip"),
    (ROOT / "scripts" / "scratch" / "zaic_tools.zip", "/content/zaic_tools.zip"),
    (ROOT / "configs" / "splits" / "dev_v1.json", "/content/dev_v1.json"),
]


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"[colab_g4] Running: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.stdout:
        print(res.stdout, flush=True)
    if res.stderr:
        print(res.stderr, file=sys.stderr, flush=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {' '.join(cmd)}")
    return res


def main():
    print("=" * 80, flush=True)
    print(f"ORCHESTRATING MINICPM5 SFT COMPARISON EXPERIMENT ON COLAB G4 (96GB VRAM)", flush=True)
    print("=" * 80, flush=True)

    # 1. Package latest code
    subprocess.run([sys.executable, str(ROOT / "scripts" / "package_code.py")], check=True)

    # 2. Check active sessions or create G4 session
    check_res = run_cmd(["colab", "sessions"], check=False)
    if SESSION not in check_res.stdout:
        print(f"Creating new Colab session '{SESSION}' with GPU '{GPU}'...", flush=True)
        run_cmd(["colab", "new", "-s", SESSION, "--gpu", GPU])
        time.sleep(5)
    else:
        print(f"Using existing session '{SESSION}'.", flush=True)

    # 3. Upload assets
    print("\nUploading assets to Colab G4...", flush=True)
    for local_p, remote_p in uploads:
        assert local_p.exists(), f"File not found: {local_p}"
        size_mb = local_p.stat().st_size / (1024 * 1024)
        print(f"  Uploading {local_p.name} ({size_mb:.2f} MB) -> {remote_p}...", flush=True)
        run_cmd(["colab", "upload", "-s", SESSION, str(local_p), remote_p])

    # 4. Execute remote worker
    print("\nExecuting SFT comparison worker on Colab G4...", flush=True)
    t0 = time.time()
    worker_script = ROOT / "scripts" / "scratch" / "worker_sft_comparison.py"
    exec_res = run_cmd(["colab", "exec", "-s", SESSION, "-f", str(worker_script), "--timeout", "3600"])
    elapsed = time.time() - t0
    print(f"\nRemote execution finished in {elapsed:.1f}s ({elapsed / 60:.2f} min).", flush=True)

    # 5. Download results
    out_dir = ROOT / "experiments" / "exp_20261110_minicpm5_sft_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    local_zip = out_dir / "sft_comparison_results.zip"

    print(f"\nDownloading comparison results -> {local_zip}...", flush=True)
    run_cmd(["colab", "download", "-s", SESSION, "/content/sft_comparison_results.zip", str(local_zip)])

    # 6. Unpack results
    if local_zip.exists():
        with zipfile.ZipFile(local_zip, "r") as z:
            z.extractall(str(out_dir))
        print(f"Extracted results to: {out_dir}", flush=True)

    # 7. Stop session cleanly to prevent CU burn
    print(f"\nStopping session '{SESSION}' to preserve compute units...", flush=True)
    run_cmd(["colab", "stop", "-s", SESSION], check=False)
    print("Done! Option 1 SFT comparison workflow completed cleanly.", flush=True)


if __name__ == "__main__":
    main()
