#!/usr/bin/env python3
"""
Production-grade Kaggle Batch Kernel Execution Runner for ZALO_AIC (LifeMate).
Enables headless, non-interactive remote GPU execution on Kaggle as an alternative
to Google Colab when compute units are exhausted.

Features:
1. Safe packaging: bundles src, configs, scripts, tests without heavy data/experiments.
2. Kernel Staging & Push: prepares kernel-metadata.json and pushes via Kaggle CLI.
3. Heartbeat Watchdog: polls `kaggle kernels status` with timestamped logs.
4. Output Retrieval: downloads predictions, metrics, and logs from /kaggle/working/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "scripts" / "scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)

DEFAULT_USERNAME = os.environ.get("KAGGLE_USERNAME", "xunthanhh")
DEFAULT_PING_SLUG = f"{DEFAULT_USERNAME}/zalo-aic-ping-gpu"
DEFAULT_EVAL_SLUG = f"{DEFAULT_USERNAME}/zalo-aic-eval"
DEFAULT_TIMEOUT = 1800  # 30 minutes


def get_kaggle_cmd() -> List[str]:
    """Detects and returns the command invocation list for Kaggle CLI."""
    k_path = shutil.which("kaggle")
    if k_path:
        return [k_path]

    if sys.platform == "win32":
        py_scripts = Path(sys.executable).parent / "Scripts" / "kaggle.exe"
        if py_scripts.exists():
            return [str(py_scripts)]

    return [sys.executable, "-m", "kaggle"]


def package_codebase(output_zip: Path) -> int:
    """Safely packages clean code without risk of recursive zip bombs or heavy data."""
    exclude_dirs = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        "archive",
        "temp",
        "reports",
        "data",
        "weights",
        "img",
        ".agents",
        "scratch",
        ".user_uploaded",
        ".system_generated",
        "experiments",
        "outputs",
        ".cache",
    }
    if output_zip.exists():
        output_zip.unlink()

    file_count = 0
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ["src", "tests", "configs", "scripts"]:
            folder_path = ROOT / folder
            if not folder_path.exists():
                continue
            for p in folder_path.rglob("*"):
                if (
                    p.is_file()
                    and p.suffix != ".zip"
                    and not any(part in exclude_dirs for part in p.parts)
                ):
                    z.write(p, p.relative_to(ROOT))
                    file_count += 1

    size_kb = output_zip.stat().st_size / 1024.0
    print(
        f"[kaggle_runner] Packaged {file_count} files ({size_kb:.1f} KB) into {output_zip.name}",
        flush=True,
    )
    return file_count


def create_kernel_metadata(
    dest_path: Path,
    kernel_slug: str,
    title: str,
    code_file: str,
    enable_gpu: bool = True,
    enable_internet: bool = True,
    kernel_type: str = "script",
    language: str = "python",
) -> Dict[str, Any]:
    """Generates kernel-metadata.json expected by Kaggle CLI."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": kernel_slug,
        "title": title,
        "code_file": code_file,
        "language": language,
        "kernel_type": kernel_type,
        "is_private": "true",
        "enable_gpu": "true" if enable_gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true" if enable_internet else "false",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    dest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_kernel_status(output_text: str) -> Tuple[str, bool, bool]:
    """
    Parses output of `kaggle kernels status <slug>`.
    Returns: (status: str, is_done: bool, is_error: bool)
    """
    cleaned = output_text.strip().lower()
    if "complete" in cleaned:
        return "complete", True, False
    if "error" in cleaned:
        return "error", True, True
    if "running" in cleaned:
        return "running", False, False
    if "queued" in cleaned:
        return "queued", False, False
    return "unknown", False, False


def push_kernel(staging_dir: Path, timeout: int = 60) -> Tuple[bool, str, Optional[str]]:
    """
    Pushes kernel staging directory to Kaggle.
    Returns: (success: bool, combined_output: str, resolved_slug: Optional[str])
    """
    cmd = get_kaggle_cmd() + ["kernels", "push", "-p", str(staging_dir)]
    print(f"[kaggle_runner] Executing: {' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        combined = (proc.stdout + "\n" + proc.stderr).strip()
        print(f"[kaggle_runner] Push output:\n{combined}", flush=True)

        resolved_slug = None
        match = re.search(r"kaggle\.com/code/([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+)", combined)
        if match:
            resolved_slug = match.group(1).strip()
            print(f"[kaggle_runner] Detected actual kernel slug: '{resolved_slug}'", flush=True)

        return proc.returncode == 0, combined, resolved_slug
    except Exception as e:
        print(f"[kaggle_runner] Push failed with exception: {e}", flush=True)
        return False, str(e), None


def poll_kernel_status(
    kernel_slug: str,
    poll_interval: int = 15,
    timeout: int = DEFAULT_TIMEOUT,
    verbose: bool = True,
) -> Tuple[str, str]:
    """
    Polls kernel status until complete, error, or timeout.
    Returns: (final_status, log_trace)
    """
    start_time = time.time()
    cmd = get_kaggle_cmd() + ["kernels", "status", kernel_slug]
    print(f"[kaggle_runner] Polling status for '{kernel_slug}' (timeout={timeout}s)...", flush=True)

    trace_lines = []
    last_status = "queued"

    while (time.time() - start_time) < timeout:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            raw = (proc.stdout + "\n" + proc.stderr).strip()
            elapsed = time.time() - start_time
            status, is_done, is_error = parse_kernel_status(raw)
            line = f"[{elapsed:5.1f}s] Status: {status} | raw: {raw}"
            trace_lines.append(line)
            if verbose and (status != last_status or elapsed % 30 < poll_interval):
                print(f"[kaggle_runner] {line}", flush=True)
            last_status = status

            if is_done:
                print(f"[kaggle_runner] Kernel reached terminal state: '{status}' in {elapsed:.1f}s", flush=True)
                return status, "\n".join(trace_lines)

        except Exception as e:
            trace_lines.append(f"Poll check exception: {e}")
            if verbose:
                print(f"[kaggle_runner] Poll exception: {e}", flush=True)

        time.sleep(poll_interval)

    print(f"[kaggle_runner] Timeout ({timeout}s) expired while polling '{kernel_slug}'!", flush=True)
    return "timeout", "\n".join(trace_lines)


def download_kernel_output(kernel_slug: str, output_dir: Path, timeout: int = 60) -> Tuple[bool, str]:
    """Downloads kernel execution output artifacts from Kaggle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = get_kaggle_cmd() + ["kernels", "output", kernel_slug, "-p", str(output_dir)]
    print(f"[kaggle_runner] Downloading output to {output_dir}...", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        combined = (proc.stdout + "\n" + proc.stderr).strip()
        print(f"[kaggle_runner] Download output:\n{combined}", flush=True)
        return proc.returncode == 0, combined
    except Exception as e:
        print(f"[kaggle_runner] Download failed: {e}", flush=True)
        return False, str(e)


def ping_kaggle(kernel_slug: str = DEFAULT_PING_SLUG, timeout: int = 300) -> bool:
    """Executes a lightweight verification ping kernel on Kaggle."""
    staging = SCRATCH / "kaggle_ping_staging"
    staging.mkdir(parents=True, exist_ok=True)

    script_path = staging / "ping_kernel.py"
    script_content = """import sys
import torch

print("=== ZALO AIC KAGGLE PING KERNEL ACTIVE ===", flush=True)
print(f"Python Version: {sys.version}", flush=True)
cuda_ok = torch.cuda.is_available()
gpu_count = torch.cuda.device_count() if cuda_ok else 0
gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "None"
print(f"CUDA Available: {cuda_ok}", flush=True)
print(f"GPU Count: {gpu_count}", flush=True)
print(f"Primary GPU: {gpu_name}", flush=True)

with open("ping_output.txt", "w", encoding="utf-8") as f:
    f.write(f"PONG_SUCCESS: GPU={gpu_name}, COUNT={gpu_count}\\n")
print("=== PING FINISHED ===", flush=True)
"""
    script_path.write_text(script_content, encoding="utf-8")
    create_kernel_metadata(
        dest_path=staging / "kernel-metadata.json",
        kernel_slug=kernel_slug,
        title="Zalo AIC Ping GPU",
        code_file="ping_kernel.py",
        enable_gpu=True,
        enable_internet=True,
    )

    pushed, push_log, resolved_slug = push_kernel(staging)
    if not pushed:
        print(f"[kaggle_runner] Ping push failed: {push_log}", flush=True)
        return False

    active_slug = resolved_slug or kernel_slug
    status, _ = poll_kernel_status(active_slug, poll_interval=10, timeout=timeout)
    if status != "complete":
        print(f"[kaggle_runner] Ping kernel did not complete successfully (status={status})", flush=True)
        return False

    out_dir = staging / "output"
    download_kernel_output(active_slug, out_dir)
    ping_out_file = out_dir / "ping_output.txt"
    if ping_out_file.exists() and "PONG_SUCCESS" in ping_out_file.read_text(encoding="utf-8"):
        print(
            f"[kaggle_runner] Ping test verified! Content: {ping_out_file.read_text(encoding='utf-8').strip()}",
            flush=True,
        )
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Batch Kernel Runner for ZALO_AIC")
    parser.add_argument("action", choices=["ping", "package"], default="ping")
    parser.add_argument("--slug", default=None, help="Kernel slug override")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds")
    args = parser.parse_args()

    if args.action == "ping":
        slug = args.slug or DEFAULT_PING_SLUG
        ok = ping_kaggle(slug, timeout=args.timeout)
        print("PING_OK" if ok else "PING_FAILED")
        sys.exit(0 if ok else 1)
    elif args.action == "package":
        out_p = SCRATCH / "code_package.zip"
        package_codebase(out_p)
        print(f"Packaged to {out_p}")
