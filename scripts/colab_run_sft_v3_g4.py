#!/usr/bin/env python3
"""Staged / detached Colab G4/L4 orchestrator for Tier 1 Clean SFT V3 (MiniCPM5-2B).

Complies with AGENTS.md 4.3:
- HF preflight before GPU actions.
- Secrets passed strictly via `colab exec --env` (never written to files/logs).
- Training launched detached (`start_new_session=True`) so no blocking `colab exec`
  exceeds 60 minutes; monitored via `--action check` and auto-resumed from
  `{PREFIX}/checkpoints/latest` if interrupted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.colab_runner import package_codebase, ping_session, restart_kernel, run_command  # noqa: E402
from src.core.hf_sync import ENV_REPO, ENV_TOKEN, HFSyncError, HFSyncer, resolve_repo_id  # noqa: E402

DEFAULT_HF_REPO = "htphong0909/zalo-aic-sft-checkpoints"
SCRATCH = ROOT / "scripts" / "scratch"
DEFAULT_SESSION = os.environ.get("COLAB_SESSION", "minicpm5-tier1")
DEFAULT_GPU = os.environ.get("COLAB_GPU", "G4")
DEFAULT_PREFIX = os.environ.get("SFT_V3_HF_PREFIX", "exp_tier1_sft_v3_clean")


def exec_local(cmd: list[str], *, check: bool = True, timeout: int = 240) -> subprocess.CompletedProcess:
    print(f"[colab_sft_v3] {' '.join(cmd)}", flush=True)
    res = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if res.stdout:
        print(res.stdout, flush=True)
    if res.stderr:
        print(res.stderr, file=sys.stderr, flush=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}")
    return res


def hf_preflight() -> tuple[str, str]:
    from huggingface_hub import get_token

    repo = resolve_repo_id() or DEFAULT_HF_REPO
    token = os.environ.get(ENV_TOKEN) or get_token()
    if not repo or not token:
        sys.exit(f"ABORT: set {ENV_REPO} and {ENV_TOKEN} (or `hf auth login`).")
    last_exc: Exception | None = None
    for attempt in range(1, 5):
        try:
            user = HFSyncer(repo, token=token, retries=8, backoff_s=2.0).preflight()
            print(f"HF preflight OK as '{user}' -> {repo}", flush=True)
            return repo, token
        except HFSyncError as exc:
            last_exc = exc
            if "ConnectError" in str(exc) or "Timeout" in str(exc):
                print(f"[HF preflight retry {attempt}/4] transient network error: {exc}", flush=True)
                time.sleep(3 * attempt)
                continue
            sys.exit(f"ABORT (HF preflight): {exc}")
    sys.exit(f"ABORT (HF preflight): {last_exc}")


def ensure_live_session(session: str, gpu: str) -> str:
    if ping_session(session, timeout=20):
        return gpu

    sessions_out = ""
    try:
        sessions_out = exec_local(["colab", "sessions"], check=False, timeout=60).stdout
    except Exception as exc:
        print(f"[colab_sft_v3] `colab sessions` failed/timed out ({exc}); attempting reconnect...", flush=True)

    if session not in sessions_out:
        try:
            exec_local(["colab", "new", "-s", session, "--gpu", gpu], timeout=120)
        except Exception as exc:
            if gpu != "L4":
                print(f"{gpu} unavailable ({exc}); falling back to L4...", flush=True)
                gpu = "L4"
                exec_local(["colab", "new", "-s", session, "--gpu", "L4"], timeout=120)
            else:
                raise
        time.sleep(6)

    if not ping_session(session, timeout=25):
        restart_kernel(session)
        if not ping_session(session, timeout=25):
            raise RuntimeError(f"Session '{session}' unreachable.")
    return gpu


def run_remote_py(session: str, code: str, *, env: list[str] | None = None, timeout: int = 120, gpu: str = DEFAULT_GPU) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(code)
        tf_path = Path(tf.name)
    try:
        res = run_command(
            str(tf_path),
            session=session,
            is_file=True,
            timeout=timeout,
            max_retries=2,
            env=env,
            stream=True,
            gpu=gpu,
        )
        if res.returncode != 0:
            raise RuntimeError(f"colab_runner.run_command failed ({res.returncode}): {res.stderr or res.stdout}")
        return res.stdout or ""
    finally:
        tf_path.unlink(missing_ok=True)


def upload_artifacts(session: str, *, include_sft_data: bool) -> None:
    pkg_zip = SCRATCH / "code_latest.zip"
    package_codebase(pkg_zip)

    static_files = [
        (ROOT / "data" / "train" / "train_tools_catalog.json", "/content/train_tools_catalog.json"),
        (SCRATCH / "train.jsonl.gz", "/content/train.jsonl.gz"),
        (SCRATCH / "train_environment.zip", "/content/train_environment.zip"),
        (SCRATCH / "zaic_tools.zip", "/content/zaic_tools.zip"),
    ]
    probe_code = (
        "from pathlib import Path\n"
        f"paths = {[r for _, r in static_files]}\n"
        "missing = [p for p in paths if not Path(p).exists() or Path(p).stat().st_size == 0]\n"
        "print('MISSING_JSON:' + __import__('json').dumps(missing))\n"
    )
    out = run_remote_py(session, probe_code, timeout=30)
    missing: set[str] = {r for _, r in static_files}
    for line in out.splitlines():
        if line.startswith("MISSING_JSON:"):
            missing = set(json.loads(line.split("MISSING_JSON:", 1)[1]))

    uploads: list[tuple[Path, str]] = [
        (pkg_zip, "/content/code_latest.zip"),
        (SCRATCH / "worker_sft_v3_g4.py", "/content/worker_sft_v3_g4.py"),
    ]
    if include_sft_data:
        sft_data = ROOT / "data" / "sft" / "minicpm5_train_sft_v3.jsonl.gz"
        assert sft_data.exists(), f"Missing {sft_data}. Run scripts/build_minicpm5_dataset_v3.py first."
        uploads.append((sft_data, "/content/minicpm5_train_sft_v3.jsonl.gz"))
    for local_p, remote_p in static_files:
        if remote_p in missing:
            assert local_p.exists(), f"Missing local upload {local_p}"
            uploads.append((local_p, remote_p))
        else:
            print(f"[colab_sft_v3] Reusing existing remote {remote_p}", flush=True)

    for local_p, remote_p in uploads:
        exec_local(["colab", "upload", "-s", session, str(local_p), remote_p])


def action_launch_train(args: argparse.Namespace, hf_repo: str, hf_token: str) -> None:
    syncer = HFSyncer(hf_repo, token=hf_token)
    worker = SCRATCH / "worker_sft_v3_g4.py"
    max_chunks = 10
    for chunk_idx in range(1, max_chunks + 1):
        if syncer.exists(f"{args.prefix}/adapter/adapter_model.safetensors"):
            print(f"Adapter complete on HF at {args.prefix}/adapter!", flush=True)
            return

        gpu = ensure_live_session(args.session, args.gpu)
        restart_kernel(args.session)
        upload_artifacts(args.session, include_sft_data=True)
        print(f"[colab_sft_v3] Starting streamed training chunk {chunk_idx}/{max_chunks} (max_wall=45m)...", flush=True)
        res = run_command(
            str(worker),
            session=args.session,
            is_file=True,
            timeout=3300,
            max_retries=2,
            env=[
                f"{ENV_TOKEN}={hf_token}",
                f"{ENV_REPO}={hf_repo}",
                f"SFT_V3_HF_PREFIX={args.prefix}",
                "SFT_V3_STAGE=train",
                f"SFT_V3_EPOCHS={args.epochs}",
                f"SFT_V3_LR={args.lr}",
                "SFT_V3_MAX_WALL_MIN=45",
            ],
            stream=True,
            gpu=gpu,
        )
        print(f"[colab_sft_v3] Chunk {chunk_idx} exited rc={res.returncode}", flush=True)
        action_check(args, hf_repo, hf_token)
        if res.returncode != 0:
            print("[colab_sft_v3] Transient failure/disconnect; auto-resuming from HF checkpoints/latest...", flush=True)
            time.sleep(5)

    if not syncer.exists(f"{args.prefix}/adapter/adapter_model.safetensors"):
        raise RuntimeError(f"SFT V3 did not finish within {max_chunks} chunks.")


def action_check(args: argparse.Namespace, hf_repo: str, hf_token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    files = sorted(f for f in api.list_repo_files(repo_id=hf_repo, repo_type="model") if f.startswith(f"{args.prefix}/"))
    ckpts = sorted({"/".join(f.split("/")[:3]) for f in files if f.startswith(f"{args.prefix}/checkpoints/")})
    has_adapter = f"{args.prefix}/adapter/adapter_model.safetensors" in files
    has_summary = f"{args.prefix}/training_summary.json" in files
    has_metrics = f"{args.prefix}/eval/metrics.json" in files
    print(
        f"[HF] prefix={args.prefix} total_files={len(files)} ckpts={ckpts} "
        f"adapter={has_adapter} summary={has_summary} metrics={has_metrics}",
        flush=True,
    )

    if not ping_session(args.session, timeout=15):
        print(f"[Colab] Session '{args.session}' is not active.", flush=True)
        return

    check_code = """
import subprocess
from pathlib import Path

pid_file = Path('/content/sft_v3_train.pid')
pid = pid_file.read_text().strip() if pid_file.exists() else ''
alive = bool(pid and pid.isdigit() and Path(f'/proc/{pid}').exists())
gpu = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'],
    capture_output=True, text=True,
).stdout.strip()
print(f'REMOTE_PID={pid or "none"} ALIVE={alive} GPU=[{gpu}]')
log = Path('/content/sft_v3_train.log')
if log.exists():
    lines = log.read_text(encoding='utf-8', errors='replace').splitlines()
    print(f'LOG_LINES={len(lines)}')
    print('\\n'.join(lines[-35:]))
else:
    print('NO_LOG_YET')
"""
    print(run_remote_py(args.session, check_code, timeout=30), flush=True)


def action_eval(args: argparse.Namespace, hf_repo: str, hf_token: str) -> None:
    gpu = ensure_live_session(args.session, args.gpu)
    upload_artifacts(args.session, include_sft_data=False)

    ckpt_subpath = args.ckpt_subpath.strip("/")
    ckpt_tag = "" if ckpt_subpath == "adapter" else Path(ckpt_subpath).name
    worker = SCRATCH / "worker_sft_v3_g4.py"
    res = run_command(
        str(worker),
        session=args.session,
        is_file=True,
        timeout=2400,
        max_retries=2,
        env=[
            f"{ENV_TOKEN}={hf_token}",
            f"{ENV_REPO}={hf_repo}",
            f"SFT_V3_HF_PREFIX={args.prefix}",
            f"SFT_V3_CKPT_SUBPATH={ckpt_subpath}",
            "SFT_V3_STAGE=eval",
        ],
        stream=True,
        gpu=gpu,
    )
    if res.returncode != 0:
        raise RuntimeError(f"SFT V3 eval worker failed ({res.returncode})")

    dir_suffix = f"_{ckpt_tag}" if ckpt_tag else ""
    out_dir = ROOT / "experiments" / f"exp_20261113_minicpm5_sft_v3_g4{dir_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    local_zip = out_dir / "sft_v3_results.zip"
    exec_local(["colab", "download", "-s", args.session, "/content/sft_v3_results.zip", str(local_zip)])
    if local_zip.exists():
        with zipfile.ZipFile(local_zip, "r") as z:
            z.extractall(str(out_dir))
        print(f"Extracted SFT V3 results -> {out_dir}", flush=True)
        m = json.loads((out_dir / "eval" / "metrics.json").read_text(encoding="utf-8"))
        print(f"SFT V3 Dev Final Score ({ckpt_subpath}): {m.get('final_score', 0):.4f} (n={m.get('n')})", flush=True)

    action_check(args, hf_repo, hf_token)
    if not args.keep_session and os.environ.get("KEEP_SESSION") != "1":
        exec_local(["colab", "stop", "-s", args.session], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged Colab G4/L4 orchestrator for Tier 1 SFT V3")
    parser.add_argument("--action", choices=["launch-train", "check", "eval", "stop"], required=True)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--ckpt-subpath", default="adapter", help="Subpath under prefix to evaluate (e.g. adapter or checkpoints/checkpoint-160)")
    parser.add_argument("--epochs", default="2.0")
    parser.add_argument("--lr", default="1.5e-4")
    parser.add_argument("--keep-session", action="store_true")
    args = parser.parse_args()

    if args.action == "stop":
        exec_local(["colab", "stop", "-s", args.session], check=False)
        return

    hf_repo, hf_token = hf_preflight()
    if args.action == "launch-train":
        action_launch_train(args, hf_repo, hf_token)
    elif args.action == "check":
        action_check(args, hf_repo, hf_token)
    elif args.action == "eval":
        action_eval(args, hf_repo, hf_token)


if __name__ == "__main__":
    main()
