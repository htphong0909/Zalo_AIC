#!/usr/bin/env python3
"""Colab orchestrator (via colab_runner) for Tier-1 Stage-2 On-Policy DPO V2 & Step Monitoring."""
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
DEFAULT_DPO_PREFIX = os.environ.get("DPO_HF_PREFIX", "exp_tier1_dpo_v2")
DEFAULT_SFT_REMOTE = os.environ.get("SFT_ADAPTER_REMOTE", "exp_tier1_sft_v3_clean/adapter")


def exec_local(cmd: list[str], *, check: bool = True, timeout: int = 240) -> subprocess.CompletedProcess:
    print(f"[colab_dpo_v2] {' '.join(cmd)}", flush=True)
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
        print(f"[colab_dpo_v2] `colab sessions` failed/timed out ({exc}); attempting reconnect...", flush=True)

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


def upload_artifacts(session: str, *, include_dpo_data: bool) -> None:
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
    out = run_remote_py(session, probe_code, timeout=45)
    missing = {r for _, r in static_files}
    for line in out.splitlines():
        if line.startswith("MISSING_JSON:"):
            missing = set(json.loads(line.split("MISSING_JSON:", 1)[1]))

    uploads: list[tuple[Path, str]] = [
        (pkg_zip, "/content/code_latest.zip"),
        (SCRATCH / "worker_dpo_v2.py", "/content/worker_dpo_v2.py"),
    ]
    if include_dpo_data:
        tr_p = ROOT / "data" / "dpo" / "minicpm5_dpo_v2_train.jsonl.gz"
        va_p = ROOT / "data" / "dpo" / "minicpm5_dpo_v2_val.jsonl.gz"
        assert tr_p.exists(), f"Missing {tr_p}"
        uploads.append((tr_p, "/content/minicpm5_dpo_v2_train.jsonl.gz"))
        if va_p.exists():
            uploads.append((va_p, "/content/minicpm5_dpo_v2_val.jsonl.gz"))
    for local_p, remote_p in static_files:
        if remote_p in missing:
            assert local_p.exists(), f"Missing {local_p}"
            uploads.append((local_p, remote_p))

    for local_p, remote_p in uploads:
        exec_local(["colab", "upload", "-s", session, str(local_p), remote_p])


def action_launch_train(args: argparse.Namespace, hf_repo: str, hf_token: str) -> None:
    syncer = HFSyncer(hf_repo, token=hf_token)
    worker = SCRATCH / "worker_dpo_v2.py"
    for chunk_idx in range(1, 7):
        if syncer.exists(f"{args.prefix}/adapter/adapter_model.safetensors"):
            print(f"DPO V2 adapter complete on HF at {args.prefix}/adapter!", flush=True)
            return

        gpu = ensure_live_session(args.session, args.gpu)
        restart_kernel(args.session)
        upload_artifacts(args.session, include_dpo_data=True)
        print(f"[colab_dpo_v2] Starting streamed DPO V2 training chunk {chunk_idx}/6...", flush=True)
        res = run_command(
            str(worker),
            session=args.session,
            is_file=True,
            timeout=3300,
            max_retries=2,
            env=[
                f"{ENV_TOKEN}={hf_token}",
                f"{ENV_REPO}={hf_repo}",
                f"DPO_HF_PREFIX={args.prefix}",
                f"SFT_ADAPTER_REMOTE={args.sft_remote}",
                "DPO_STAGE=train",
                f"DPO_EPOCHS={args.epochs}",
                f"DPO_LR={args.lr}",
                f"DPO_BETA={args.beta}",
                f"DPO_LABEL_SMOOTHING={args.label_smoothing}",
                f"DPO_SFT_WEIGHT={args.sft_weight}",
                "DPO_MAX_WALL_MIN=45",
            ],
            stream=True,
            gpu=gpu,
        )
        print(f"[colab_dpo_v2] Chunk {chunk_idx} exited rc={res.returncode}", flush=True)
        try:
            st_out = run_remote_py(
                args.session,
                "from pathlib import Path; p=Path('/content/dpo_v2_status.txt'); print('WORKER_STATUS:' + (p.read_text().strip() if p.exists() else 'UNKNOWN'))",
                timeout=30,
            )
            if "WORKER_STATUS:FAILED:" in st_out:
                raise RuntimeError(f"Chunk {chunk_idx} reported worker failure; aborting to save CU.")
        except RuntimeError:
            raise
        except Exception:
            pass
        if res.returncode == 0 and syncer.exists(f"{args.prefix}/adapter/adapter_model.safetensors"):
            return
        time.sleep(5)


def action_check(args: argparse.Namespace, hf_repo: str, hf_token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    files = sorted(f for f in api.list_repo_files(repo_id=hf_repo, repo_type="model") if f.startswith(f"{args.prefix}/"))
    ckpts = sorted({"/".join(f.split("/")[:3]) for f in files if f.startswith(f"{args.prefix}/checkpoints/")})
    print(f"[HF] prefix={args.prefix} ckpts={ckpts} files={len(files)}", flush=True)

    if not ping_session(args.session, timeout=15):
        print(f"[Colab] Session '{args.session}' inactive.", flush=True)
        return

    check_code = f"""
import json, subprocess
from pathlib import Path

pid_file = Path('/content/dpo_v2_train.pid')
pid = pid_file.read_text().strip() if pid_file.exists() else ''
alive = bool(pid and pid.isdigit() and Path(f'/proc/{{pid}}').exists())
gpu = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'],
    capture_output=True, text=True,
).stdout.strip()
print(f'REMOTE_PID={{pid or "none"}} ALIVE={{alive}} GPU=[{{gpu}}]')
sm = Path('/content/experiments/{args.prefix}/step_metrics.jsonl')
if sm.exists():
    rows = [json.loads(l) for l in sm.read_text(encoding='utf-8').splitlines() if l.strip()]
    print(f'STEP_METRICS_COUNT={{len(rows)}} (showing last 25):')
    for r in rows[-25:]:
        print(json.dumps(r, ensure_ascii=False))
log = Path('/content/dpo_v2_train.log')
if log.exists():
    lines = log.read_text(encoding='utf-8', errors='replace').splitlines()
    print('--- LOG TAIL ---')
    print('\\n'.join(lines[-25:]))
"""
    print(run_remote_py(args.session, check_code, timeout=60, gpu=args.gpu), flush=True)


def action_eval(args: argparse.Namespace, hf_repo: str, hf_token: str) -> None:
    gpu = ensure_live_session(args.session, args.gpu)
    upload_artifacts(args.session, include_dpo_data=False)

    worker = SCRATCH / "worker_dpo_v2.py"
    res = run_command(
        str(worker),
        session=args.session,
        is_file=True,
        timeout=3600,
        max_retries=2,
        env=[
            f"{ENV_TOKEN}={hf_token}",
            f"{ENV_REPO}={hf_repo}",
            f"DPO_HF_PREFIX={args.prefix}",
            f"SFT_ADAPTER_REMOTE={args.sft_remote}",
            "DPO_STAGE=eval",
            f"DPO_CKPT_SUBPATH={args.ckpt}",
            f"DPO_SOUP_ALPHA={args.alpha}",
        ],
        stream=True,
        gpu=gpu,
    )
    if res.returncode != 0:
        raise RuntimeError(f"DPO V2 eval worker failed ({res.returncode})")

    tag = f"{args.ckpt.replace('/', '_')}_a{float(args.alpha):.2f}"
    out_dir = ROOT / "experiments" / f"exp_20261113_minicpm5_dpo_v2_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    local_zip = out_dir / "dpo_v2_results.zip"
    exec_local(["colab", "download", "-s", args.session, "/content/dpo_v2_results.zip", str(local_zip)])
    if local_zip.exists():
        with zipfile.ZipFile(local_zip, "r") as z:
            z.extractall(str(out_dir))
        m = json.loads((out_dir / "eval" / "metrics.json").read_text(encoding="utf-8"))
        print(f"DPO V2 ({tag}) Dev Final Score: {m.get('final_score', 0):.4f} (n={m.get('n')})", flush=True)

    if not args.keep_session and os.environ.get("KEEP_SESSION") != "1":
        exec_local(["colab", "stop", "-s", args.session], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab orchestrator for Tier 1 DPO V2")
    parser.add_argument("--action", choices=["launch-train", "check", "eval", "stop"], required=True)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--prefix", default=DEFAULT_DPO_PREFIX)
    parser.add_argument("--sft-remote", default=DEFAULT_SFT_REMOTE)
    parser.add_argument("--epochs", default="1.0")
    parser.add_argument("--lr", default="1e-6")
    parser.add_argument("--beta", default="0.1")
    parser.add_argument("--label-smoothing", default="0.05")
    parser.add_argument("--sft-weight", default="0.40")
    parser.add_argument("--ckpt", default="adapter")
    parser.add_argument("--alpha", default="1.0")
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
