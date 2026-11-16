#!/usr/bin/env python3
"""Orchestrate LoopTool GCP Capability Probe of SFT V2 on 1,246 train_only samples via Colab G4/L4."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.colab_runner import package_codebase, ping_session, restart_kernel, run_command, run_remote_py  # noqa: E402
from src.core.hf_sync import ENV_REPO, ENV_TOKEN, HFSyncError, HFSyncer, resolve_repo_id  # noqa: E402

DEFAULT_HF_REPO = "htphong0909/zalo-aic-sft-checkpoints"
SCRATCH = ROOT / "scripts" / "scratch"
SESSION = os.environ.get("COLAB_SESSION", "minicpm5-probe")
GPU = os.environ.get("COLAB_GPU", "G4")
PREFIX = os.environ.get("PROBE_HF_PREFIX", "looptool_probe_sft_v3")
ADAPTER_REMOTE = os.environ.get("PROBE_ADAPTER_REMOTE", "exp_tier1_sft_v3_clean/adapter")


def exec_local(cmd: list[str], *, check: bool = True, timeout: int = 240) -> subprocess.CompletedProcess:
    print(f"[colab_probe] {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
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
    try:
        user = HFSyncer(repo, token=token, retries=5).preflight()
    except HFSyncError as exc:
        sys.exit(f"ABORT (HF preflight): {exc}")
    print(f"HF preflight OK as '{user}' -> {repo}", flush=True)
    return repo, token


def main() -> None:
    global GPU
    parser = argparse.ArgumentParser(description="Orchestrate LoopTool GCP Capability Probe on 1,246 train_only samples.")
    parser.add_argument("--session", default=SESSION)
    parser.add_argument("--gpu", default=GPU)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--adapter", default=ADAPTER_REMOTE)
    parser.add_argument("--max-wall-min", default="42")
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--keep-session", action="store_true")
    args = parser.parse_args()

    session = args.session
    GPU = args.gpu
    prefix = args.prefix
    adapter_remote = args.adapter

    hf_repo, hf_token = hf_preflight()
    syncer = HFSyncer(hf_repo, token=hf_token, retries=5)
    pkg_zip = SCRATCH / "code_latest.zip"
    package_codebase(pkg_zip)

    static_uploads = [
        (ROOT / "data" / "train" / "train_tools_catalog.json", "/content/train_tools_catalog.json"),
        (SCRATCH / "train.jsonl.gz", "/content/train.jsonl.gz"),
        (SCRATCH / "train_environment.zip", "/content/train_environment.zip"),
        (SCRATCH / "zaic_tools.zip", "/content/zaic_tools.zip"),
    ]

    for chunk in range(1, args.max_chunks + 1):
        if syncer.exists(f"{prefix}/eval/metrics.json"):
            print(f"Probe already complete on HF ({prefix}/eval/metrics.json)!", flush=True)
            break

        sessions_out = exec_local(["colab", "sessions"], check=False).stdout
        if session not in sessions_out:
            try:
                exec_local(["colab", "new", "-s", session, "--gpu", GPU])
            except Exception as exc:
                if GPU != "L4":
                    print(f"G4 unavailable ({exc}); falling back to L4...", flush=True)
                    GPU = "L4"
                    exec_local(["colab", "new", "-s", session, "--gpu", "L4"])
                else:
                    raise
            time.sleep(6)

        if not ping_session(session, timeout=20):
            restart_kernel(session)
            if not ping_session(session, timeout=20):
                raise RuntimeError(f"Session '{session}' unreachable.")

        probe_code = (
            "from pathlib import Path\n"
            f"paths = {[r for _, r in static_uploads]}\n"
            "missing = [p for p in paths if not Path(p).exists() or Path(p).stat().st_size == 0]\n"
            "print('MISSING_JSON:' + __import__('json').dumps(missing))\n"
        )
        out = run_remote_py(session, probe_code, timeout=45, gpu=GPU)
        missing = {r for _, r in static_uploads}
        for line in out.splitlines():
            if line.startswith("MISSING_JSON:"):
                missing = set(json.loads(line.split("MISSING_JSON:", 1)[1]))

        uploads: list[tuple[Path, str]] = [(pkg_zip, "/content/code_latest.zip")]
        for local_p, remote_p in static_uploads:
            if remote_p in missing:
                uploads.append((local_p, remote_p))
        for local_p, remote_p in uploads:
            assert local_p.exists(), f"Missing {local_p}"
            exec_local(["colab", "upload", "-s", session, str(local_p), remote_p])

        worker = SCRATCH / "worker_looptool_probe.py"
        print(f"[colab_probe] Launching probe chunk {chunk}/{args.max_chunks} (prefix={prefix})...", flush=True)
        res = run_command(
            str(worker),
            session=session,
            is_file=True,
            timeout=3300,
            max_retries=2,
            env=[
                f"{ENV_TOKEN}={hf_token}",
                f"{ENV_REPO}={hf_repo}",
                f"PROBE_HF_PREFIX={prefix}",
                f"PROBE_ADAPTER_REMOTE={adapter_remote}",
                f"PROBE_MAX_WALL_MIN={args.max_wall_min}",
            ],
            stream=True,
            gpu=GPU,
        )
        print(f"[colab_probe] Chunk {chunk} exited rc={res.returncode}", flush=True)
        if res.returncode != 0:
            raise RuntimeError(f"Probe worker failed ({res.returncode})")
        if syncer.exists(f"{prefix}/eval/metrics.json"):
            break
        restart_kernel(session)

    out_dir = ROOT / "experiments" / f"exp_20261113_{prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    syncer.download_dir(f"{prefix}/eval", out_dir.parent)
    syncer.download_dir(f"{prefix}/preds", out_dir.parent)
    hf_sub = out_dir.parent / prefix
    if hf_sub.exists() and hf_sub != out_dir:
        for sub in ("eval", "preds"):
            src_sub = hf_sub / sub
            dst_sub = out_dir / sub
            if src_sub.exists():
                dst_sub.mkdir(parents=True, exist_ok=True)
                for f in src_sub.glob("*"):
                    import shutil
                    shutil.copy2(f, dst_sub / f.name)

    sessions_out = exec_local(["colab", "sessions"], check=False).stdout
    if session in sessions_out:
        local_zip = out_dir / "looptool_probe_results.zip"
        exec_local(["colab", "download", "-s", session, "/content/looptool_probe_results.zip", str(local_zip)], check=False)
        if local_zip.exists():
            with zipfile.ZipFile(local_zip, "r") as z:
                z.extractall(str(out_dir))
        if not args.keep_session and os.environ.get("KEEP_SESSION") != "1":
            exec_local(["colab", "stop", "-s", session], check=False)

    if (out_dir / "eval" / "metrics.json").exists():
        print(f"Extracted probe results -> {out_dir}", flush=True)
        m = json.loads((out_dir / "eval" / "metrics.json").read_text(encoding="utf-8"))
        print(f"Train-Only Final Score: {m.get('final_score', 0):.4f} (n={m.get('n')})", flush=True)
    else:
        raise RuntimeError(f"Probe did not finish within {args.max_chunks} chunks; partial preds saved to HF {prefix}/preds.")


if __name__ == "__main__":
    main()
