#!/usr/bin/env python3
"""Self-healing Colab G4/L4 orchestrator, HF asset stager, and Orca watchdog for Qwen3.5-4B SFT V2.

Follows AGENTS.md section 4.3 & checkpoint-sync-hf:
- Mandatory HF preflight before GPU actions.
- Secrets (`HF_TOKEN`) passed strictly in memory via `colab exec --env` (never written to disk/logs).
- Staging all code + V2 dataset + eval sandboxes onto HF `<prefix>/assets` so any fresh Colab VM can
  auto-recover after runtime preemption without depending on Jupyter upload availability.
- Training and vLLM evaluation run detached (`start_new_session=True`), supervised via short idempotent
  `check` / `ensure` / `precheck` actions and periodic Orca `antigravity` automation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.colab_runner import (  # noqa: E402
    package_codebase,
    ping_session,
    restart_kernel,
    run_command,
    run_remote_py,
)
from src.core.hf_sync import ENV_REPO, ENV_TOKEN, HFSyncError, HFSyncer, resolve_repo_id  # noqa: E402

DEFAULT_HF_REPO = "htphong0909/zalo-aic-sft-checkpoints"
DEFAULT_SESSION = os.environ.get("COLAB_SESSION", "qwen38-work")
DEFAULT_GPU = os.environ.get("COLAB_GPU", "G4")
DEFAULT_PREFIX = os.environ.get("QWEN35_SFT_HF_PREFIX", "exp_qwen35_sft_v2")
DEFAULT_SFT_CONFIG = "configs/qwen35_sft_v2.yaml"
DEFAULT_EVAL_CONFIG = "configs/oracle_qwen35_4b_sft_v2.yaml"
DEFAULT_ORCA_NAME = "Qwen3.5-4B SFT V2 Colab Watchdog"
DEFAULT_DATA_FILE = "data/sft/qwen35_train_sft_v2.jsonl.gz"
STALL_THRESHOLD_S = 1800.0  # 30 minutes of zero progress with 0% GPU triggers auto-restart
MAX_AUTO_RECOVERIES_BEFORE_WAKE = 3
FATAL_LOG_MARKERS = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "loss': nan",
    "loss: nan",
    "HEALTH_ANOMALY",
    "NAN_LOSS",
    "GRAD_EXPLOSION",
    "LOSS_DIVERGENCE",
    "WebSocketClosedError",
    "Connection was lost",
    "503 Service Unavailable",
    "RuntimeError: Expected",
    "ImportError:",
    "ModuleNotFoundError:",
    "SyntaxError:",
    "Traceback (most recent call last):",
)
EVAL_SPLITS: tuple[tuple[str, str, str], ...] = (
    ("fold0", "configs/splits/fold_0_ids.json", "fold0_preds.jsonl"),
    ("dev299", "configs/splits/dev_v1.json", "dev_preds.jsonl"),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _asset_specs(
    root: Path = ROOT,
    *,
    data_file: str | Path | None = None,
    sft_config: str | Path | None = None,
    eval_config: str | Path | None = None,
    eval_splits: list[str] | None = None,
    transcripts_file: str | Path | None = None,
) -> list[tuple[Path, str]]:
    scratch = root / "scripts" / "scratch"
    fold0 = root / "configs" / "splits" / "fold_0_ids.json"
    if not fold0.exists():
        fold0 = scratch / "fold_0_ids.json"

    data_p = Path(data_file) if data_file else (root / DEFAULT_DATA_FILE)
    if not data_p.is_absolute():
        data_p = root / data_p

    target_splits = set(eval_splits) if eval_splits is not None else {"fold0", "dev299"}
    specs: list[tuple[Path, str]] = [
        (scratch / "code_latest.zip", "code_latest.zip"),
        (data_p, data_p.name),
        (root / "data" / "train" / "train_tools_catalog.json", "train_tools_catalog.json"),
        (scratch / "train.jsonl.gz", "train.jsonl.gz"),
        (scratch / "train_environment.zip", "train_environment.zip"),
        (scratch / "zaic_tools.zip", "zaic_tools.zip"),
    ]
    if "dev299" in target_splits:
        specs.append((root / "configs" / "splits" / "dev_v1.json", "dev_v1.json"))
    if "fold0" in target_splits:
        specs.append((fold0, "fold_0_ids.json"))
    for cfg in (sft_config, eval_config):
        if cfg:
            cp = Path(cfg) if Path(cfg).is_absolute() else (root / cfg)
            specs.append((cp, cp.name))
    if transcripts_file:
        tp = Path(transcripts_file) if Path(transcripts_file).is_absolute() else (root / transcripts_file)
        if tp.exists():
            specs.append((tp, tp.name))
    return specs


def _exp_dir(prefix: str, root: Path = ROOT) -> Path:
    d = root / "experiments" / prefix
    d.mkdir(parents=True, exist_ok=True)
    return d


def _append_log(prefix: str, msg: str, root: Path = ROOT) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_file = _exp_dir(prefix, root) / "supervisor.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_supervisor_state(prefix: str, root: Path = ROOT) -> dict[str, Any]:
    path = _exp_dir(prefix, root) / "supervisor_state.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_supervisor_state(prefix: str, state: dict[str, Any], root: Path = ROOT) -> None:
    path = _exp_dir(prefix, root) / "supervisor_state.json"
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def hf_preflight(hf_repo: str | None = None) -> tuple[str, str, HFSyncer]:
    from huggingface_hub import get_token

    repo = resolve_repo_id(hf_repo) or DEFAULT_HF_REPO
    token = os.environ.get(ENV_TOKEN) or get_token()
    if not repo or not token:
        raise HFSyncError(f"Set {ENV_REPO} and {ENV_TOKEN} (or run `hf auth login`).")
    last_exc: Exception | None = None
    for attempt in range(1, 5):
        try:
            syncer = HFSyncer(repo, token=token, retries=6, backoff_s=2.0)
            syncer.preflight()
            return repo, token, syncer
        except HFSyncError as exc:
            last_exc = exc
            if any(k in str(exc) for k in ("ConnectError", "Timeout", "ReadTimeout")):
                time.sleep(2 * attempt)
                continue
            raise
    raise HFSyncError(f"HF preflight failed: {last_exc}")


def _download_file(syncer: HFSyncer, path_in_repo: str, local_dir: Path) -> Path | None:
    """Download a single file from the Hub repo if it exists."""
    if not syncer.exists(path_in_repo):
        return None
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = syncer._call_with_retry(
        lambda: hf_hub_download(
            repo_id=syncer.repo_id,
            repo_type=syncer.repo_type,
            filename=path_in_repo,
            local_dir=str(local_dir),
            token=syncer.token,
        )
    )
    return Path(downloaded)


def sync_assets(
    syncer: HFSyncer,
    prefix: str,
    *,
    force: bool = False,
    root: Path = ROOT,
    data_file: str | Path | None = None,
    sft_config: str | Path | None = None,
    eval_config: str | Path | None = None,
    eval_splits: list[str] | None = None,
    transcripts_file: str | Path | None = None,
) -> dict[str, Any]:
    """Package codebase + build SHA256 manifest + push missing/changed assets to HF `<prefix>/assets`."""
    pkg_zip = root / "scripts" / "scratch" / "code_latest.zip"
    pkg_zip.parent.mkdir(parents=True, exist_ok=True)
    package_codebase(pkg_zip)

    specs = _asset_specs(
        root,
        data_file=data_file,
        sft_config=sft_config,
        eval_config=eval_config,
        eval_splits=eval_splits,
        transcripts_file=transcripts_file,
    )
    manifest: dict[str, dict[str, Any]] = {}
    for local_p, name in specs:
        if not local_p.exists() or local_p.stat().st_size == 0:
            raise FileNotFoundError(f"Required asset missing: {local_p}")
        manifest[name] = {"sha256": _sha256(local_p), "size": local_p.stat().st_size}

    cache_dir = _exp_dir(prefix, root) / "hf_cache"
    remote_manifest: dict[str, Any] | None = None
    if not force:
        try:
            m_path = _download_file(syncer, f"{prefix}/assets/manifest.json", cache_dir)
            if m_path is not None and m_path.exists():
                remote_manifest = json.loads(m_path.read_text(encoding="utf-8"))
        except Exception:
            remote_manifest = None

    if not force and remote_manifest == manifest:
        _append_log(prefix, f"HF assets already up-to-date ({len(manifest)} files).", root)
        return manifest

    with tempfile.TemporaryDirectory(prefix="sft_assets_") as td:
        stage = Path(td)
        for local_p, name in specs:
            shutil.copy2(local_p, stage / name)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        syncer.push(stage, f"{prefix}/assets", f"sync SFT assets ({len(manifest)} files)", raise_on_fail=True)
    _append_log(prefix, f"Synced {len(manifest)} assets to HF {syncer.repo_id}/{prefix}/assets.", root)
    return manifest


def inspect_hf_state(
    syncer: HFSyncer,
    prefix: str,
    root: Path = ROOT,
    *,
    eval_splits: list[str] | None = None,
) -> dict[str, Any]:
    """Query HF repo files and latest progress metadata for `<prefix>`."""
    files = sorted(
        f for f in syncer._call_with_retry(lambda: syncer.api.list_repo_files(repo_id=syncer.repo_id, repo_type="model"))
        if f.startswith(f"{prefix}/")
    )
    ckpt_steps = sorted(
        {
            int(part.split("checkpoint-", 1)[1])
            for f in files
            if f.startswith(f"{prefix}/checkpoints/checkpoint-")
            for part in [f.split("/")[2]]
            if part.startswith("checkpoint-") and part.split("checkpoint-", 1)[1].isdigit()
        }
    )
    has_adapter = f"{prefix}/adapter/adapter_model.safetensors" in files
    has_merged = any(f.startswith(f"{prefix}/merged/model") and f.endswith(".safetensors") for f in files)
    has_summary = f"{prefix}/training_summary.json" in files
    has_fold0 = f"{prefix}/eval_fold0/metrics.json" in files
    has_dev299 = f"{prefix}/eval_dev299/metrics.json" in files
    target_splits = eval_splits or ["fold0", "dev299"]
    eval_done = all((has_fold0 if s == "fold0" else has_dev299) for s in target_splits)
    latest_progress: dict[str, Any] = {}
    if f"{prefix}/checkpoints/latest/progress.json" in files:
        cache_dir = _exp_dir(prefix, root) / "hf_cache"
        prog_file = _download_file(syncer, f"{prefix}/checkpoints/latest/progress.json", cache_dir)
        if prog_file is not None and prog_file.exists():
            try:
                latest_progress = json.loads(prog_file.read_text(encoding="utf-8"))
            except Exception:
                latest_progress = {}

    return {
        "total_files": len(files),
        "ckpt_steps": ckpt_steps,
        "latest_step": latest_progress.get("global_step") or (ckpt_steps[-1] if ckpt_steps else 0),
        "latest_max_steps": latest_progress.get("max_steps") or 0,
        "latest_epoch": latest_progress.get("epoch"),
        "latest_loss": latest_progress.get("last_loss"),
        "latest_grad_norm": latest_progress.get("last_grad_norm"),
        "latest_lr": latest_progress.get("last_lr"),
        "latest_saved_at": latest_progress.get("saved_at"),
        "has_adapter": has_adapter,
        "has_merged": has_merged,
        "has_summary": has_summary,
        "has_fold0_metrics": has_fold0,
        "has_dev299_metrics": has_dev299,
        "train_done": bool(has_adapter and has_summary),
        "all_done": bool(has_adapter and has_summary and eval_done),
    }


def ensure_live_session(session: str, gpu: str) -> str:
    """Guarantee a responsive Colab session, creating G4 (with L4 fallback only on creation failure)."""
    if ping_session(session, timeout=20):
        return gpu

    sessions_out = ""
    try:
        sessions_out = subprocess.run(
            ["colab", "sessions"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45
        ).stdout
    except Exception:
        sessions_out = ""

    if session not in sessions_out:
        try:
            subprocess.run(["colab", "new", "-s", session, "--gpu", gpu], check=True, timeout=120)
        except Exception:
            if gpu != "L4":
                gpu = "L4"
                subprocess.run(["colab", "new", "-s", session, "--gpu", "L4"], check=True, timeout=120)
            else:
                raise
        time.sleep(6)

    if not ping_session(session, timeout=25):
        restart_kernel(session)
        if not ping_session(session, timeout=25):
            raise RuntimeError(f"Colab session '{session}' remained unreachable after kernel restart.")
    return gpu


def probe_colab_state(
    session: str,
    prefix: str,
    gpu: str,
    *,
    data_name: str | None = None,
    eval_splits: list[str] | None = None,
) -> dict[str, Any]:
    """Non-destructive snapshot of Colab processes, GPU metrics, live_metrics.json, disk space, and log tails."""
    if not ping_session(session, timeout=20):
        return {"colab_alive": False}

    target_data = data_name or Path(DEFAULT_DATA_FILE).name
    splits_list = eval_splits or ["fold0", "dev299"]
    probe_code = f"""
import json, shutil, subprocess
from pathlib import Path

prefix = {prefix!r}
target_data = {target_data!r}
target_splits = {splits_list!r}

def _proc_alive(pid_str):
    st_p = Path(f'/proc/{{pid_str}}/status')
    if not st_p.exists():
        return False
    try:
        txt = st_p.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return False
    for ln in txt.splitlines():
        if ln.startswith('State:'):
            code = ln.split(':', 1)[1].strip()[:1].upper()
            return code not in ('Z', 'X', '')
    return False

def _pid_alive(pid_file, pattern):
    p = Path(pid_file)
    pid = p.read_text(encoding='utf-8').strip() if p.exists() else ''
    if pid.isdigit() and _proc_alive(pid):
        return pid, True
    res = subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True)
    found = [x.strip() for x in res.stdout.strip().splitlines() if x.strip().isdigit() and _proc_alive(x.strip())]
    return (found[0] if found else ''), bool(found)

train_pid, train_alive = _pid_alive('/content/sft_v2_train.pid', 'train_qwen38_sft.py')
eval_pid, eval_alive = _pid_alive('/content/sft_v2_eval.pid', 'colab_qwen35_sft.*worker-eval|scripts/run_agent.py')

gpu_name, gpu_util, vram_used, vram_total = '', 0, 0, 0
try:
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    if out:
        parts = [x.strip() for x in out.splitlines()[0].split(',')]
        gpu_name = parts[0]
        gpu_util = int(float(parts[1]))
        vram_used = int(float(parts[2]))
        vram_total = int(float(parts[3]))
except Exception:
    pass

try:
    disk_free_gb = round(shutil.disk_usage('/content').free / (1024 ** 3), 2)
except Exception:
    disk_free_gb = None

live_metrics = {{}}
lm_path = Path(f'/content/experiments/{{prefix}}/live_metrics.json')
if lm_path.exists():
    try:
        live_metrics = json.loads(lm_path.read_text(encoding='utf-8'))
    except Exception:
        live_metrics = {{}}

def _tail(path_str, n=80):
    p = Path(path_str)
    if not p.exists():
        return 0, []
    lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
    return len(lines), lines[-n:]

t_cnt, t_tail = _tail('/content/sft_v2_train.log')
e_cnt, e_tail = _tail('/content/sft_v2_eval.log')

stamp = {{}}
sp = Path('/content/.assets_stamp.json')
if sp.exists():
    try:
        stamp = json.loads(sp.read_text(encoding='utf-8'))
    except Exception:
        stamp = {{}}

def _line_count(path_str):
    p = Path(path_str)
    if not p.exists():
        return 0
    return sum(1 for ln in p.read_text(encoding='utf-8', errors='replace').splitlines() if ln.strip())

required_paths = [
    '/content/scripts/train_qwen38_sft.py',
    f'/content/{{target_data}}',
    '/content/data/train/train.jsonl',
    '/content/data/train/zaic_tools/executor.py',
    '/content/data/train/train_environment/train_0001',
]
if 'fold0' in target_splits:
    required_paths.append('/content/configs/splits/fold_0_ids.json')
if 'dev299' in target_splits:
    required_paths.append('/content/configs/splits/dev_v1.json')
assets_ok = all(Path(p).exists() for p in required_paths)

payload = {{
    'colab_alive': True,
    'train_pid': train_pid,
    'train_alive': train_alive,
    'eval_pid': eval_pid,
    'eval_alive': eval_alive,
    'gpu_name': gpu_name,
    'gpu_util': gpu_util,
    'vram_used_mb': vram_used,
    'vram_total_mb': vram_total,
    'disk_free_gb': disk_free_gb,
    'live_metrics': live_metrics,
    'assets_ok': assets_ok,
    'assets_stamp': stamp,
    'has_merged': (
        Path(f'/content/experiments/{{prefix}}/merged/.complete').exists()
        and Path(f'/content/experiments/{{prefix}}/merged/config.json').exists()
        and Path(f'/content/experiments/{{prefix}}/merged/tokenizer.json').exists()
    ),
    'train_log_lines': t_cnt,
    'train_log_tail': t_tail,
    'eval_log_lines': e_cnt,
    'eval_log_tail': e_tail,
    'fold0_pred_count': _line_count(f'/content/experiments/{{prefix}}/preds_fold0/fold0_preds.jsonl'),
    'dev299_pred_count': _line_count(f'/content/experiments/{{prefix}}/preds_dev299/dev_preds.jsonl'),
}}
print('PROBE_JSON:' + json.dumps(payload))
"""
    try:
        raw = run_remote_py(session, probe_code, timeout=45, gpu=gpu)
    except Exception as exc:
        return {"colab_alive": False, "probe_error": str(exc)[-500:]}
    for line in raw.splitlines():
        if line.startswith("PROBE_JSON:"):
            return json.loads(line.split("PROBE_JSON:", 1)[1])
    return {"colab_alive": True, "probe_error": raw[-500:]}


def is_proc_status_alive(status_text: str) -> bool:
    """Return False for Linux zombie ('Z'), dead ('X'), or unreadable /proc/<pid>/status text."""
    for ln in status_text.splitlines():
        if ln.startswith("State:"):
            code = ln.split(":", 1)[1].strip()[:1].upper()
            return code not in ("Z", "X", "")
    return False


def detect_log_anomaly(lines: list[str]) -> str | None:
    last_launch = -1
    for i, ln in enumerate(lines):
        if "=== LAUNCH " in ln or "=== EVAL LAUNCH " in ln:
            last_launch = i
    recent = lines[last_launch + 1 :] if last_launch >= 0 else lines
    text = "\n".join(recent[-80:])
    for marker in FATAL_LOG_MARKERS:
        if marker.lower() in text.lower():
            return marker
    return None


def assess_status(
    hf_state: dict[str, Any],
    remote_state: dict[str, Any],
    prev_state: dict[str, Any],
    *,
    now_ts: float | None = None,
    finalized_locally: bool = False,
    max_sync_age_s: float = 900.0,
) -> dict[str, Any]:
    """Pure state assessment deciding health, recovery needs, and agent wake signal."""
    now = time.time() if now_ts is None else now_ts
    prev_step = int(prev_state.get("last_observed_step") or 0)
    prev_log_lines = int(prev_state.get("last_train_log_lines") or 0)
    prev_eval_lines = int(prev_state.get("last_eval_log_lines") or 0)
    prev_preds = int(prev_state.get("last_pred_total") or 0)

    live_metrics = remote_state.get("live_metrics") or {}
    live_step = int(live_metrics.get("step") or 0) if isinstance(live_metrics, dict) else 0
    cur_step = int(hf_state.get("latest_step") or 0)
    cur_log_lines = int(remote_state.get("train_log_lines") or 0)
    cur_eval_lines = int(remote_state.get("eval_log_lines") or 0)
    cur_preds = int(remote_state.get("fold0_pred_count") or 0) + int(remote_state.get("dev299_pred_count") or 0)

    progressed = (
        cur_step > prev_step
        or live_step > int(prev_state.get("last_live_step") or 0)
        or cur_log_lines > prev_log_lines
        or cur_eval_lines > prev_eval_lines
        or cur_preds > prev_preds
    )
    last_progress_ts = now if progressed or not prev_state.get("last_progress_ts") else float(prev_state["last_progress_ts"])
    last_hf_progress_ts = (
        now
        if (cur_step > prev_step or not prev_state.get("last_hf_progress_ts"))
        else float(prev_state["last_hf_progress_ts"])
    )
    stall_s = max(0.0, now - last_progress_ts)
    hf_sync_age_s = max(0.0, now - last_hf_progress_ts)

    live_anomaly = str(live_metrics.get("anomaly") or "").strip() if isinstance(live_metrics, dict) else ""
    log_anomaly = detect_log_anomaly(remote_state.get("train_log_tail") or []) or detect_log_anomaly(
        remote_state.get("eval_log_tail") or []
    )
    anomaly = live_anomaly or log_anomaly
    recovery_count = 0 if cur_step > prev_step else int(prev_state.get("recovery_count") or 0)

    disk_free = remote_state.get("disk_free_gb")
    sync_stale = bool(
        remote_state.get("train_alive")
        and live_step > cur_step + 40
        and hf_sync_age_s >= max_sync_age_s
    )

    base: dict[str, Any] = {
        "checked_at": now,
        "last_observed_step": max(cur_step, prev_step),
        "last_live_step": max(live_step, int(prev_state.get("last_live_step") or 0)),
        "last_train_log_lines": cur_log_lines,
        "last_eval_log_lines": cur_eval_lines,
        "last_pred_total": max(cur_preds, prev_preds),
        "last_progress_ts": last_progress_ts,
        "last_hf_progress_ts": last_hf_progress_ts,
        "stall_s": round(stall_s, 1),
        "hf_sync_age_s": round(hf_sync_age_s, 1),
        "sync_stale": sync_stale,
        "recovery_count": recovery_count,
        "anomaly": anomaly,
        "live_metrics": live_metrics if isinstance(live_metrics, dict) else {},
        "disk_free_gb": disk_free,
    }

    if hf_state.get("all_done"):
        status = "COMPLETE"
        wake = not finalized_locally
        reason = "finalization_needed" if wake else "already_finalized"
        return {**base, "status": status, "next_action": "download_results" if wake else "none", "wake_agent": wake, "reason": reason}

    if not remote_state.get("colab_alive"):
        return {
            **base,
            "status": "NEEDS_RECOVERY",
            "next_action": "recover_and_resume",
            "wake_agent": recovery_count >= MAX_AUTO_RECOVERIES_BEFORE_WAKE,
            "reason": "session_unreachable",
        }

    if disk_free is not None and float(disk_free) < 5.0:
        return {
            **base,
            "status": "LOW_DISK",
            "next_action": "cleanup_disk",
            "wake_agent": True,
            "reason": f"low_disk_{float(disk_free):.1f}gb",
        }

    if hf_state.get("train_done"):
        if remote_state.get("eval_alive"):
            if anomaly:
                return {**base, "status": "ANOMALY", "next_action": "intervene", "wake_agent": True, "reason": anomaly}
            if stall_s >= STALL_THRESHOLD_S and int(remote_state.get("gpu_util") or 0) == 0:
                return {**base, "status": "STALLED", "next_action": "recover_eval", "wake_agent": True, "reason": "eval_stalled"}
            return {**base, "status": "EVAL_RUNNING", "next_action": "wait", "wake_agent": False, "reason": "eval_healthy"}
        return {
            **base,
            "status": "READY_FOR_EVAL",
            "next_action": "launch_eval",
            "wake_agent": bool(anomaly or recovery_count >= MAX_AUTO_RECOVERIES_BEFORE_WAKE),
            "reason": anomaly or "eval_not_running",
        }

    if remote_state.get("train_alive"):
        if anomaly:
            return {**base, "status": "ANOMALY", "next_action": "intervene", "wake_agent": True, "reason": anomaly}
        if sync_stale:
            return {**base, "status": "ANOMALY", "next_action": "intervene", "wake_agent": True, "reason": "hf_sync_stale"}
        if stall_s >= STALL_THRESHOLD_S and int(remote_state.get("gpu_util") or 0) == 0:
            return {**base, "status": "STALLED", "next_action": "recover_train", "wake_agent": True, "reason": "train_stalled"}
        return {**base, "status": "TRAINING_OK", "next_action": "wait", "wake_agent": False, "reason": "train_healthy"}

    reason = anomaly or ("process_exited" if cur_log_lines > 0 else "not_started")
    wake = bool(anomaly or recovery_count >= MAX_AUTO_RECOVERIES_BEFORE_WAKE)
    return {**base, "status": "NEEDS_RECOVERY", "next_action": "recover_and_resume", "wake_agent": wake, "reason": reason}


def bootstrap_colab(
    session: str,
    hf_repo: str,
    hf_token: str,
    prefix: str,
    gpu: str,
    *,
    expected_manifest: dict[str, Any] | None = None,
) -> None:
    """Pull missing/stale assets from HF `<prefix>/assets` on Colab and unpack workspaces."""
    manifest_json = json.dumps(expected_manifest or {}, sort_keys=True)
    bootstrap_code = f"""
import gzip, importlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
from huggingface_hub import snapshot_download

prefix = {prefix!r}
hf_repo = {hf_repo!r}
expected = json.loads({manifest_json!r})
stamp_path = Path('/content/.assets_stamp.json')
current = json.loads(stamp_path.read_text()) if stamp_path.exists() else {{}}

need_dl = not expected or current != expected or not all(
    Path('/content/' + name).exists() for name in (expected or {{}})
)
if need_dl:
    dl_root = Path('/content/hf_assets_dl')
    shutil.rmtree(dl_root, ignore_errors=True)
    snapshot_download(
        repo_id=hf_repo,
        repo_type='model',
        allow_patterns=f'{{prefix}}/assets/*',
        local_dir=str(dl_root),
        token=os.environ.get('HF_TOKEN'),
    )
    src_dir = dl_root / prefix / 'assets'
    for item in src_dir.iterdir():
        if item.is_file() and item.name != 'manifest.json':
            shutil.copy2(item, Path('/content') / item.name)
    m_src = src_dir / 'manifest.json'
    if m_src.exists():
        shutil.copy2(m_src, stamp_path)
    shutil.rmtree(dl_root, ignore_errors=True)
    print('ASSETS_DOWNLOADED', flush=True)
else:
    print('ASSETS_CACHED', flush=True)

with zipfile.ZipFile('/content/code_latest.zip', 'r') as z:
    z.extractall('/content')

data_train = Path('/content/data/train')
data_train.mkdir(parents=True, exist_ok=True)
if Path('/content/train_tools_catalog.json').exists():
    shutil.copy2('/content/train_tools_catalog.json', data_train / 'train_tools_catalog.json')

env_dir = data_train / 'train_environment'
env_dir.mkdir(parents=True, exist_ok=True)
if not (env_dir / 'train_0001').exists() and Path('/content/train_environment.zip').exists():
    with zipfile.ZipFile('/content/train_environment.zip', 'r') as z:
        z.extractall(str(env_dir))

zt_dir = data_train / 'zaic_tools'
zt_dir.mkdir(parents=True, exist_ok=True)
if not (zt_dir / 'executor.py').exists() and Path('/content/zaic_tools.zip').exists():
    with zipfile.ZipFile('/content/zaic_tools.zip', 'r') as z:
        z.extractall(str(zt_dir))

train_jsonl = data_train / 'train.jsonl'
if not train_jsonl.exists() and Path('/content/train.jsonl.gz').exists():
    with gzip.open('/content/train.jsonl.gz', 'rt', encoding='utf-8') as f_in, open(train_jsonl, 'w', encoding='utf-8') as f_out:
        for ln in f_in:
            f_out.write(ln)

splits = Path('/content/configs/splits')
splits.mkdir(parents=True, exist_ok=True)
for fn in ('dev_v1.json', 'fold_0_ids.json'):
    if Path('/content/' + fn).exists():
        shutil.copy2('/content/' + fn, splits / fn)

for mod, pkg in [('liger_kernel', 'liger-kernel'), ('peft', 'peft'), ('trl', 'trl'), ('vllm', 'vllm')]:
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)
try:
    import importlib.metadata as _md
    _ao_ver = _md.version('torchao')
    _ao_parts = tuple(int(''.join(ch for ch in p if ch.isdigit()) or 0) for p in _ao_ver.split('.')[:2])
    if _ao_parts < (0, 16):
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True)
        print(f'Uninstalled incompatible torchao=={{_ao_ver}}', flush=True)
except Exception:
    pass
for mod, pkg in [('fla', 'flash-linear-attention'), ('kernels', 'kernels>=0.17.2,<0.18.0')]:
    try:
        import importlib.metadata as _md
        importlib.import_module(mod)
        if mod == 'kernels':
            _kver = _md.version('kernels')
            _kp = tuple(int(x) for x in _kver.split('.')[:2])
            if not ((0, 17) <= _kp < (0, 18)):
                raise ImportError(_kver)
    except ImportError:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True, timeout=120)
        except Exception as exc:
            print(f'WARN: optional kernel package {{pkg}} skipped: {{exc}}', flush=True)
print('BOOTSTRAP_OK', flush=True)
"""
    res_out = run_remote_py(
        session,
        bootstrap_code,
        env=[f"{ENV_TOKEN}={hf_token}", f"{ENV_REPO}={hf_repo}"],
        timeout=300,
        gpu=gpu,
    )
    if "BOOTSTRAP_OK" not in res_out:
        raise RuntimeError(f"Colab bootstrap failed:\n{res_out[-1500:]}")
    _append_log(prefix, f"Colab '{session}' bootstrap verified.")


def _free_zombie_vram_if_idle(session: str, remote_state: dict[str, Any]) -> None:
    if (
        remote_state.get("colab_alive")
        and not remote_state.get("train_alive")
        and not remote_state.get("eval_alive")
        and int(remote_state.get("vram_used_mb") or 0) > 3500
    ):
        try:
            run_remote_py(
                session,
                "import subprocess, time\n"
                "subprocess.run(['pkill', '-9', '-f', '[t]rain_qwen38_sft\\.py|[c]olab_qwen35_sft.*worker-eval|[s]cripts/run_agent\\.py|[V]LLM::EngineCore'], check=False)\n"
                "time.sleep(2)\n",
                timeout=30,
            )
        except Exception:
            pass
        restart_kernel(session)
        ping_session(session, timeout=25)


def launch_train(
    session: str,
    hf_repo: str,
    hf_token: str,
    prefix: str,
    gpu: str,
    sft_config: str,
    *,
    data_file: str = DEFAULT_DATA_FILE,
    save_steps: int = 40,
    hf_interval_min: float = 8.0,
    push_merged: bool = True,
) -> None:
    """Launch train_qwen38_sft.py detached on Colab and confirm startup."""
    data_name = Path(data_file).name
    cmd_args = [
        '/content/scripts/train_qwen38_sft.py',
        '--config', f'/content/{sft_config}',
        '--data', f'/content/{data_name}',
        '--out', f'/content/experiments/{prefix}',
        '--hf-repo', hf_repo,
        '--hf-prefix', prefix,
        '--save-steps', str(int(save_steps)),
        '--hf-interval-min', str(hf_interval_min),
        '--max-wall-minutes', '0',
    ]
    if push_merged:
        cmd_args.append('--push-merged')
    launcher = f"""
import os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, '/content')
from src.core.hf_sync import HFSyncer

syncer = HFSyncer({hf_repo!r})
user = syncer.preflight()
print(f'Colab HF preflight OK: {{user}} -> {hf_repo}', flush=True)

def _proc_alive(pid_str):
    st_p = Path(f'/proc/{{pid_str}}/status')
    if not st_p.exists():
        return False
    try:
        txt = st_p.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return False
    for ln in txt.splitlines():
        if ln.startswith('State:'):
            return ln.split(':', 1)[1].strip()[:1].upper() not in ('Z', 'X', '')
    return False

chk = subprocess.run(['pgrep', '-af', 'train_qwen38_sft.py'], capture_output=True, text=True)
live_lines = [ln for ln in chk.stdout.strip().splitlines() if ln.strip() and ln.split()[0].isdigit() and _proc_alive(ln.split()[0])]
if live_lines:
    print('ALREADY_RUNNING:', '; '.join(live_lines), flush=True)
    sys.exit(0)

log_path = Path('/content/sft_v2_train.log')
pid_path = Path('/content/sft_v2_train.pid')
cmd = [sys.executable, '-u', *{cmd_args!r}]
env = {{**os.environ, 'PYTHONPATH': '/content:' + os.environ.get('PYTHONPATH', ''), 'HF_SYNC_REPO': {hf_repo!r}}}
log_f = open(log_path, 'a', encoding='utf-8')
log_f.write(f"\\n=== LAUNCH {{time.strftime('%Y-%m-%d %H:%M:%S')}} ===\\n")
log_f.flush()
proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log_f, stderr=subprocess.STDOUT, env=env, start_new_session=True, close_fds=True)
pid_path.write_text(str(proc.pid), encoding='utf-8')
time.sleep(15)
alive = Path(f'/proc/{{proc.pid}}').exists() and proc.poll() is None
print(f'TRAIN_LAUNCHED PID={{proc.pid}} ALIVE={{alive}}', flush=True)
if not alive:
    print('LOG_TAIL:\\n' + log_path.read_text(encoding='utf-8', errors='replace')[-2000:], flush=True)
    sys.exit(1)
"""
    out = run_remote_py(
        session,
        launcher,
        env=[f"{ENV_TOKEN}={hf_token}", f"{ENV_REPO}={hf_repo}"],
        timeout=120,
        gpu=gpu,
    )
    if "TRAIN_LAUNCHED" not in out and "ALREADY_RUNNING" not in out:
        raise RuntimeError(f"Failed launching training:\n{out[-1500:]}")
    _append_log(prefix, f"Detached training started on '{session}' ({gpu}).")


def _build_remote_eval_args(
    *,
    prefix: str,
    split: str = "fold0",
    transcripts: str = "oracle",
    enable_thinking: bool = True,
    eval_config: str = DEFAULT_EVAL_CONFIG,
    model_id: str | None = None,
    ids_path: str | None = None,
    preds_file: str | None = None,
    trace_file: str | None = None,
    root: Path = Path("/content"),
) -> list[str]:
    split_map = {tag: (ids_rel, pred_name) for tag, ids_rel, pred_name in EVAL_SPLITS}
    default_ids_rel, default_pred_name = split_map.get(split, ("configs/splits/fold_0_ids.json", f"{split}_preds.jsonl"))
    out_dir = root / "experiments" / prefix
    m_id = model_id or str(out_dir / "merged")
    ids_p = ids_path or str(root / default_ids_rel)
    p_file = preds_file or str(out_dir / f"preds_{split}" / default_pred_name)
    t_file = trace_file or str(out_dir / f"preds_{split}" / f"{split}_traces.jsonl")
    return [
        sys.executable,
        str(root / "scripts" / "run_agent.py"),
        "--config",
        str(root / eval_config),
        "--split",
        "train",
        "--root",
        str(root / "data"),
        "--ids",
        str(ids_p),
        "--backend",
        "vllm",
        "--model-id",
        str(m_id),
        "--prompt-style",
        "qwen35",
        "--enable-thinking" if enable_thinking else "--no-enable-thinking",
        "--transcripts",
        str(transcripts),
        "--out",
        str(p_file),
        "--trace-out",
        str(t_file),
        "--resume",
    ]


def _build_remote_eval_cmd(
    *,
    prefix: str,
    split: str = "fold0",
    transcripts: str = "oracle",
    enable_thinking: bool = True,
    eval_config: str = DEFAULT_EVAL_CONFIG,
    root: Path = Path("/content"),
) -> str:
    return " ".join(
        _build_remote_eval_args(
            prefix=prefix,
            split=split,
            transcripts=transcripts,
            enable_thinking=enable_thinking,
            eval_config=eval_config,
            root=root,
        )
    )


def launch_eval(
    session: str,
    hf_repo: str,
    hf_token: str,
    prefix: str,
    gpu: str,
    eval_config: str,
    *,
    eval_splits: list[str] | None = None,
    transcripts: str = "oracle",
    enable_thinking: bool = True,
) -> None:
    """Launch detached post-SFT Fold0 / Dev vLLM evaluation worker on Colab."""
    worker_args = [
        '-m', 'scripts.colab_qwen35_sft',
        '--worker-eval',
        '--prefix', prefix,
        '--hf-repo', hf_repo,
        '--eval-config', eval_config,
        '--eval-transcripts', transcripts,
        '--enable-thinking' if enable_thinking else '--no-enable-thinking',
    ]
    if eval_splits:
        worker_args.extend(['--eval-splits', *eval_splits])
    launcher = f"""
import os, subprocess, sys, time
from pathlib import Path

def _proc_alive(pid_str):
    st_p = Path(f'/proc/{{pid_str}}/status')
    if not st_p.exists():
        return False
    try:
        txt = st_p.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return False
    for ln in txt.splitlines():
        if ln.startswith('State:'):
            return ln.split(':', 1)[1].strip()[:1].upper() not in ('Z', 'X', '')
    return False

chk = subprocess.run(['pgrep', '-af', 'colab_qwen35_sft.*worker-eval|scripts/run_agent.py'], capture_output=True, text=True)
live_lines = [ln for ln in chk.stdout.strip().splitlines() if ln.strip() and ln.split()[0].isdigit() and _proc_alive(ln.split()[0])]
if live_lines:
    print('EVAL_ALREADY_RUNNING:', '; '.join(live_lines), flush=True)
    sys.exit(0)

log_path = Path('/content/sft_v2_eval.log')
pid_path = Path('/content/sft_v2_eval.pid')
cmd = [sys.executable, '-u', *{worker_args!r}]
env = {{
    **os.environ,
    'PYTHONPATH': '/content:' + os.environ.get('PYTHONPATH', ''),
    'HF_SYNC_REPO': {hf_repo!r},
    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
}}
log_f = open(log_path, 'a', encoding='utf-8')
log_f.write(f"\\n=== EVAL LAUNCH {{time.strftime('%Y-%m-%d %H:%M:%S')}} ===\\n")
log_f.flush()
proc = subprocess.Popen(cmd, cwd='/content', stdin=subprocess.DEVNULL, stdout=log_f, stderr=subprocess.STDOUT, env=env, start_new_session=True, close_fds=True)
pid_path.write_text(str(proc.pid), encoding='utf-8')
time.sleep(10)
alive = Path(f'/proc/{{proc.pid}}').exists() and proc.poll() is None
print(f'EVAL_LAUNCHED PID={{proc.pid}} ALIVE={{alive}}', flush=True)
if not alive:
    print('LOG_TAIL:\\n' + log_path.read_text(encoding='utf-8', errors='replace')[-2000:], flush=True)
    sys.exit(1)
"""
    out = run_remote_py(
        session,
        launcher,
        env=[f"{ENV_TOKEN}={hf_token}", f"{ENV_REPO}={hf_repo}"],
        timeout=120,
        gpu=gpu,
    )
    if "EVAL_LAUNCHED" not in out and "EVAL_ALREADY_RUNNING" not in out:
        raise RuntimeError(f"Failed launching eval:\n{out[-1500:]}")
    _append_log(prefix, f"Detached evaluation started on '{session}' ({gpu}).")


def is_merged_checkpoint_ready(merged_dir: Path) -> bool:
    """Return True only when .complete, config, tokenizer, and weights all live at top level of merged_dir."""
    return (
        (merged_dir / ".complete").exists()
        and (merged_dir / "config.json").exists()
        and (merged_dir / "tokenizer.json").exists()
        and any(merged_dir.glob("*.safetensors"))
    )


def prepare_merged_checkpoint(syncer: Any, prefix: str, out_dir: Path) -> bool:
    """Ensure flattened pre-merged checkpoint files live directly under out_dir/'merged'."""
    merged_dir = out_dir / "merged"
    if is_merged_checkpoint_ready(merged_dir):
        return True

    nested_dir = merged_dir / prefix / "merged"
    if (nested_dir / "config.json").exists() and (nested_dir / "tokenizer.json").exists():
        merged_dir.mkdir(parents=True, exist_ok=True)
        for src in nested_dir.iterdir():
            if src.is_file() and src.name != ".complete":
                shutil.move(str(src), str(merged_dir / src.name))
        shutil.rmtree(merged_dir / prefix, ignore_errors=True)
        shutil.rmtree(merged_dir / ".cache", ignore_errors=True)
        if (
            (merged_dir / "config.json").exists()
            and (merged_dir / "tokenizer.json").exists()
            and any(merged_dir.glob("*.safetensors"))
        ):
            (merged_dir / ".complete").write_text("ok", encoding="utf-8")
            print(f"[worker-eval] Flattened cached pre-merged checkpoint into {merged_dir}", flush=True)
            return True

    if not syncer.exists(f"{prefix}/merged/config.json"):
        return False

    (merged_dir / ".complete").unlink(missing_ok=True)
    print(f"[worker-eval] Downloading pre-merged pure-text checkpoint from HF {prefix}/merged", flush=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    dl_merged = Path(syncer.download_dir(f"{prefix}/merged", out_dir.parent))
    if dl_merged.resolve() != merged_dir.resolve() and dl_merged.exists():
        merged_dir.mkdir(parents=True, exist_ok=True)
        for src in dl_merged.iterdir():
            if src.is_file() and src.name != ".complete":
                shutil.copy2(src, merged_dir / src.name)
    if (
        (merged_dir / "config.json").exists()
        and (merged_dir / "tokenizer.json").exists()
        and any(merged_dir.glob("*.safetensors"))
    ):
        (merged_dir / ".complete").write_text("ok", encoding="utf-8")
        return True
    return False


def run_remote_eval_worker(
    prefix: str,
    hf_repo: str | None,
    eval_config: str,
    root: Path = Path("/content"),
    *,
    eval_splits: list[str] | None = None,
    transcripts: str = "oracle",
    enable_thinking: bool = True,
) -> None:
    """In-Colab worker: reuse/merge pure-text Qwen3_5ForCausalLM, then run requested split vLLM eval."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    repo, _token, syncer = hf_preflight(hf_repo)
    out_dir = root / "experiments" / prefix
    adapter_dir = out_dir / "adapter"
    merged_dir = out_dir / "merged"

    # Prefer ready HF <prefix>/merged/ snapshot to avoid redundant merging on L4
    prepare_merged_checkpoint(syncer, prefix, out_dir)

    if not is_merged_checkpoint_ready(merged_dir):
        if not (adapter_dir / "adapter_model.safetensors").exists():
            hf_ad = syncer.download_dir(f"{prefix}/adapter", root / "hf_eval_adapter")
            adapter_dir = hf_ad
        assert (adapter_dir / "adapter_model.safetensors").exists(), f"Adapter not found at {adapter_dir}"

        import gc
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.agent.qwen35_sft import save_qwen35_text_only_checkpoint

        base_id = "Qwen/Qwen3.5-4B"
        print(f"[worker-eval] Merging {base_id} + {adapter_dir} -> {merged_dir}", flush=True)
        tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_id, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}
        )
        peft_m = PeftModel.from_pretrained(base, str(adapter_dir))
        merged = peft_m.merge_and_unload()
        save_qwen35_text_only_checkpoint(merged, tok, merged_dir)
        (merged_dir / ".complete").write_text("ok", encoding="utf-8")
        del merged, peft_m, base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    from src.core.hf_sync import DirWatcher

    resolved_transcripts = transcripts
    if transcripts == "asr":
        cand = root / "fold0_transcripts.jsonl"
        if not cand.exists():
            cand = root / "experiments" / "asr_fold0" / "fold0_transcripts.jsonl"
        if not cand.exists() and syncer.exists(f"{prefix}/assets/fold0_transcripts.jsonl"):
            dl = _download_file(syncer, f"{prefix}/assets/fold0_transcripts.jsonl", root / "hf_eval_assets")
            if dl is not None and dl.exists():
                cand = dl
        resolved_transcripts = str(cand)

    target_splits = eval_splits or [tag for tag, _, _ in EVAL_SPLITS]
    env = {**os.environ, "PYTHONPATH": f"{root}:{os.environ.get('PYTHONPATH', '')}"}
    for tag, ids_rel, pred_name in EVAL_SPLITS:
        if tag not in target_splits:
            continue
        hf_eval_path = f"{prefix}/eval_{tag}"
        hf_preds_path = f"{prefix}/preds_{tag}"
        if syncer.exists(f"{hf_eval_path}/metrics.json"):
            print(f"[worker-eval] Skipping {tag}: {hf_eval_path}/metrics.json already on HF.", flush=True)
            continue

        preds_dir = out_dir / f"preds_{tag}"
        eval_dir = out_dir / f"eval_{tag}"
        preds_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        preds_file = preds_dir / pred_name
        trace_file = preds_dir / f"{tag}_traces.jsonl"

        if syncer.exists(f"{hf_preds_path}/{pred_name}"):
            dl = syncer.download_dir(hf_preds_path, root / "hf_eval_resume")
            if (dl / pred_name).exists():
                shutil.copy2(dl / pred_name, preds_file)
            if (dl / trace_file.name).exists():
                shutil.copy2(dl / trace_file.name, trace_file)

        watcher = DirWatcher(syncer, [(preds_dir, hf_preds_path)], interval_s=300, message_prefix=f"{prefix} {tag}").start()
        t0 = time.time()
        try:
            cmd = _build_remote_eval_args(
                prefix=prefix,
                split=tag,
                transcripts=resolved_transcripts,
                enable_thinking=enable_thinking,
                eval_config=eval_config,
                model_id=str(merged_dir),
                ids_path=str(root / ids_rel),
                preds_file=str(preds_file),
                trace_file=str(trace_file),
                root=root,
            )
            rc = subprocess.run(cmd, env=env).returncode
            if rc != 0:
                print(f"[worker-eval] vLLM failed (rc={rc}) for {tag}; falling back to transformers.", flush=True)
                time.sleep(3)
                cmd[cmd.index("vllm")] = "transformers"
                subprocess.run(cmd, env=env, check=True)
        finally:
            watcher.stop(final=True)

        eval_cmd = [
            sys.executable, str(root / "scripts" / "run_eval.py"),
            "--predictions", str(preds_file),
            "--root", str(root / "data"),
            "--ids", str(root / ids_rel),
            "--out", str(eval_dir),
        ]
        subprocess.run(eval_cmd, env=env, check=True)
        (eval_dir / "timing.json").write_text(
            json.dumps({"model": str(merged_dir), "wall_time_s": round(time.time() - t0, 2)}, indent=2),
            encoding="utf-8",
        )
        syncer.push(eval_dir, hf_eval_path, f"{prefix} {tag} eval metrics", raise_on_fail=True)
        print(f"[worker-eval] {tag} DONE in {time.time() - t0:.1f}s -> pushed {hf_eval_path}", flush=True)


def download_results(syncer: HFSyncer, prefix: str, root: Path = ROOT) -> Path:
    exp = _exp_dir(prefix, root)
    cache_root = exp / "_hf_download"

    for rel_file in ("training_summary.json", "checkpoints/latest/progress.json", "eval_dev299/metrics.json", "eval_dev299/timing.json"):
        src = _download_file(syncer, f"{prefix}/{rel_file}", cache_root)
        if src is not None and src.exists():
            dest = exp / rel_file
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    if syncer.exists(f"{prefix}/eval_fold0/metrics.json"):
        dl = syncer.download_dir(f"{prefix}/eval_fold0", cache_root)
        target = exp / "eval_fold0"
        target.mkdir(parents=True, exist_ok=True)
        for p in dl.rglob("*"):
            if p.is_file():
                dest = target / p.relative_to(dl)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)

    _append_log(prefix, f"Downloaded HF artifacts to {exp}", root)
    return exp


def _resolve_target_session_and_gpu(args: argparse.Namespace, hf_state: dict[str, Any]) -> tuple[str, str]:
    if hf_state.get("train_done"):
        e_gpu = getattr(args, "eval_gpu", None) or args.gpu
        e_sess = getattr(args, "eval_session", None)
        if not e_sess:
            e_sess = f"{args.session}-eval" if e_gpu != args.gpu else args.session
        return e_sess, e_gpu
    return args.session, args.gpu


def _resolve_transcripts_asset(eval_transcripts: str) -> str | None:
    if eval_transcripts in ("oracle", ""):
        return None
    if eval_transcripts == "asr":
        cand = ROOT / "experiments" / "asr_fold0" / "fold0_transcripts.jsonl"
        return str(cand) if cand.exists() else None
    p = Path(eval_transcripts)
    return str(p) if p.exists() else None


def action_check(args: argparse.Namespace) -> dict[str, Any]:
    hf_repo, _token, syncer = hf_preflight(args.hf_repo)
    eval_splits = getattr(args, "eval_splits", None)
    hf_state = inspect_hf_state(syncer, args.prefix, eval_splits=eval_splits)
    target_session, target_gpu = _resolve_target_session_and_gpu(args, hf_state)
    data_name = Path(getattr(args, "data_file", DEFAULT_DATA_FILE)).name
    remote_state = probe_colab_state(
        target_session,
        args.prefix,
        target_gpu,
        data_name=data_name,
        eval_splits=eval_splits,
    )
    prev = _load_supervisor_state(args.prefix)
    finalized = (_exp_dir(args.prefix) / "REPORT.md").exists()
    verdict = assess_status(hf_state, remote_state, prev, finalized_locally=finalized)
    snapshot = {
        **verdict,
        "target_session": target_session,
        "target_gpu": target_gpu,
        "hf": hf_state,
        "remote": {k: v for k, v in remote_state.items() if k != "assets_stamp"},
    }
    _save_supervisor_state(args.prefix, snapshot)
    lm = remote_state.get("live_metrics") or {}
    _append_log(
        args.prefix,
        f"CHECK status={verdict['status']} step={hf_state['latest_step']}/{hf_state['latest_max_steps']} "
        f"live_step={lm.get('step')} loss={lm.get('loss') or hf_state['latest_loss']} "
        f"grad_norm={lm.get('grad_norm') or hf_state.get('latest_grad_norm')} "
        f"vram_gb={lm.get('vram_peak_gb')}/{lm.get('vram_total_gb')} speed={lm.get('steps_per_sec')}it/s "
        f"gpu={remote_state.get('gpu_util')}%/{remote_state.get('vram_used_mb')}MB "
        f"disk={remote_state.get('disk_free_gb')}GB "
        f"train_alive={remote_state.get('train_alive')} eval_alive={remote_state.get('eval_alive')} reason={verdict['reason']}",
    )
    return snapshot


def action_ensure(args: argparse.Namespace) -> dict[str, Any]:
    hf_repo, hf_token, syncer = hf_preflight(args.hf_repo)
    eval_splits = getattr(args, "eval_splits", None)
    data_file = getattr(args, "data_file", DEFAULT_DATA_FILE)
    eval_transcripts = getattr(args, "eval_transcripts", "oracle")
    enable_thinking = bool(getattr(args, "enable_thinking", True))
    hf_state = inspect_hf_state(syncer, args.prefix, eval_splits=eval_splits)
    finalized = (_exp_dir(args.prefix) / "REPORT.md").exists()
    if hf_state["all_done"]:
        download_results(syncer, args.prefix)
        prev = _load_supervisor_state(args.prefix)
        verdict = assess_status(hf_state, {"colab_alive": True}, prev, finalized_locally=finalized)
        _save_supervisor_state(args.prefix, {**verdict, "hf": hf_state})
        return {**verdict, "hf": hf_state}

    target_session, target_gpu = _resolve_target_session_and_gpu(args, hf_state)
    # Stop finished G4 training session cleanly once switching over to dedicated L4 eval session
    if hf_state["train_done"] and target_session != args.session:
        subprocess.run(["colab", "stop", "-s", args.session], check=False, capture_output=True)

    t_asset = _resolve_transcripts_asset(eval_transcripts)
    manifest = sync_assets(
        syncer,
        args.prefix,
        data_file=data_file,
        sft_config=args.sft_config,
        eval_config=args.eval_config,
        eval_splits=eval_splits,
        transcripts_file=t_asset,
    )
    remote_state = probe_colab_state(
        target_session, args.prefix, target_gpu, data_name=Path(data_file).name, eval_splits=eval_splits
    )
    prev = _load_supervisor_state(args.prefix)
    verdict = assess_status(hf_state, remote_state, prev, finalized_locally=finalized)

    if verdict["status"] in ("TRAINING_OK", "EVAL_RUNNING"):
        snap = {**verdict, "hf": hf_state, "remote": {k: v for k, v in remote_state.items() if k != "assets_stamp"}}
        _save_supervisor_state(args.prefix, snap)
        return snap

    if verdict["status"] == "ANOMALY" and str(verdict.get("reason") or "") in ("NAN_LOSS", "GRAD_EXPLOSION", "LOSS_DIVERGENCE"):
        _append_log(args.prefix, f"Refusing blind relaunch on model anomaly {verdict['reason']}; escalating to agent.")
        snap = {**verdict, "wake_agent": True, "hf": hf_state, "remote": remote_state}
        _save_supervisor_state(args.prefix, snap)
        return snap

    # Recovery or stage launch
    recovery_count = int(prev.get("recovery_count") or 0) + (1 if verdict["status"] in ("NEEDS_RECOVERY", "STALLED", "ANOMALY") else 0)
    gpu = ensure_live_session(target_session, target_gpu)
    remote_state = probe_colab_state(
        target_session, args.prefix, gpu, data_name=Path(data_file).name, eval_splits=eval_splits
    )
    if verdict["status"] == "STALLED":
        restart_kernel(target_session)
        run_remote_py(
            target_session,
            "import subprocess; subprocess.run(['pkill', '-f', '[t]rain_qwen38_sft\\.py|[c]olab_qwen35_sft.*worker-eval|[s]cripts/run_agent\\.py|[V]LLM::EngineCore'])",
            timeout=30,
            gpu=gpu,
        )
        remote_state = probe_colab_state(
            target_session, args.prefix, gpu, data_name=Path(data_file).name, eval_splits=eval_splits
        )

    _free_zombie_vram_if_idle(target_session, remote_state)
    bootstrap_colab(target_session, hf_repo, hf_token, args.prefix, gpu, expected_manifest=manifest)
    if hf_state["train_done"]:
        launch_eval(
            target_session,
            hf_repo,
            hf_token,
            args.prefix,
            gpu,
            args.eval_config,
            eval_splits=eval_splits,
            transcripts=eval_transcripts,
            enable_thinking=enable_thinking,
        )
    else:
        launch_train(
            target_session,
            hf_repo,
            hf_token,
            args.prefix,
            gpu,
            args.sft_config,
            data_file=data_file,
            save_steps=int(getattr(args, "save_steps", 40)),
            hf_interval_min=float(getattr(args, "hf_interval_min", 8.0)),
            push_merged=bool(getattr(args, "push_merged", True)),
        )

    after_remote = probe_colab_state(
        target_session, args.prefix, gpu, data_name=Path(data_file).name, eval_splits=eval_splits
    )
    after_verdict = assess_status(hf_state, after_remote, {**prev, "recovery_count": recovery_count}, finalized_locally=finalized)
    snap = {**after_verdict, "recovery_count": recovery_count, "hf": hf_state, "remote": after_remote}
    _save_supervisor_state(args.prefix, snap)
    return snap


def _format_cli_shared_flags(args: argparse.Namespace) -> str:
    parts = [
        f"--session {args.session}",
        f"--gpu {args.gpu}",
        f"--prefix {args.prefix}",
        f"--hf-repo {args.hf_repo or DEFAULT_HF_REPO}",
        f"--sft-config {args.sft_config}",
        f"--eval-config {args.eval_config}",
        f"--data-file {args.data_file}",
        f"--save-steps {args.save_steps}",
        f"--hf-interval-min {args.hf_interval_min}",
        f"--eval-transcripts {args.eval_transcripts}",
        "--enable-thinking" if args.enable_thinking else "--no-enable-thinking",
        "--push-merged" if args.push_merged else "--no-push-merged",
    ]
    if args.eval_gpu:
        parts.append(f"--eval-gpu {args.eval_gpu}")
    if args.eval_session:
        parts.append(f"--eval-session {args.eval_session}")
    if args.eval_splits:
        parts.append("--eval-splits " + " ".join(args.eval_splits))
    return " ".join(parts)


def setup_orca_automation(args: argparse.Namespace) -> None:
    """Create or update the periodic Orca `antigravity` automation with Python `--precheck`."""
    py_exe = sys.executable
    script_path = Path(__file__).resolve()
    shared = _format_cli_shared_flags(args)
    precheck_cmd = f'"{py_exe}" "{script_path}" --action precheck {shared}'
    orca_name = args.orca_name or DEFAULT_ORCA_NAME
    prompt = (
        f"Automated Colab SFT watchdog escalation for {args.prefix}. "
        f"Run `{py_exe} {script_path} --action check {shared}`. "
        f"Inspect experiments/{args.prefix}/supervisor_state.json (including live_metrics loss, grad_norm, lr, vram_peak_gb, speed) and supervisor.log. "
        f"If COMPLETE, run `--action download-results`, write experiments/{args.prefix}/REPORT.md, "
        f"update reports/LLM_AGENT_EXPERIMENTS_DASHBOARD.md, verify HF repo files, stop idle Colab sessions, "
        f"and disable this Orca automation (`orca automations disable`). Otherwise diagnose the anomaly/stall/OOM/WebSocket drop, "
        f"fix the code/config/tests, and run `--action ensure {shared}` to resume cleanly from HF checkpoints/latest."
    )
    existing = json.loads(
        subprocess.run(["orca", "automations", "list", "--json"], capture_output=True, text=True, check=True).stdout
    )
    items = (existing.get("result") or {}).get("automations") or existing.get("result") or []
    match = next((a for a in items if isinstance(a, dict) and a.get("name") == orca_name), None)

    if match and match.get("id"):
        cmd = [
            "orca", "automations", "edit", str(match["id"]),
            "--trigger", args.cron,
            "--precheck", precheck_cmd,
            "--precheck-timeout", "300",
            "--prompt", prompt,
            "--provider", "antigravity",
            "--workspace", "active",
            "--reuse-session",
            "--enabled",
            "--json",
        ]
    else:
        cmd = [
            "orca", "automations", "create",
            "--name", orca_name,
            "--trigger", args.cron,
            "--precheck", precheck_cmd,
            "--precheck-timeout", "300",
            "--prompt", prompt,
            "--provider", "antigravity",
            "--workspace", "active",
            "--reuse-session",
            "--enabled",
            "--json",
        ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(res.stdout, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=["sync-assets", "bootstrap", "launch", "eval-launch", "check", "ensure", "precheck", "download-results", "setup-orca", "stop"],
        default="check",
    )
    parser.add_argument("--worker-eval", action="store_true", help="Internal Colab eval worker entrypoint")
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--eval-session", default=None)
    parser.add_argument("--eval-gpu", default=None)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--sft-config", default=DEFAULT_SFT_CONFIG)
    parser.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--eval-splits", nargs="+", default=None)
    parser.add_argument("--eval-transcripts", default="oracle")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--push-merged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-steps", type=int, default=40)
    parser.add_argument("--hf-interval-min", type=float, default=8.0)
    parser.add_argument("--orca-name", default=DEFAULT_ORCA_NAME)
    parser.add_argument("--cron", default="*/5 * * * *")
    parser.add_argument("--force-assets", action="store_true")
    parser.add_argument("--no-auto-heal", action="store_true")
    args = parser.parse_args(argv)

    if args.worker_eval:
        run_remote_eval_worker(
            args.prefix,
            args.hf_repo,
            args.eval_config,
            eval_splits=args.eval_splits,
            transcripts=args.eval_transcripts,
            enable_thinking=args.enable_thinking,
        )
        return 0

    if args.action == "stop":
        subprocess.run(["colab", "stop", "-s", args.session], check=False)
        if args.eval_session:
            subprocess.run(["colab", "stop", "-s", args.eval_session], check=False)
        return 0

    if args.action == "setup-orca":
        setup_orca_automation(args)
        return 0

    hf_repo, hf_token, syncer = hf_preflight(args.hf_repo)
    t_asset = _resolve_transcripts_asset(args.eval_transcripts)

    if args.action == "sync-assets":
        sync_assets(
            syncer,
            args.prefix,
            force=args.force_assets,
            data_file=args.data_file,
            sft_config=args.sft_config,
            eval_config=args.eval_config,
            eval_splits=args.eval_splits,
            transcripts_file=t_asset,
        )
        return 0
    if args.action == "bootstrap":
        manifest = sync_assets(
            syncer,
            args.prefix,
            force=args.force_assets,
            data_file=args.data_file,
            sft_config=args.sft_config,
            eval_config=args.eval_config,
            eval_splits=args.eval_splits,
            transcripts_file=t_asset,
        )
        gpu = ensure_live_session(args.session, args.gpu)
        bootstrap_colab(args.session, hf_repo, hf_token, args.prefix, gpu, expected_manifest=manifest)
        return 0
    if args.action == "launch":
        manifest = sync_assets(
            syncer,
            args.prefix,
            force=args.force_assets,
            data_file=args.data_file,
            sft_config=args.sft_config,
            eval_config=args.eval_config,
            eval_splits=args.eval_splits,
            transcripts_file=t_asset,
        )
        gpu = ensure_live_session(args.session, args.gpu)
        remote = probe_colab_state(
            args.session, args.prefix, gpu, data_name=Path(args.data_file).name, eval_splits=args.eval_splits
        )
        _free_zombie_vram_if_idle(args.session, remote)
        bootstrap_colab(args.session, hf_repo, hf_token, args.prefix, gpu, expected_manifest=manifest)
        launch_train(
            args.session,
            hf_repo,
            hf_token,
            args.prefix,
            gpu,
            args.sft_config,
            data_file=args.data_file,
            save_steps=args.save_steps,
            hf_interval_min=args.hf_interval_min,
            push_merged=args.push_merged,
        )
        return 0
    if args.action == "eval-launch":
        e_gpu = args.eval_gpu or args.gpu
        e_sess = args.eval_session or (f"{args.session}-eval" if e_gpu != args.gpu else args.session)
        manifest = sync_assets(
            syncer,
            args.prefix,
            force=args.force_assets,
            data_file=args.data_file,
            sft_config=args.sft_config,
            eval_config=args.eval_config,
            eval_splits=args.eval_splits,
            transcripts_file=t_asset,
        )
        gpu = ensure_live_session(e_sess, e_gpu)
        remote = probe_colab_state(
            e_sess, args.prefix, gpu, data_name=Path(args.data_file).name, eval_splits=args.eval_splits
        )
        _free_zombie_vram_if_idle(e_sess, remote)
        bootstrap_colab(e_sess, hf_repo, hf_token, args.prefix, gpu, expected_manifest=manifest)
        launch_eval(
            e_sess,
            hf_repo,
            hf_token,
            args.prefix,
            gpu,
            args.eval_config,
            eval_splits=args.eval_splits,
            transcripts=args.eval_transcripts,
            enable_thinking=args.enable_thinking,
        )
        return 0
    if args.action == "download-results":
        download_results(syncer, args.prefix)
        return 0
    if args.action == "check":
        snap = action_check(args)
        print(json.dumps(snap, indent=2, ensure_ascii=False))
        return 0
    if args.action == "ensure":
        snap = action_ensure(args)
        print(json.dumps(snap, indent=2, ensure_ascii=False))
        return 0
    if args.action == "precheck":
        snap = action_check(args)
        if snap.get("wake_agent"):
            _append_log(args.prefix, f"[PRECHECK WAKE] status={snap['status']} reason={snap['reason']}")
            return 0
        if not args.no_auto_heal and snap["status"] in ("NEEDS_RECOVERY", "READY_FOR_EVAL", "STALLED"):
            try:
                healed = action_ensure(args)
                if healed.get("wake_agent"):
                    _append_log(args.prefix, f"[PRECHECK WAKE] post-heal status={healed['status']}")
                    return 0
                _append_log(args.prefix, f"[PRECHECK SKIP] auto-healed -> {healed['status']}")
                return 1
            except Exception as exc:
                _append_log(args.prefix, f"[PRECHECK WAKE] auto-heal failed: {exc}")
                return 0
        _append_log(args.prefix, f"[PRECHECK SKIP] status={snap['status']}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
