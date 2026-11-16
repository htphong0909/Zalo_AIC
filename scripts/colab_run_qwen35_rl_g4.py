"""Orchestrate Task 0 (3-beam ASR cache) and chunked Qwen3.5-4B RL/OPD training on Colab G4.

All RL/ASR outputs persist permanently to public HF repo ``htphong0909/zalo-aic-rl-checkpoints``,
while read-only base SFT V2 weights and cached audio archives pull from ``htphong0909/zalo-aic-sft-checkpoints``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.colab_runner import (  # noqa: E402
    ensure_session,
    package_codebase,
    ping_session,
    restart_kernel,
    run_command,
    upload_file,
)
from src.core.hf_sync import ENV_REPO, ENV_TOKEN, HFSyncer, resolve_repo_id  # noqa: E402

DEFAULT_RL_REPO = "htphong0909/zalo-aic-rl-checkpoints"
DEFAULT_BASE_REPO = "htphong0909/zalo-aic-sft-checkpoints"
SCRATCH = ROOT / "scripts" / "scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)


def _bootstrap_worker_header(
    *,
    rl_repo: str,
    base_repo: str,
    need_audio: bool,
    install_rl_pkgs: bool,
) -> str:
    return f'''try:
    import gc as _gc, sys as _sys
    _ip = globals().get("get_ipython", lambda: None)()
    if _ip is not None:
        for _d in (getattr(_ip, "user_ns", {{}}), getattr(_ip, "user_ns_hidden", {{}})):
            for _k in list(_d):
                if not str(_k).startswith("_"):
                    _d.pop(_k, None)
    for _attr in ("last_traceback", "last_value", "last_type", "last_exc"):
        setattr(_sys, _attr, None)
    _gc.collect()
    if "torch" in _sys.modules:
        _t = _sys.modules["torch"]
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
            _t.cuda.ipc_collect()
except Exception:
    pass

import gzip
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["WANDB_DISABLED"] = "true"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

try:
    subprocess.run(["pkill", "-9", "-f", "scripts/train_qwen35_rl.py"], check=False)
except Exception:
    pass

try:
    _apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    )
    _keep = {{os.getpid(), os.getppid()}}
    _killed = False
    for _ln in _apps.splitlines():
        if _ln.strip().isdigit() and int(_ln.strip()) not in _keep:
            try:
                os.kill(int(_ln.strip()), signal.SIGKILL)
                print("Cleaned stale GPU pid:", _ln.strip(), flush=True)
                _killed = True
            except OSError:
                pass
    if _killed:
        time.sleep(0.5)
    print("GPU:", subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip(), flush=True)
except Exception as _gexc:
    print("GPU check warn:", _gexc, flush=True)

work = Path("/content")
with zipfile.ZipFile("/content/code_latest.zip", "r") as z:
    z.extractall(str(work))
sys.path.insert(0, str(work))

try:
    import importlib.metadata as _md
    _ao = tuple(int("".join(c for c in p if c.isdigit()) or 0) for p in _md.version("torchao").split(".")[:2])
    if _ao < (0, 16):
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
except Exception:
    pass

req_pkgs = [("yaml", "pyyaml"), ("huggingface_hub", "huggingface_hub")]
if {need_audio!r}:
    req_pkgs.extend([("soundfile", "soundfile"), ("jiwer", "jiwer")])
if {install_rl_pkgs!r}:
    req_pkgs.extend([
        ("peft", "peft"),
        ("accelerate", "accelerate"),
        ("fla", "flash-linear-attention"),
        ("kernels", "kernels>=0.17.2,<0.18.0"),
    ])
for mod, pkg in req_pkgs:
    if importlib.util.find_spec(mod) is None:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True, timeout=180)
        except Exception as exc:
            print(f"WARN optional install {{pkg}} failed: {{exc}}", flush=True)
if {install_rl_pkgs!r} and importlib.util.find_spec("vllm") is None:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--only-binary=:all:", "vllm"],
            check=True,
            timeout=300,
        )
    except Exception as _vexc:
        print(f"WARN optional vllm install failed: {{_vexc}}", flush=True)

from huggingface_hub import hf_hub_download
from src.core.hf_sync import DirWatcher, HFSyncer

token = os.environ.get("HF_TOKEN") or None
rl_syncer = HFSyncer({rl_repo!r}, token=token, private=False, retries=5)
print("HF preflight OK:", rl_syncer.preflight(), "->", {rl_repo!r}, flush=True)

data_train = Path("/content/data/train")
data_train.mkdir(parents=True, exist_ok=True)
cache_dl = Path("/content/_hf_assets")
cache_dl.mkdir(parents=True, exist_ok=True)
asset_prefix = "exp_qwen35_sft_v3_clean/assets"

def _fetch_asset(name: str, dest_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id={base_repo!r},
            filename=f"{{asset_prefix}}/{{name}}",
            repo_type="model",
            token=token,
            local_dir=str(dest_dir),
        )
    )

if not (data_train / "train_tools_catalog.json").exists():
    shutil.copy2(_fetch_asset("train_tools_catalog.json", cache_dl), data_train / "train_tools_catalog.json")

if not (data_train / "train.jsonl").exists():
    gz_p = _fetch_asset("train.jsonl.gz", cache_dl)
    with gzip.open(gz_p, "rb") as fi, open(data_train / "train.jsonl", "wb") as fo:
        shutil.copyfileobj(fi, fo)

if not (data_train / "zaic_tools" / "executor.py").exists():
    zt_zip = _fetch_asset("zaic_tools.zip", cache_dl)
    (data_train / "zaic_tools").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zt_zip, "r") as z:
        z.extractall(str(data_train / "zaic_tools"))

if not (data_train / "train_environment" / "train_0001").exists():
    env_zip = _fetch_asset("train_environment.zip", cache_dl)
    (data_train / "train_environment").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(env_zip, "r") as z:
        z.extractall(str(data_train / "train_environment"))

if {need_audio!r}:
    audio_dir = data_train / "audio"
    if not (audio_dir.exists() and any(audio_dir.iterdir())):
        azip = _fetch_asset("audio_train_only.zip", cache_dl)
        with zipfile.ZipFile(azip, "r") as z:
            first_wav = next((n for n in z.namelist() if n.endswith(".wav")), "")
            if first_wav.startswith("data/"):
                dest_audio = "/content"
            elif first_wav.startswith("audio/"):
                dest_audio = str(data_train)
            else:
                audio_dir.mkdir(parents=True, exist_ok=True)
                dest_audio = str(audio_dir)
            z.extractall(dest_audio)
    wav_count = len(list(audio_dir.glob("*/*.wav")))
    assert wav_count > 0, f"No wav files extracted under {{audio_dir}}"
    print("Audio ready:", wav_count, "wavs", flush=True)
'''


def run_asr_cache_job(
    *,
    session: str,
    gpu: str,
    rl_repo: str,
    base_repo: str,
    hf_token: str,
    asr_remote_path: str,
) -> None:
    if not ping_session(session, timeout=25):
        restart_kernel(session)
    pkg_zip = SCRATCH / "code_latest.zip"
    package_codebase(pkg_zip)
    upload_file(pkg_zip, "/content/code_latest.zip", session=session, gpu=gpu)

    remote_prefix = str(Path(asr_remote_path).parent).replace("\\", "/")
    out_file = f"/content/{asr_remote_path}"
    metrics_file = f"/content/{remote_prefix}/folds1_4_asr_metrics.json"

    worker_code = _bootstrap_worker_header(
        rl_repo=rl_repo,
        base_repo=base_repo,
        need_audio=True,
        install_rl_pkgs=False,
    ) + f'''
out_path = Path({out_file!r})
metrics_path = Path({metrics_file!r})
out_path.parent.mkdir(parents=True, exist_ok=True)

if rl_syncer.exists({asr_remote_path!r}):
    try:
        dl = Path(
            hf_hub_download(
                repo_id={rl_repo!r},
                filename={asr_remote_path!r},
                repo_type="model",
                token=token,
                local_dir="/content/_hf_resume_asr",
            )
        )
        if dl.exists() and dl.stat().st_size > 0:
            hf_n = sum(1 for _ in open(dl, encoding="utf-8"))
            loc_n = sum(1 for _ in open(out_path, encoding="utf-8")) if out_path.exists() else 0
            if hf_n > loc_n:
                shutil.copy2(dl, out_path)
            print("Resumed partial ASR cache:", max(hf_n, loc_n), "rows", flush=True)
        elif out_path.exists() and out_path.stat().st_size == 0:
            out_path.unlink()
    except Exception as _exc:
        print("WARN ignoring unreadable remote partial ASR:", _exc, flush=True)

env = {{**os.environ, "PYTHONPATH": "/content:" + os.environ.get("PYTHONPATH", ""), "PYTHONUNBUFFERED": "1"}}
env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
cmd = [
    sys.executable, "-u", "/content/scripts/run_asr.py",
    "--split", "train",
    "--root", "/content/data",
    "--ids", "/content/configs/splits/folds_1_to_4_ids.json",
    "--engine", "whisper",
    "--model-id", "kzap201/phowhisper-large-zaloai",
    "--dtype", "float16",
    "--no-context-bias",
    "--batch-size", "4",
    "--nbest-beams", "3",
    "--nbest-max-delta", "0.65",
    "--out", str(out_path),
    "--metrics-out", str(metrics_path),
    "--resume",
]
print("[ASR Task 0] Running:", " ".join(cmd), flush=True)
watcher = DirWatcher(rl_syncer, [(out_path.parent, {remote_prefix!r})], interval_s=300, message_prefix="asr task0")
watcher.start()
rc = 1
try:
    proc = subprocess.Popen(cmd, cwd="/content", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
finally:
    has_rows = out_path.exists() and out_path.stat().st_size > 0
    watcher.stop(final=bool(has_rows))
if rc != 0:
    raise RuntimeError(f"run_asr.py failed with returncode={{rc}}")
n_rows = sum(1 for l in open(out_path, encoding="utf-8") if l.strip())
print("[ASR Task 0] Generated rows:", n_rows, flush=True)
assert n_rows == 996, f"Expected 996 folds_1_to_4 rows, got {{n_rows}}"
assert rl_syncer.push(out_path.parent, {remote_prefix!r}, f"Upload folds_1_to_4 3-beam PhoWhisper cache ({{n_rows}} rows)", raise_on_fail=True)
print("ASR_TASK0_" + "COMPLETE", flush=True)
'''
    worker_py = SCRATCH / "worker_qwen35_asr_cache.py"
    worker_py.write_text(worker_code, encoding="utf-8")
    res = run_command(
        str(worker_py),
        session=session,
        is_file=True,
        timeout=3300,
        max_retries=2,
        env=[f"{ENV_TOKEN}={hf_token}", f"{ENV_REPO}={rl_repo}", "HF_SYNC_PRIVATE=0"],
        stream=True,
        gpu=gpu,
    )
    _ok = any(ln.strip() == "ASR_TASK0_COMPLETE" for ln in (res.stdout or "").splitlines())
    if res.returncode != 0 or not _ok or not _asr_cache_complete(rl_repo, hf_token, asr_remote_path):
        raise RuntimeError(f"ASR Task 0 failed or incomplete remote file {rl_repo}/{asr_remote_path}")


def _asr_cache_complete(repo_id: str, token: str, path_in_repo: str) -> bool:
    s = HFSyncer(repo_id, token=token, private=False, retries=4)
    if not s.exists(path_in_repo):
        return False
    try:
        import json as _json
        import tempfile as _tmp
        from huggingface_hub import hf_hub_download as _dl

        with _tmp.TemporaryDirectory() as td:
            p = Path(_dl(repo_id=repo_id, filename=path_in_repo, repo_type="model", token=token, local_dir=td))
            if p.stat().st_size == 0:
                return False
            rows = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            return len(rows) == 996 and bool(_json.loads(rows[0]).get("id"))
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="rl-g4")
    parser.add_argument("--config", default="configs/qwen35_v2_rl_stage2.yaml")
    parser.add_argument("--gpu", default="G4")
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--override-total-steps", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--prepare-asr-only", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    from huggingface_hub import get_token

    hf_token = os.environ.get(ENV_TOKEN) or get_token() or ""
    rl_repo = resolve_repo_id(args.hf_repo or cfg.get("hf_repo") or DEFAULT_RL_REPO)
    base_repo = cfg.get("base_hf_repo") or DEFAULT_BASE_REPO
    if not hf_token:
        sys.exit("ABORT: HF_TOKEN not found (export HF_TOKEN or run `hf auth login`).")

    syncer = HFSyncer(rl_repo, token=hf_token, private=bool(cfg.get("hf_private", False)), retries=5)
    who = syncer.preflight()
    print(f"[colab_rl] HF preflight OK as '{who}' -> {rl_repo} (base read from {base_repo})", flush=True)

    gpu = args.gpu.upper()
    ensure_session(args.session, gpu=gpu)

    asr_remote_path = cfg.get("asr_cache_hf") or "exp_qwen35_v2_rl/asr_cache/folds1_4_phowhisper_3beam_nbest.jsonl"
    if not args.skip_asr and not _asr_cache_complete(rl_repo, hf_token, asr_remote_path):
        print(f"[colab_rl] Running Task 0: generating {asr_remote_path}...", flush=True)
        run_asr_cache_job(
            session=args.session,
            gpu=gpu,
            rl_repo=rl_repo,
            base_repo=base_repo,
            hf_token=hf_token,
            asr_remote_path=asr_remote_path,
        )
    else:
        print(f"[colab_rl] ASR cache verified on Hub: {rl_repo}/{asr_remote_path}", flush=True)

    if args.prepare_asr_only:
        return 0

    total_steps = int(args.override_total_steps or cfg["optim"]["total_steps"])
    hf_prefix = cfg["hf_prefix"]
    done_marker = f"{hf_prefix}/checkpoints/checkpoint-{total_steps}/adapter_config.json"

    for chunk in range(1, args.max_chunks + 1):
        if syncer.exists(done_marker):
            print(f"[colab_rl] Complete: {rl_repo}/{done_marker}", flush=True)
            return 0
        print(f"\n[colab_rl] === Launching Chunk {chunk}/{args.max_chunks} (target step {total_steps}) ===", flush=True)
        if not ping_session(args.session, timeout=25):
            restart_kernel(args.session)

        pkg_zip = SCRATCH / "code_latest.zip"
        package_codebase(pkg_zip)
        upload_file(pkg_zip, "/content/code_latest.zip", session=args.session, gpu=gpu)
        upload_file(ROOT / args.config, f"/content/{Path(args.config).name}", session=args.session, gpu=gpu)

        step_flag = f", '--override-total-steps', '{args.override_total_steps}'" if args.override_total_steps else ""
        worker_code = _bootstrap_worker_header(
            rl_repo=rl_repo,
            base_repo=base_repo,
            need_audio=False,
            install_rl_pkgs=True,
        ) + f'''
env = {{**os.environ, "PYTHONPATH": "/content:" + os.environ.get("PYTHONPATH", ""), "PYTHONUNBUFFERED": "1"}}
env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
cmd = [
    sys.executable, "-u", "/content/scripts/train_qwen35_rl.py",
    "--config", {('/content/' + Path(args.config).name)!r},
    "--hf-repo", {rl_repo!r}{step_flag},
]
print("[RL Chunk] Running:", " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, cwd="/content", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
assert proc.stdout is not None
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
if rc != 0:
    raise RuntimeError(f"train_qwen35_rl.py exited with returncode={{rc}}")
print("RL_CHUNK_" + "COMPLETE", flush=True)
'''
        worker_py = SCRATCH / "worker_qwen35_rl_g4.py"
        worker_py.write_text(worker_code, encoding="utf-8")
        res = run_command(
            str(worker_py),
            session=args.session,
            is_file=True,
            timeout=3300,
            max_retries=2,
            env=[f"{ENV_TOKEN}={hf_token}", f"{ENV_REPO}={rl_repo}", "HF_SYNC_PRIVATE=0"],
            stream=True,
            gpu=gpu,
        )
        _ok = any(ln.strip() == "RL_CHUNK_COMPLETE" for ln in (res.stdout or "").splitlines())
        if res.returncode != 0 or not _ok:
            raise RuntimeError(f"Chunk {chunk} failed (returncode={res.returncode}, missing RL_CHUNK_COMPLETE sentinel)")
        if syncer.exists(done_marker):
            print(f"[colab_rl] Complete: {rl_repo}/{done_marker}", flush=True)
            return 0

    if not syncer.exists(done_marker):
        raise RuntimeError(f"Reached max_chunks={args.max_chunks} before completion marker {done_marker}")
    print(f"[colab_rl] Complete: {rl_repo}/{done_marker}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
