import gzip
import json
import os
import shutil
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

print("=" * 80, flush=True)
print("STAGE 2: TARGETED DPO TRAINING & EVALUATION ON COLAB GPU", flush=True)
print("Base Model: openbmb/MiniCPM5-2B-SFT (Warm-start from SFT V2 if available)")
print("Dataset: minicpm5_dpo_pairs.jsonl.gz (982 verified contrastive pairs)")
print("Durable checkpoints: every ~10 min -> Hugging Face (AGENTS.md 4.3)", flush=True)
print("=" * 80, flush=True)

# 1. GPU Diagnostics
import torch
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total VRAM: {vram:.2f} GB", flush=True)

# 2. Dependencies
print("\n[1/5] Checking dependencies...", flush=True)
try:
    import torchao
    print("Uninstalling incompatible torchao...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
except ImportError:
    pass

for pkg in ["transformers", "peft", "trl", "datasets", "accelerate", "sentencepiece", "huggingface_hub", "vllm"]:
    try:
        mod = __import__(pkg)
        print(f"  {pkg}: {getattr(mod, '__version__', 'ok')}", flush=True)
    except ImportError:
        print(f"Installing {pkg}...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", pkg], check=True)

# 3. Unpack Code and Sandboxes
print("\n[2/5] Unpacking codebase and test sandboxes...", flush=True)
with zipfile.ZipFile("/content/code_latest.zip", "r") as z:
    z.extractall("/content")
sys.path.insert(0, "/content")

from src.core.hf_sync import DirWatcher, HFSyncer, resolve_repo_id  # noqa: E402

HF_REPO = resolve_repo_id()
if not HF_REPO or not os.environ.get("HF_TOKEN"):
    sys.exit("ABORT: HF_SYNC_REPO / HF_TOKEN not set on Colab kernel; refusing to run without durable checkpoints.")
syncer = HFSyncer(HF_REPO)
print(f"HF preflight OK as '{syncer.preflight()}' -> {HF_REPO}", flush=True)

data_train = Path("/content/data/train")
data_train.mkdir(parents=True, exist_ok=True)

if Path("/content/train_tools_catalog.json").exists():
    shutil.copy2("/content/train_tools_catalog.json", data_train / "train_tools_catalog.json")

env_target = data_train / "train_environment"
env_target.mkdir(parents=True, exist_ok=True)
if not (env_target / "sample_0000").exists() and Path("/content/train_environment.zip").exists():
    with zipfile.ZipFile("/content/train_environment.zip", "r") as z:
        z.extractall(str(env_target))

zt_target = data_train / "zaic_tools"
zt_target.mkdir(parents=True, exist_ok=True)
if not (zt_target / "executor.py").exists() and Path("/content/zaic_tools.zip").exists():
    with zipfile.ZipFile("/content/zaic_tools.zip", "r") as z:
        z.extractall(str(zt_target))
print(f"Sandboxes ready ({len(list(env_target.iterdir()))} environments).", flush=True)

train_gz = Path("/content/train.jsonl.gz")
train_out = data_train / "train.jsonl"
if train_gz.exists() and not train_out.exists():
    with gzip.open(train_gz, "rt", encoding="utf-8") as f_in, open(train_out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            f_out.write(line)

splits_dir = Path("/content/configs/splits")
splits_dir.mkdir(parents=True, exist_ok=True)
if Path("/content/dev_v1.json").exists():
    shutil.copy2("/content/dev_v1.json", splits_dir / "dev_v1.json")

dpo_data = Path("/content/minicpm5_dpo_pairs.jsonl.gz")
assert dpo_data.exists(), "DPO dataset not found at /content/minicpm5_dpo_pairs.jsonl.gz!"

env_vars = os.environ.copy()
env_vars["PYTHONPATH"] = f"/content:{env_vars.get('PYTHONPATH', '')}"

BASE_MODEL = "openbmb/MiniCPM5-2B-SFT"
OUT_DIR = Path("/content/experiments/exp_branchA_dpo_v1")
PREFIX = OUT_DIR.name
METRICS_REMOTE = f"{PREFIX}/eval/metrics.json"

# Check if SFT V2 adapter can be pulled from HF to warm-start DPO
sft_adapter_dir = Path("/content/sft_v2_adapter")
init_adapter_arg = []
if syncer.exists("exp_branchA_sft_v2/adapter/adapter_model.safetensors"):
    print("Found finished SFT V2 adapter on HF! Downloading to warm-start DPO...", flush=True)
    sft_adapter_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = syncer.download_dir("exp_branchA_sft_v2/adapter", sft_adapter_dir)
    adapter_path = downloaded_path if (downloaded_path / "adapter_config.json").exists() else sft_adapter_dir
    print(f"SFT V2 adapter verified at: {adapter_path}", flush=True)
    init_adapter_arg = ["--init-adapter", str(adapter_path)]
else:
    print("SFT V2 adapter not found on HF; training DPO directly on openbmb/MiniCPM5-2B-SFT.", flush=True)

# 4. Check if DPO already finished
if syncer.exists(METRICS_REMOTE):
    print(f"DPO already finished on HF ({METRICS_REMOTE}); downloading and skipping.", flush=True)
    syncer.download_dir(f"{PREFIX}/eval", OUT_DIR.parent)
else:
    print("\n" + "=" * 80, flush=True)
    print(f"[3/5] TRAINING TARGETED DPO: {BASE_MODEL} for 1.0 Epoch on 982 pairs", flush=True)
    print("=" * 80, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0_train = time.time()
    train_cmd = [
        sys.executable, "/content/scripts/train_minicpm5_dpo.py",
        "--base-model", BASE_MODEL,
        "--data", str(dpo_data),
        "--out", str(OUT_DIR),
        "--epochs", "1.0",
        "--batch-size", "4",
        "--grad-accum", "4",
        "--lr", "3.0e-6",
        "--beta", "0.1",
        "--lora-r", "64",
        "--lora-alpha", "128",
        "--max-length", "8192",
        "--hf-repo", HF_REPO,
        "--hf-prefix", PREFIX,
        "--hf-interval-min", "10",
    ] + init_adapter_arg

    # Real-time streaming from training script
    proc = subprocess.Popen(
        train_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env_vars,
        bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"train_minicpm5_dpo.py failed with code {ret}")

    t_train = time.time() - t0_train
    print(f"\nDPO training completed in {t_train / 60:.2f} minutes.", flush=True)

    # 5. Evaluate on 299 Blind Dev Samples with vLLM
    print("\n" + "=" * 80, flush=True)
    print("[4/5] EVALUATING DPO ON 299 SEALED DEV SAMPLES (vLLM Backend)", flush=True)
    print("=" * 80, flush=True)
    preds_dir = OUT_DIR / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_file = preds_dir / "dev_preds.jsonl"
    eval_dir = OUT_DIR / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    agent_cmd = [
        sys.executable, "/content/scripts/run_agent.py",
        "--config", "/content/configs/oracle_minicpm5_2b.yaml",
        "--split", "train",
        "--root", "/content/data",
        "--ids", "/content/configs/splits/dev_v1.json",
        "--backend", "vllm",
        "--model-id", str(OUT_DIR / "merged"),
        "--enable-thinking",
        "--transcripts", "oracle",
        "--out", str(preds_file),
        "--resume",
    ]

    t0_eval = time.time()
    watcher = DirWatcher(syncer, [(preds_dir, f"{PREFIX}/preds")], interval_s=600, message_prefix=f"{PREFIX} preds").start()
    try:
        res_agent = subprocess.run(agent_cmd, env=env_vars)
        if res_agent.returncode != 0:
            print(f"run_agent.py with vLLM returned code {res_agent.returncode}", flush=True)
            print("Retrying evaluation with --backend transformers fallback...", flush=True)
            fallback_cmd = list(agent_cmd)
            idx = fallback_cmd.index("vllm")
            fallback_cmd[idx] = "transformers"
            subprocess.run(fallback_cmd, check=True, env=env_vars)
    finally:
        watcher.stop(final=True)
    t_eval = time.time() - t0_eval

    # Run evaluation harness
    subprocess.run(
        [
            sys.executable, "/content/scripts/run_eval.py",
            "--predictions", str(preds_file),
            "--root", "/content/data",
            "--ids", "/content/configs/splits/dev_v1.json",
            "--out", str(eval_dir),
        ],
        check=True,
        env=env_vars,
    )
    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    syncer.push(eval_dir, f"{PREFIX}/eval", f"{PREFIX} eval metrics", raise_on_fail=True)
    print(f"Dev evaluation finished in {t_eval / 60:.2f} min. Final Score: {metrics.get('final_score', 0):.4f}", flush=True)

# 6. Package Results
print("\n" + "=" * 80, flush=True)
print("[5/5] DPO RESULTS SUMMARY", flush=True)
print("=" * 80, flush=True)
eval_dir = OUT_DIR / "eval"
metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
print(f"Final Score:      {metrics.get('final_score', 0):.4f}")
print(f"State Match Rate: {metrics.get('state_match_rate', 0):.4f}")
print(f"Precision:        {metrics.get('precision', 0):.4f}")
print(f"Recall:           {metrics.get('recall', 0):.4f}")
print(f"F1:               {metrics.get('f1', 0):.4f}")
print("\nBy Category:")
for cat, stats in metrics.get("by_category", {}).items():
    print(f"  {cat:15s}: score={stats.get('score', 0):.4f} (n={stats.get('n', 0)})")

zip_path = Path("/content/dpo_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in eval_dir.glob("*"):
        z.write(f, f"eval/{f.name}")
    if (OUT_DIR / "training_summary.json").exists():
        z.write(OUT_DIR / "training_summary.json", "training_summary.json")

print(f"\nArtifacts packaged to {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB).", flush=True)
print("STAGE 2 DPO EXPERIMENT COMPLETED SUCCESSFULLY!", flush=True)
