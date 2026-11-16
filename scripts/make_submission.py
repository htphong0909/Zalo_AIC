#!/usr/bin/env python3
"""Assemble and synchronize the clean Docker submission bundle for Qwen3.5-4B SFT V2 E2E."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUBMISSION_DIR = ROOT / "submission" / "qwen35_4b_sft_v2_e2e"

RUNTIME_SRC_FILES = [
    "src/__init__.py",
    "src/core/__init__.py",
    "src/core/config.py",
    "src/core/dataset.py",
    "src/core/schemas.py",
    "src/env/__init__.py",
    "src/env/safe_env.py",
    "src/asr/__init__.py",
    "src/asr/audio.py",
    "src/asr/aura_nbest.py",
    "src/asr/base.py",
    "src/asr/itn.py",
    "src/asr/whisper_engine.py",
    "src/agent/__init__.py",
    "src/agent/boundary_verifier.py",
    "src/agent/history.py",
    "src/agent/llm_client.py",
    "src/agent/minicpm5_parser.py",
    "src/agent/prompts.py",
    "src/agent/react_loop.py",
    "src/agent/tool_caller.py",
    "src/agent/tool_selector.py",
]

TRAINING_SCRIPT_FILES = [
    "scripts/build_qwen38_sft_data.py",
    "scripts/train_qwen38_sft.py",
    "scripts/prune_qwen35_text_only.py",
    "scripts/train_phowhisper_lora.py",
    "scripts/run_asr.py",
    "scripts/run_agent.py",
    "scripts/run_eval.py",
    "scripts/count_params.py",
]

TRAINING_CONFIG_FILES = [
    "configs/qwen35_sft_v2.yaml",
    "configs/splits/dev_v1.json",
    "scripts/scratch/fold_0_ids.json",
]

TRAINING_SRC_DIRS = [
    "src/core",
    "src/env",
    "src/asr",
    "src/agent",
    "src/data_gen",
    "src/eval",
]


def _copy_lf(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    dst.write_text(text, encoding="utf-8", newline="\n")


def build_predict_notebook() -> nbf.NotebookNode:
    """Create the 4-cell predict_notebook.ipynb matching PDF Appendix Pages 15-18."""
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.12",
    }

    cell1 = """# Bước 1: Thiết lập biến môi trường offline, fix random seed và import module suy luận
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

CODE_ROOT = Path("/code") if Path("/code/agent.py").exists() else Path.cwd()
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from agent import (
    DEFAULT_CONFIG_PATH,
    build_audio_capped_chunks,
    discover_split,
    import_zaic_tools,
    infer_batch,
    infer_sample,
    load_models,
    seed_everything,
)
from src.core.dataset import load_catalog, load_split

seed_everything(42)
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/result/jupyter_submission.jsonl"))
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
print(f"CODE_ROOT={CODE_ROOT} | DATA_ROOT={DATA_ROOT} | OUTPUT_PATH={OUTPUT_PATH}")"""

    cell2 = """# Bước 2: Load offline models (Qwen3.5-4B SFT V2 + PhoWhisper-Large-ZaloAI)
# Hỗ trợ inject model giả lập qua MODULE hook khi chạy unit test; mặc định load model offline thật.
t_load_start = time.perf_counter()
_inj_asr = globals().get("_INJECTED_ASR_ENGINE")
_inj_llm = globals().get("_INJECTED_LLM_BACKEND")
cfg, raw_cfg, asr_engine, llm_backend = load_models(
    CONFIG_PATH,
    asr_engine=_inj_asr,
    llm_backend=_inj_llm,
)
load_ms = (time.perf_counter() - t_load_start) * 1000.0
print(f"load model: {load_ms:.2f} ms")"""

    cell3 = """# Bước 3: Khám phá động split trong /data, import zaic_tools từ split và nạp catalog + samples
SPLIT = discover_split(DATA_ROOT, split_override=os.environ.get("DATA_SPLIT") or None)
zaic_tools = import_zaic_tools(DATA_ROOT, SPLIT)
catalog = load_catalog(SPLIT, DATA_ROOT)
test_cases = load_split(SPLIT, DATA_ROOT)

_limit_env = os.environ.get("SAMPLE_LIMIT")
if _limit_env:
    test_cases = test_cases[: int(_limit_env)]

print(f"SPLIT={SPLIT} | total_samples={len(test_cases)} | zaic_tools={zaic_tools.__file__}")"""

    cell4 = """# Bước 4: Chạy suy luận tối ưu batch theo audio budget (giữ nguyên thứ tự), đo time từ trước bước đọc audio, ghi /result/jupyter_submission.jsonl
seed_everything(cfg.seed)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
_strict_mode = os.environ.get("STRICT_INFERENCE", "0") == "1"
_batch_size = max(1, int(getattr(cfg.agent, "batch_size", 1)))

results = []
t_pred_start = time.perf_counter()
with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\\n") as fout, tqdm(total=len(test_cases), desc=f"Infer {SPLIT}") as pbar:
    if _batch_size == 1:
        for item in test_cases:
            row = infer_sample(
                item,
                split=SPLIT,
                data_root=DATA_ROOT,
                catalog=catalog,
                asr_engine=asr_engine,
                llm_backend=llm_backend,
                cfg=cfg,
                raw_cfg=raw_cfg,
                strict=_strict_mode,
            )
            fout.write(json.dumps(row, ensure_ascii=False) + "\\n")
            fout.flush()
            results.append(row)
            pbar.update(1)
    else:
        for chunk in build_audio_capped_chunks(test_cases, max_batch_size=_batch_size, max_chunk_audios=8):
            chunk_rows = infer_batch(
                chunk,
                split=SPLIT,
                data_root=DATA_ROOT,
                catalog=catalog,
                asr_engine=asr_engine,
                llm_backend=llm_backend,
                cfg=cfg,
                raw_cfg=raw_cfg,
                strict=_strict_mode,
            )
            for row in chunk_rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\\n")
            fout.flush()
            results.extend(chunk_rows)
            pbar.update(len(chunk_rows))

pred_ms = (time.perf_counter() - t_pred_start) * 1000.0
avg_ms = pred_ms / max(1, len(results))
print(f"predict: {len(results)} mẫu, {pred_ms:.2f} ms, trung bình {avg_ms:.2f} ms/mẫu")
print(f"Saved -> {OUTPUT_PATH}")"""

    nb.cells = [
        nbf.v4.new_code_cell(cell1),
        nbf.v4.new_code_cell(cell2),
        nbf.v4.new_code_cell(cell3),
        nbf.v4.new_code_cell(cell4),
    ]
    return nb


def assemble_submission(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sync clean inference runtime src/ (never copies eval/, data_gen/, or zaic_tools/)
    dst_src = out_dir / "src"
    if dst_src.exists():
        shutil.rmtree(dst_src)
    for rel in RUNTIME_SRC_FILES:
        src = ROOT / rel
        if not src.is_file():
            raise FileNotFoundError(f"Missing runtime source file: {src}")
        _copy_lf(src, out_dir / rel)

    # 2. Sync complete training_code/ (scripts, configs, full src excluding __pycache__ & private eval caches)
    t_dir = out_dir / "training_code"
    for subdir in ("scripts", "configs", "src"):
        if (t_dir / subdir).exists():
            shutil.rmtree(t_dir / subdir)

    for rel in TRAINING_SCRIPT_FILES:
        src = ROOT / rel
        if src.is_file():
            _copy_lf(src, t_dir / rel)

    _copy_lf(ROOT / "configs" / "qwen35_sft_v2.yaml", t_dir / "configs" / "qwen35_sft_v2.yaml")
    _copy_lf(ROOT / "configs" / "splits" / "dev_v1.json", t_dir / "configs" / "splits" / "dev_v1.json")
    f0 = ROOT / "scripts" / "scratch" / "fold_0_ids.json"
    if f0.is_file():
        _copy_lf(f0, t_dir / "configs" / "splits" / "fold_0_ids.json")

    _copy_lf(ROOT / "src" / "__init__.py", t_dir / "src" / "__init__.py")
    for src_sub in TRAINING_SRC_DIRS:
        for py in sorted((ROOT / src_sub).glob("*.py")):
            _copy_lf(py, t_dir / src_sub / py.name)

    # 3. Generate valid 4-cell predict_notebook.ipynb
    nb = build_predict_notebook()
    nb_path = out_dir / "predict_notebook.ipynb"
    with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
        nbf.write(nb, f)

    # 4. Normalize LF endings on top-level shell/Docker/Python files in out_dir
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and (p.suffix in {".py", ".sh", ".yaml", ".yml", ".md", ".txt"} or p.name == "Dockerfile"):
            _copy_lf(p, p)

    print(f"[make_submission] Staged clean submission bundle at {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_SUBMISSION_DIR))
    args = parser.parse_args()
    assemble_submission(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
