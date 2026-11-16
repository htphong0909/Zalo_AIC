#!/usr/bin/env python3
"""Verify Docker submission compliance against every requirement in Docker_Guide_ZAC2026_LifeMate.pdf."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUBMISSION_DIR = ROOT / "submission" / "qwen35_4b_sft_v2_e2e"

REQUIRED_FILES = [
    "Dockerfile",
    "README.md",
    "requirements.txt",
    "predict.py",
    "predict.sh",
    "start_jupyter.sh",
    "predict_notebook.ipynb",
    "agent.py",
    "prepare_models.py",
    "configs/e2e_qwen35_4b_sft_v2.yaml",
    "training_code/README.md",
    "training_code/configs/qwen35_sft_v2.yaml",
    "training_code/scripts/build_qwen38_sft_data.py",
    "training_code/scripts/train_qwen38_sft.py",
]

FORBIDDEN_PATHS = [
    "zaic_tools",
    "data",
    "train_environment",
    "public_test_environment",
    "private_test_environment",
    "src/eval",
    "src/data_gen",
]


def check_static_bundle(sub_dir: Path) -> list[str]:
    errors: list[str] = []
    for rel in REQUIRED_FILES:
        if not (sub_dir / rel).is_file():
            errors.append(f"Missing required submission file: {rel}")

    for bad in FORBIDDEN_PATHS:
        if (sub_dir / bad).exists():
            errors.append(f"Forbidden path inside runtime submission dir: {bad}")

    # Enforce LF line endings on shell scripts & Dockerfile
    for fn in ("predict.sh", "start_jupyter.sh", "docker_build_and_verify.sh", "Dockerfile"):
        p = sub_dir / fn
        if p.is_file() and b"\r\n" in p.read_bytes():
            errors.append(f"CRLF (\\r\\n) line endings found in {fn}; Linux Docker requires LF (\\n).")

    # Check Dockerfile mandates CUDA 13.2
    df = sub_dir / "Dockerfile"
    if df.is_file():
        df_txt = df.read_text(encoding="utf-8")
        if "nvidia/cuda:13.2" not in df_txt:
            errors.append("Dockerfile FROM image must use CUDA 13.2 per guide pages 4 & 7.")
        if "HF_HUB_OFFLINE=1" not in df_txt or "TRANSFORMERS_OFFLINE=1" not in df_txt:
            errors.append("Dockerfile missing offline env guards (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1).")

    # Check start_jupyter.sh port 9777 and password/token zac2026
    sj = sub_dir / "start_jupyter.sh"
    if sj.is_file():
        sj_txt = sj.read_text(encoding="utf-8")
        if "9777" not in sj_txt or "zac2026" not in sj_txt:
            errors.append("start_jupyter.sh missing port 9777 or zac2026 auth.")

    # Check runtime Python code never reads _environment or calls get_state()
    runtime_py = [sub_dir / "agent.py", sub_dir / "predict.py", *(sub_dir / "src").rglob("*.py")]
    for py in runtime_py:
        if not py.is_file():
            continue
        txt = py.read_text(encoding="utf-8")
        tree = ast.parse(txt, filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "get_state":
                errors.append(f"Illegal env.get_state() reference in runtime file {py.relative_to(sub_dir)}")
            if isinstance(node, ast.ImportFrom) and node.module and (
                "zaic_tools" in node.module and py.name not in ("agent.py", "safe_env.py")
            ):
                errors.append(f"Unexpected static zaic_tools import in {py.relative_to(sub_dir)}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "state.json" in node.value and py.name != "safe_env.py":
                    errors.append(f"Forbidden state.json reference in {py.relative_to(sub_dir)}")

    # Check predict_notebook.ipynb structure (4 valid cells)
    nb_path = sub_dir / "predict_notebook.ipynb"
    if nb_path.is_file():
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = nb.get("cells") or []
        if len(cells) < 3:
            errors.append(f"predict_notebook.ipynb must have >= 3 cells (has {len(cells)})")
        joined = "\n".join("".join(c.get("source") or []) for c in cells)
        for token in ("seed_everything", "load_models", "discover_split", "infer_sample", "jupyter_submission.jsonl"):
            if token not in joined:
                errors.append(f"predict_notebook.ipynb missing expected step token: {token}")

    return errors


def check_predictions_jsonl(
    sub_jsonl: Path,
    *,
    data_root: Path,
    split: str,
    jupyter_jsonl: Path | None = None,
    max_latency_ms: float = 30000.0,
) -> tuple[list[str], list[str], list[str]]:
    errors: list[str] = []
    if not sub_jsonl.is_file():
        return [f"Missing predictions file: {sub_jsonl}"], [], []

    samples_path = data_root / split / f"{split}.jsonl"
    expected_ids = [json.loads(l)["id"] for l in samples_path.open(encoding="utf-8") if l.strip()]

    def _load(path: Path) -> list[dict]:
        return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    rows = _load(sub_jsonl)
    row_map = {r["id"]: r for r in rows}
    if len(rows) != len(row_map):
        errors.append(f"Duplicate IDs in {sub_jsonl}: {len(rows)} lines vs {len(row_map)} unique")

    # Import split's zaic_tools to run schema_validate + catalog checks cleanly
    split_dir = str((data_root / split).resolve())
    if split_dir not in sys.path:
        sys.path.insert(0, split_dir)
    from zaic_tools import ToolEnv, catalog_path, schema_validate

    cat = json.load(open(catalog_path(split, str(data_root)), encoding="utf-8"))
    for r in rows:
        for req_k in ("id", "tool_calls", "state_hash", "time"):
            if req_k not in r:
                errors.append(f"{sub_jsonl.name} sample {r.get('id')} missing key '{req_k}'")
        t_val = r.get("time", 0)
        if not isinstance(t_val, (int, float)) or t_val < 0 or t_val > max_latency_ms:
            errors.append(f"{sub_jsonl.name} sample {r.get('id')} invalid/slow time={t_val}ms (> {max_latency_ms}ms)")
        allowed = ToolEnv(split, r["id"], str(data_root)).allowed
        for i, res in enumerate(schema_validate(r.get("tool_calls") or [], cat, allowed)):
            if not res["ok"]:
                errors.append(f"{sub_jsonl.name} {r['id']} call {i} {res['name']}: {res['error']}")

    if jupyter_jsonl is not None:
        if not jupyter_jsonl.is_file():
            errors.append(f"Missing jupyter predictions file: {jupyter_jsonl}")
        else:
            j_rows = _load(jupyter_jsonl)
            j_map = {r["id"]: r for r in j_rows}
            if set(row_map) != set(j_map):
                errors.append(f"ID mismatch CLI ({len(row_map)}) vs Jupyter ({len(j_map)})")
            for sid in row_map:
                if sid not in j_map:
                    continue
                if row_map[sid]["tool_calls"] != j_map[sid]["tool_calls"]:
                    errors.append(f"tool_calls mismatch on {sid} between CLI and Jupyter")
                    break
                if row_map[sid]["state_hash"] != j_map[sid]["state_hash"]:
                    errors.append(f"state_hash mismatch on {sid} between CLI and Jupyter")
                    break
            for r in j_rows:
                t_val = r.get("time", 0)
                if not isinstance(t_val, (int, float)) or t_val < 0 or t_val > max_latency_ms:
                    errors.append(f"{jupyter_jsonl.name} sample {r.get('id')} invalid/slow time={t_val}ms")

    return errors, expected_ids, list(row_map)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", default=str(DEFAULT_SUBMISSION_DIR))
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--jupyter-predictions", default=None)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--split", default="public_test")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    sub_dir = Path(args.submission_dir)
    errs = check_static_bundle(sub_dir)
    if args.predictions:
        p_errs, expected_ids, got_ids = check_predictions_jsonl(
            Path(args.predictions),
            data_root=Path(args.data_root),
            split=args.split,
            jupyter_jsonl=Path(args.jupyter_predictions) if args.jupyter_predictions else None,
        )
        errs.extend(p_errs)
        if not args.allow_partial and expected_ids and set(got_ids) != set(expected_ids):
            errs.append(f"Incomplete submission: {len(got_ids)}/{len(expected_ids)} samples")

    if errs:
        for e in errs:
            print(f"[FAIL] {e}", file=sys.stderr)
        return 1
    print(f"[PASS] Submission compliance checks passed for {sub_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
