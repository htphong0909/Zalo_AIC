from __future__ import annotations

import json
import subprocess
import sys


def test_run_agent_cli_fake_backend(tmp_path):
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps({"dev_ids": ["train_0001", "train_0019"]}), encoding="utf-8")
    out_file = tmp_path / "preds.jsonl"
    trace_file = tmp_path / "traces.jsonl"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_agent.py",
            "--split",
            "train",
            "--root",
            "data",
            "--ids",
            str(ids_file),
            "--backend",
            "fake",
            "--out",
            str(out_file),
            "--trace-out",
            str(trace_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["id"] for r in rows] == ["train_0001", "train_0019"]
    assert all("tool_calls" in r and "state_hash" in r for r in rows)
    traces = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(traces) == 2


def test_run_eval_cli_accepts_ids_list_and_dict(tmp_path):
    ids_file = tmp_path / "random_ids.json"
    ids_file.write_text(json.dumps({"ids": ["train_0001", "train_0019"]}), encoding="utf-8")
    preds_file = tmp_path / "preds.jsonl"
    out_dir = tmp_path / "eval_out"

    proc_agent = subprocess.run(
        [
            sys.executable,
            "scripts/run_agent.py",
            "--split",
            "train",
            "--root",
            "data",
            "--ids",
            str(ids_file),
            "--backend",
            "fake",
            "--out",
            str(preds_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_agent.returncode == 0, proc_agent.stderr

    proc_eval = subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--predictions",
            str(preds_file),
            "--root",
            "data",
            "--ids",
            str(ids_file),
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_eval.returncode == 0, proc_eval.stderr
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n"] == 2


def test_run_agent_cli_with_qwen38_config_and_prompt_style_override(tmp_path):
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps({"ids": ["train_0001"]}), encoding="utf-8")
    out_file = tmp_path / "preds.jsonl"
    trace_file = tmp_path / "traces.jsonl"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_agent.py",
            "--config",
            "configs/oracle_qwen38_4b_distill.yaml",
            "--split",
            "train",
            "--root",
            "data",
            "--ids",
            str(ids_file),
            "--backend",
            "fake",
            "--prompt-style",
            "qwen35",
            "--out",
            str(out_file),
            "--trace-out",
            str(trace_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["id"] == "train_0001"


def test_run_agent_forwards_asr_phonetic_diff(tmp_path):
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps({"ids": ["train_0001"]}), encoding="utf-8")
    transcripts_file = tmp_path / "transcripts.jsonl"
    transcripts_file.write_text(
        json.dumps(
            {
                "id": "train_0001",
                "audio_text": "gọi cho anh Hàng đi",
                "history_texts": {},
                "asr_phonetic_diff": '[Gợi ý đồng âm ASR: "Hàng" <-> "Hàn"]',
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    from scripts.run_agent import _load_transcripts_map, _resolve_sample_transcript_inputs
    from src.agent.react_loop import run_react_loop
    from src.agent.llm_client import SequenceFakeBackend
    from src.core.dataset import load_catalog, load_split
    from src.env.safe_env import SafeEnv

    tmap = _load_transcripts_map(transcripts_file)
    curr_text, hist_texts, diff_hint = _resolve_sample_transcript_inputs(tmap, "train_0001")
    sample = next(s for s in load_split("train", "data") if s.id == "train_0001")
    fake_llm = SequenceFakeBackend(["Không thể xử lý."])
    run_react_loop(
        sample,
        SafeEnv("train", sample.id, "data"),
        load_catalog("train", "data"),
        fake_llm,
        current_text=curr_text,
        history_texts=hist_texts,
        asr_phonetic_diff=diff_hint,
    )
    user_msg = fake_llm.calls_log[0]["messages"][-1]["content"]
    assert '[Gợi ý đồng âm ASR: "Hàng" <-> "Hàn"]' in user_msg


