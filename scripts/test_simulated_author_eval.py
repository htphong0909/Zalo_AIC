#!/usr/bin/env python3
"""Simulate author test evaluation on the simulated test dataset.

Runs the exact end-to-end pipeline against data/simulated_eval/simulation_test,
validates the output with zaic_tools, checks latency limits, and runs verify_submission.py.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
SUB_DIR = ROOT / "submission" / "qwen35_4b_sft_v2_e2e"
DATA_ROOT = ROOT / "data" / "simulated_eval"
SPLIT = "simulation_test"
RESULT_DIR = DATA_ROOT / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_JSONL = RESULT_DIR / "submission.jsonl"
JUPYTER_JSONL = RESULT_DIR / "jupyter_submission.jsonl"

sys.path.insert(0, str(SUB_DIR))
import agent
from agent import (
    DEFAULT_CONFIG_PATH,
    discover_split,
    import_zaic_tools,
    load_models,
    run_inference,
    seed_everything,
)
from src.core.dataset import load_catalog, load_split


class RealisticASREngine:
    """ASR engine returning transcripts matching the audio samples."""

    def __init__(self):
        # Known realistic transcripts for the 6 chosen samples
        self.transcripts = {
            "sim_0001": "thời tiết ngày mai thế nào",
            "sim_0002": "thêm bài hát vào danh sách phát giúp tôi",
            "sim_0003": "xác nhận gửi luôn đi",
            "sim_0004": "tắt đèn phòng khách",
            "sim_0005": "xem lịch hôm nay có cuộc hẹn nào không",
            "sim_0006": "đặt báo thức lúc 5 giờ 30 phút sáng mai",
        }

    def transcribe(self, audio_p, **kw):
        t0 = time.perf_counter()
        stem = Path(audio_p).stem
        txt = self.transcripts.get(stem, f"yêu cầu {stem}")
        return SimpleNamespace(
            text=txt,
            raw_text=txt,
            latency_s=time.perf_counter() - t0,
            nbest=[txt],
            nbest_scores=[-0.05],
        )

    def transcribe_batch(self, paths, **kw):
        return [self.transcribe(p, **kw) for p in paths]


class RealisticLLMBackend:
    """LLM backend executing author-compliant actions across the 6 simulated samples."""

    def __init__(self):
        self.call_counts: dict[str, int] = {}

    def generate(self, messages: list[dict], tools: list[dict] | None = None, **kw) -> str:
        user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        prompt = "\n".join(str(x) for x in user_msgs)
        tool_results = "\n".join(str(m.get("content", "")) for m in tool_msgs)

        # sim_0001: Out of scope (weather requested but weather app might not have or out of scope)
        if "thời tiết" in prompt:
            # Check if weather tool available
            avail = {((t.get("function") or t).get("name") or "") for t in (tools or [])}
            if "weather.get_current" in avail:
                if not tool_msgs:
                    return '<tool_call><function=weather.get_current>{}</function></tool_call>'
                return 'Thời tiết hiện tại tốt.'
            return 'Yêu cầu ngoài phạm vi các ứng dụng đã cài đặt.'

        # sim_0002: Ask missing info
        if "danh sách phát" in prompt:
            return '<tool_call><function=sys.ask_missing_info>{"params": ["playlist_id", "track_id"], "tool_name": "music.add_to_playlist"}</function></tool_call>'

        # sim_0003: History follow-up confirmation (chat / sms)
        if "xác nhận gửi" in prompt:
            avail = {((t.get("function") or t).get("name") or "") for t in (tools or [])}
            if "chat_ott.send_message" in avail:
                return '<tool_call><function=chat_ott.send_message>{"chat_id": "chat_001", "message": "Đã xác nhận"}</function></tool_call>'
            if "sms.send" in avail:
                return '<tool_call><function=sms.send>{"phone_number": "0901234567", "message": "Đã xác nhận"}</function></tool_call>'
            return 'Đã gửi xác nhận.'

        # sim_0004: Smart home light set
        if "tắt đèn phòng khách" in prompt:
            if not tool_msgs:
                return '<tool_call><function=smart_home.light_set>{"device_id": "light_living_room", "power": "off"}</function></tool_call>'
            return 'Đã tắt đèn phòng khách cho bạn.'

        # sim_0005: Calendar query
        if "xem lịch" in prompt or "cuộc hẹn" in prompt:
            if not tool_msgs:
                return '<tool_call><function=calendar.list_events>{"start_date": "2026-11-16T00:00:00+07:00", "end_date": "2026-11-16T23:59:59+07:00"}</function></tool_call>'
            return 'Bạn không có cuộc hẹn nào hôm nay.'

        # sim_0006: Alarm list & create
        if "báo thức" in prompt:
            if not tool_msgs:
                return '<tool_call><function=alarm.list>{}</function></tool_call>'
            if "alarm.list" in tool_results and "alarm.create" not in str(messages):
                return '<tool_call><function=alarm.create>{"time": "2026-11-17T05:30:00+07:00"}</function></tool_call>'
            return 'Đã đặt báo thức lúc 5h30 sáng mai.'

        return 'Tôi có thể giúp gì cho bạn?'


def test_simulation():
    print("=" * 80)
    print("RUNNING AUTHOR-COMPLIANT SIMULATED EVALUATION TEST")
    print(f"Data Root: {DATA_ROOT}")
    print(f"Target Split: {SPLIT}")
    print("=" * 80)

    # 1. Discover split dynamically
    disc_split = discover_split(DATA_ROOT)
    print(f"[1/6] Discovered split: {disc_split}")
    assert disc_split == SPLIT, f"Expected {SPLIT}, got {disc_split}"

    # 2. Import zaic_tools directly from split
    zt = import_zaic_tools(DATA_ROOT, disc_split)
    print(f"[2/6] Imported zaic_tools from: {zt.__file__}")

    # 3. Load catalog and samples
    cat = load_catalog(disc_split, DATA_ROOT)
    samples = load_split(disc_split, DATA_ROOT)
    tools_cnt = len(cat.get("tools", [])) if isinstance(cat, dict) else len(cat)
    print(f"[3/6] Loaded catalog ({tools_cnt} tools) and {len(samples)} samples:")
    for s in samples:
        print(f"   - Sample ID: {s.id}, Audio: {s.audio}")

    # 4. Run CLI inference
    print("\n[4/6] Executing CLI inference runner (predict.py flow)...")
    t0 = time.perf_counter()
    rows_cli = run_inference(
        data_root=DATA_ROOT,
        output_path=SUBMISSION_JSONL,
        config_path=DEFAULT_CONFIG_PATH,
        split_override=disc_split,
        asr_engine=RealisticASREngine(),
        llm_backend=RealisticLLMBackend(),
        strict=True,
    )
    cli_ms = (time.perf_counter() - t0) * 1000.0
    print(f"CLI inference finished in {cli_ms:.2f} ms ({len(rows_cli)} samples)")

    # 5. Run Jupyter notebook simulation (parity check)
    print("\n[5/6] Executing Jupyter Notebook runner (predict_notebook.ipynb flow)...")
    nb = json.loads((SUB_DIR / "predict_notebook.ipynb").read_text(encoding="utf-8"))
    os.environ["DATA_ROOT"] = str(DATA_ROOT)
    os.environ["OUTPUT_PATH"] = str(JUPYTER_JSONL)
    os.environ["CONFIG_PATH"] = str(DEFAULT_CONFIG_PATH)
    os.environ["STRICT_INFERENCE"] = "1"
    os.environ["DATA_SPLIT"] = disc_split

    ns = {
        "__name__": "__main__",
        "_INJECTED_ASR_ENGINE": RealisticASREngine(),
        "_INJECTED_LLM_BACKEND": RealisticLLMBackend(),
    }
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            exec(compile(code, f"cell_{idx+1}", "exec"), ns)

    rows_nb = [json.loads(l) for l in JUPYTER_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Notebook inference finished: {len(rows_nb)} samples")

    # Verify parity
    assert len(rows_cli) == len(rows_nb), f"Count mismatch: {len(rows_cli)} vs {len(rows_nb)}"
    for r_cli, r_nb in zip(rows_cli, rows_nb):
        assert r_cli["id"] == r_nb["id"], f"ID mismatch: {r_cli['id']} vs {r_nb['id']}"
        assert r_cli["tool_calls"] == r_nb["tool_calls"], f"Tool calls mismatch on {r_cli['id']}"
        assert r_cli["state_hash"] == r_nb["state_hash"], f"State hash mismatch on {r_cli['id']}"
        assert r_cli["time"] <= 30000, f"Latency violation: {r_cli['time']} ms > 30000 ms"
        print(f"   Sample {r_cli['id']} PARITY 100%: state_hash={r_cli['state_hash'][:16]}..., calls={len(r_cli['tool_calls'])}, time={r_cli['time']}ms")

    # 6. Run official verification with zaic_tools --validate
    print("\n[6/6] Running official zaic_tools validation on both JSONL files...")
    for pred_path in (SUBMISSION_JSONL, JUPYTER_JSONL):
        v_res = subprocess.run(
            [
                sys.executable,
                "-m",
                "zaic_tools",
                "--split",
                disc_split,
                "--root",
                str(DATA_ROOT),
                "--validate",
                str(pred_path),
            ],
            cwd=str(DATA_ROOT / disc_split),
            capture_output=True,
            text=True,
        )
        print(f"zaic_tools validation on {pred_path.name}:")
        print(v_res.stdout.strip())
        assert v_res.returncode == 0, v_res.stderr
        assert "0 schema errors" in v_res.stdout, f"Schema error found: {v_res.stdout}"

    # Also run verify_submission.py
    print("\nRunning scripts/verify_submission.py on simulation predictions...")
    vproc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_submission.py"),
            "--submission-dir",
            str(SUB_DIR),
            "--predictions",
            str(SUBMISSION_JSONL),
            "--jupyter-predictions",
            str(JUPYTER_JSONL),
            "--data-root",
            str(DATA_ROOT),
            "--split",
            disc_split,
        ],
        capture_output=True,
        text=True,
    )
    print(vproc.stdout.strip())
    if vproc.returncode != 0:
        print(vproc.stderr, file=sys.stderr)
    assert vproc.returncode == 0, "verify_submission.py failed on simulated test!"

    print("\n" + "=" * 80)
    print("ALL SIMULATION AND COMPLIANCE CHECKS PASSED WITH ZERO ERRORS!")
    print("=" * 80)


if __name__ == "__main__":
    test_simulation()
