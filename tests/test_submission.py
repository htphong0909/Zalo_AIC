from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SUB_DIR = ROOT / "submission" / "qwen35_4b_sft_v2_e2e"


class _MockASREngine:
    def __init__(self):
        self.calls: list[dict] = []

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        extra_context: str | None = None,
        num_beams_nbest: int = 1,
        diverse_nbest: bool = True,
    ):
        self.calls.append(
            {
                "mode": "single",
                "audio_path": str(audio_path),
                "extra_context": extra_context,
                "num_beams_nbest": num_beams_nbest,
                "diverse_nbest": diverse_nbest,
            }
        )
        name = Path(audio_path).stem
        if num_beams_nbest > 1:
            return SimpleNamespace(
                text=f"bật đèn phòng khách giúp tôi {name}",
                raw_text=f"bật đèn phòng khách giúp tôi {name}",
                latency_s=0.01,
                nbest=[f"bật đèn phòng khách giúp tôi {name}", f"bật đèn phòng ngủ giúp tôi {name}"],
                nbest_scores=[-0.1, -0.25],
            )
        return SimpleNamespace(
            text=f"lịch sử {name}",
            raw_text=f"lịch sử {name}",
            latency_s=0.01,
            nbest=[],
            nbest_scores=[],
        )

    def transcribe_batch(
        self,
        audio_paths: list[str | Path],
        *,
        batch_size: int = 8,
        extra_contexts: list[str | None] | None = None,
        num_beams_nbest: int = 1,
        diverse_nbest: bool = True,
    ):
        self.calls.append(
            {
                "mode": "batch",
                "count": len(audio_paths),
                "num_beams_nbest": num_beams_nbest,
            }
        )
        return [
            SimpleNamespace(
                text=f"lịch sử {Path(p).stem}",
                raw_text=f"lịch sử {Path(p).stem}",
                latency_s=0.01,
                nbest=[],
                nbest_scores=[],
            )
            for p in audio_paths
        ]


class _ScriptedLLMBackend:
    """Deterministic LLM stub covering action, ask_missing_info, and OOS behaviors."""

    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        force_tool_schema: bool = False,
    ) -> str:
        user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        prompt_text = "\n".join(str(x) for x in user_msgs)
        available = {((t.get("function") or t).get("name") or "") for t in (tools or [])}

        # Case 1: if alarm.list is installed and no tool result yet, inspect alarms first then stop
        if "public_0003" in prompt_text and "alarm.list" in available:
            if not tool_msgs:
                return "<tool_call>\n<function=alarm.list>\n</function>\n</tool_call>"
            return "Đã kiểm tra danh sách báo thức của bạn."

        # Case 2: if music.add_to_playlist is NOT present, try ask_missing_info or OOS
        if "public_0002" in prompt_text and "weather.get_current" in available:
            if not tool_msgs:
                return (
                    "<tool_call>\n<function=weather.get_current>\n"
                    "</function>\n</tool_call>"
                )
            return "Thời tiết hiện tại đã được cập nhật."

        # Case 3: default clean OOS []
        return "Yêu cầu này nằm ngoài phạm vi hỗ trợ của các ứng dụng hiện có."


def test_static_submission_compliance():
    sys.path.insert(0, str(ROOT / "scripts"))
    from verify_submission import check_static_bundle

    errors = check_static_bundle(SUB_DIR)
    assert not errors, f"Static submission errors: {errors}"

    cfg = yaml.safe_load((SUB_DIR / "configs" / "e2e_qwen35_4b_sft_v2.yaml").read_text(encoding="utf-8"))
    assert cfg["seed"] == 42
    assert cfg["llm"]["max_model_len"] == 16384
    assert cfg["llm"]["enable_thinking"] is False
    assert cfg["llm"]["guided_decoding"] is True
    assert cfg["agent"]["prompt_style"] == "qwen35"
    assert all(cfg["agent"]["guardrails"].values())


def test_dynamic_split_discovery_and_isolated_zaic_tools(tmp_path, monkeypatch):
    # Prepare isolated split mount simulating BTC /data/private_test/
    custom_split = tmp_path / "private_test"
    zt_dir = custom_split / "zaic_tools"
    zt_dir.mkdir(parents=True)
    (zt_dir / "__init__.py").write_text(
        "class ToolEnv:\n"
        "    def __init__(self, split, sid, root='.'):\n"
        "        self.split, self.sid = split, sid\n",
        encoding="utf-8",
    )
    (custom_split / "private_test_tools_catalog.json").write_text(json.dumps({"apps": []}), encoding="utf-8")
    (custom_split / "private_test.jsonl").write_text("", encoding="utf-8")

    monkeypatch.delenv("DATA_SPLIT", raising=False)
    sys.path.insert(0, str(SUB_DIR))
    import agent as sub_agent

    detected = sub_agent.discover_split(tmp_path)
    assert detected == "private_test"
    mod = sub_agent.import_zaic_tools(tmp_path, detected)
    assert Path(mod.__file__).resolve().is_relative_to(custom_split.resolve())


def test_cli_vs_notebook_determinism_and_public_test_validation(tmp_path, monkeypatch):
    # Pick 6 real public_test samples including ones WITH multi-turn history and audio
    pub_jsonl = ROOT / "data" / "public_test" / "public_test.jsonl"
    raw_rows = [json.loads(l) for l in pub_jsonl.open(encoding="utf-8") if l.strip()]
    with_hist = [r for r in raw_rows if len([h for h in r.get("history", []) if h.get("role") == "user" and h.get("audio")]) >= 2][:3]
    no_hist = [r for r in raw_rows if not r.get("history")][:3]
    selected = no_hist + with_hist
    assert len(selected) == 6

    # Build isolated mounted DATA_ROOT containing ONLY public_test copy (symlinks/junction avoided by copying minimal slice)
    data_mount = tmp_path / "data"
    pub_dst = data_mount / "public_test"
    shutil.copytree(ROOT / "data" / "public_test" / "zaic_tools", pub_dst / "zaic_tools")
    shutil.copy2(ROOT / "data" / "public_test" / "public_test_tools_catalog.json", pub_dst / "public_test_tools_catalog.json")
    (pub_dst / "public_test.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in selected) + "\n",
        encoding="utf-8",
    )
    for r in selected:
        sid = r["id"]
        shutil.copytree(
            ROOT / "data" / "public_test" / "public_test_environment" / sid,
            pub_dst / "public_test_environment" / sid,
        )
        shutil.copytree(ROOT / "data" / "public_test" / "audio" / sid, pub_dst / "audio" / sid)

    sys.path.insert(0, str(SUB_DIR))
    import agent as sub_agent

    # Run 1 (CLI run_inference)
    out1 = tmp_path / "result1" / "submission.jsonl"
    asr1 = _MockASREngine()
    rows1 = sub_agent.run_inference(
        data_root=data_mount,
        output_path=out1,
        config_path=SUB_DIR / "configs" / "e2e_qwen35_4b_sft_v2.yaml",
        asr_engine=asr1,
        llm_backend=_ScriptedLLMBackend(),
        strict=True,
    )

    # Run 2 (CLI repeated determinism check)
    out2 = tmp_path / "result2" / "submission.jsonl"
    rows2 = sub_agent.run_inference(
        data_root=data_mount,
        output_path=out2,
        config_path=SUB_DIR / "configs" / "e2e_qwen35_4b_sft_v2.yaml",
        asr_engine=_MockASREngine(),
        llm_backend=_ScriptedLLMBackend(),
        strict=True,
    )

    # Run 3: Execute every cell of predict_notebook.ipynb in sequence inside an isolated namespace!
    nb = json.loads((SUB_DIR / "predict_notebook.ipynb").read_text(encoding="utf-8"))
    nb_out = tmp_path / "result_nb" / "jupyter_submission.jsonl"
    monkeypatch.setenv("DATA_ROOT", str(data_mount))
    monkeypatch.setenv("OUTPUT_PATH", str(nb_out))
    monkeypatch.setenv("CONFIG_PATH", str(SUB_DIR / "configs" / "e2e_qwen35_4b_sft_v2.yaml"))
    monkeypatch.setenv("STRICT_INFERENCE", "1")
    monkeypatch.delenv("DATA_SPLIT", raising=False)
    monkeypatch.delenv("SAMPLE_LIMIT", raising=False)
    monkeypatch.chdir(SUB_DIR)

    ns: dict = {
        "__name__": "__main__",
        "_INJECTED_ASR_ENGINE": _MockASREngine(),
        "_INJECTED_LLM_BACKEND": _ScriptedLLMBackend(),
    }
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            exec(compile(code, "predict_notebook.ipynb", "exec"), ns)

    # Verify history batch path was actually exercised cleanly with configured nbest_beams
    cfg_yaml = yaml.safe_load((SUB_DIR / "configs" / "e2e_qwen35_4b_sft_v2.yaml").read_text(encoding="utf-8"))
    expected_nbest = int((cfg_yaml.get("asr") or {}).get("nbest_beams", 1))
    assert any(c["mode"] == "batch" for c in asr1.calls)
    assert any(c["num_beams_nbest"] == expected_nbest for c in asr1.calls)

    nb_rows = [json.loads(line) for line in nb_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows1) == len(rows2) == len(nb_rows) == len(selected)

    # Verify CLI Run 1 == CLI Run 2 == Notebook Run and exact JSONL schema!
    expected_keys = {"id", "tool_calls", "state_hash", "time"}
    for r1, r2, rnb in zip(rows1, rows2, nb_rows):
        assert set(r1.keys()) == expected_keys
        assert set(rnb.keys()) == expected_keys
        assert isinstance(r1["time"], int) and r1["time"] >= 0
        assert isinstance(rnb["time"], int) and rnb["time"] >= 0
        assert r1["id"] == r2["id"] == rnb["id"]
        assert r1["tool_calls"] == r2["tool_calls"] == rnb["tool_calls"]
        assert r1["state_hash"] == r2["state_hash"] == rnb["state_hash"]

    # Official zaic_tools --validate CLI check on both outputs
    for pred_file in (out1, nb_out):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "zaic_tools",
                "--split",
                "public_test",
                "--root",
                str(data_mount),
                "--validate",
                str(pred_file),
            ],
            cwd=str(pub_dst),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "0 schema errors" in proc.stdout

    # Run scripts/verify_submission.py across both files
    vproc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_submission.py"),
            "--submission-dir",
            str(SUB_DIR),
            "--predictions",
            str(out1),
            "--jupyter-predictions",
            str(nb_out),
            "--data-root",
            str(data_mount),
            "--split",
            "public_test",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert vproc.returncode == 0, vproc.stderr + vproc.stdout

    # Also verify isolated subprocess import & predict.py CLI execution (no repo root on PYTHONPATH) mimicking /code container
    iso_env = {**os.environ, "PYTHONPATH": str(SUB_DIR)}
    help_proc = subprocess.run(
        [sys.executable, "predict.py", "--help"],
        cwd=str(SUB_DIR),
        env=iso_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_proc.returncode == 0, help_proc.stderr
    assert "--data-root" in help_proc.stdout and "--output" in help_proc.stdout

    iso_out = tmp_path / "result_iso" / "submission.jsonl"
    iso_script = (
        "import sys, agent, predict\n"
        "from types import SimpleNamespace\n"
        "from pathlib import Path\n"
        "class DummyASR:\n"
        "    def transcribe(self, p, **kw):\n"
        "        return SimpleNamespace(text='xin chao', raw_text='xin chao', latency_s=0.01, nbest=['xin chao'], nbest_scores=[-0.1])\n"
        "    def transcribe_batch(self, ps, **kw):\n"
        "        return [self.transcribe(p) for p in ps]\n"
        "class DummyLLM:\n"
        "    def generate(self, messages, tools=None, **kw):\n"
        "        return 'Yêu cầu ngoài phạm vi.'\n"
        "orig_load = agent.load_models\n"
        "def fake_load_models(config_path=agent.DEFAULT_CONFIG_PATH, **kw):\n"
        "    cfg, raw_cfg, _, _ = orig_load(config_path, asr_engine=DummyASR(), llm_backend=DummyLLM())\n"
        "    return cfg, raw_cfg, DummyASR(), DummyLLM()\n"
        "agent.load_models = fake_load_models\n"
        f"sys.argv = ['predict.py', '--data-root', {str(data_mount)!r}, '--output', {str(iso_out)!r}, '--limit', '2', '--strict']\n"
        "predict.main()\n"
    )
    iso_proc = subprocess.run(
        [sys.executable, "-c", iso_script],
        cwd=str(SUB_DIR),
        env=iso_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert iso_proc.returncode == 0, iso_proc.stderr
    iso_rows = [json.loads(l) for l in iso_out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(iso_rows) == 2
    assert all(set(r.keys()) == expected_keys for r in iso_rows)

    for m in [k for k in list(sys.modules) if k == "zaic_tools" or k.startswith("zaic_tools.")]:
        del sys.modules[m]
    if str(pub_dst) in sys.path:
        sys.path.remove(str(pub_dst))


def test_prepare_models_param_counter(tmp_path):
    sys.path.insert(0, str(SUB_DIR))
    from prepare_models import count_safetensors_params

    # Build minimal valid safetensors header + dummy payload
    hdr = json.dumps({"w1": {"dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64]}}).encode("utf-8")
    blob = struct.pack("<Q", len(hdr)) + hdr + (b"\x00" * 64)
    (tmp_path / "model.safetensors").write_bytes(blob)
    assert count_safetensors_params(tmp_path) == 32


def test_offline_tokenizer_and_whisper_processor_configs():
    qwen_meta = ROOT / ".cache" / "qwen35_merged_meta"
    whisper_meta = ROOT / ".cache" / "phowhisper_meta"
    if not (qwen_meta.exists() and whisper_meta.exists()):
        pytest.skip("Local metadata cache not populated")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(qwen_meta), trust_remote_code=True, local_files_only=True)
    cfg = AutoConfig.from_pretrained(str(qwen_meta), trust_remote_code=True, local_files_only=True)
    proc = AutoProcessor.from_pretrained(str(whisper_meta), local_files_only=True)
    assert cfg.architectures == ["Qwen3_5ForCausalLM"]
    assert not hasattr(cfg, "vision_config")
    assert tok.eos_token_id is not None
    assert proc is not None
