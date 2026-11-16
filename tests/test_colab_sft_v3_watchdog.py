"""Tests for SFT V3 monitoring telemetry, health circuit breakers, and Colab/Orca watchdog."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.hf_sync import BackgroundPusher, HFSyncer, make_checkpoint_callback


class _FakeHfApi:
    def __init__(self) -> None:
        self.folder_calls: list[dict] = []
        self.uploaded_progress: dict[str, dict] = {}

    def whoami(self):
        return {"name": "tester"}

    def create_repo(self, **_kw):
        pass

    def upload_file(self, **_kw):
        pass

    def upload_folder(self, **kw):
        self.folder_calls.append(kw)
        prog_path = Path(kw["folder_path"]) / "progress.json"
        if prog_path.exists():
            self.uploaded_progress[kw["path_in_repo"]] = json.loads(prog_path.read_text(encoding="utf-8"))

    def file_exists(self, **_kw):
        return False


def test_hf_checkpoint_progress_records_grad_norm_and_lr(tmp_path: Path):
    api = _FakeHfApi()
    syncer = HFSyncer("user/repo", api=api, token="t", sleep=lambda _s: None)
    pusher = BackgroundPusher(syncer, latest_wins=False)
    cb = make_checkpoint_callback(
        pusher,
        work_dir=tmp_path / "work",
        interval_s=300.0,
        save_steps=10,
        path_in_repo="exp_v3/checkpoints",
        save_latest_pointer=True,
    )

    class DummyModel:
        def save_pretrained(self, p):
            (Path(p) / "adapter_model.safetensors").write_bytes(b"ok")

    state = SimpleNamespace(
        is_world_process_zero=True,
        global_step=10,
        max_steps=100,
        epoch=0.25,
        log_history=[
            {"step": 5, "loss": 0.91, "grad_norm": 1.8, "learning_rate": 4.0e-5},
            {"step": 10, "loss": 0.62, "grad_norm": 1.15, "learning_rate": 3.7e-5},
        ],
    )
    cb.on_step_end(None, state, None, model=DummyModel())
    assert pusher.flush(timeout=10)
    pusher.close(timeout=5)

    assert "exp_v3/checkpoints/checkpoint-10" in api.uploaded_progress
    assert "exp_v3/checkpoints/latest" in api.uploaded_progress
    assert not list((tmp_path / "work").glob("snap-*")), "Expected local snap dir cleaned after upload"
    progress = api.uploaded_progress["exp_v3/checkpoints/checkpoint-10"]
    assert progress["last_loss"] == pytest.approx(0.62)
    assert progress["last_grad_norm"] == pytest.approx(1.15)
    assert progress["last_lr"] == pytest.approx(3.7e-5)
    assert len(progress["recent_losses"]) == 2
    assert progress["recent_losses"][-1]["step"] == 10


def test_model_health_monitor_callback_live_metrics_and_circuit_breakers(tmp_path: Path):
    from scripts.train_qwen38_sft import ModelHealthMonitorCallback

    # 1. Healthy logs write live_metrics.json cleanly without stopping
    cb = ModelHealthMonitorCallback(out_dir=tmp_path)
    ctrl = SimpleNamespace(should_training_stop=False)
    state = SimpleNamespace(global_step=20, max_steps=200, epoch=0.2)
    cb.on_log(None, state, ctrl, logs={"loss": 0.55, "grad_norm": 1.3, "learning_rate": 3.9e-5})
    assert ctrl.should_training_stop is False
    live = json.loads((tmp_path / "live_metrics.json").read_text(encoding="utf-8"))
    assert live["status"] == "HEALTHY"
    assert live["loss"] == pytest.approx(0.55)
    assert live["grad_norm"] == pytest.approx(1.3)

    # 2. Single transient grad spike does NOT trip circuit breaker
    cb.on_log(None, SimpleNamespace(global_step=25, max_steps=200, epoch=0.25), ctrl, logs={"loss": 0.54, "grad_norm": 62.0, "learning_rate": 3.8e-5})
    assert ctrl.should_training_stop is False

    # 3. Normal step resets spike streak, then 3 consecutive spikes DO trip circuit breaker
    cb.on_log(None, SimpleNamespace(global_step=30, max_steps=200, epoch=0.3), ctrl, logs={"loss": 0.52, "grad_norm": 1.1, "learning_rate": 3.7e-5})
    for step in (35, 40, 45):
        cb.on_log(None, SimpleNamespace(global_step=step, max_steps=200, epoch=step / 100.0), ctrl, logs={"loss": 1.8, "grad_norm": 75.0, "learning_rate": 3.6e-5})
    assert ctrl.should_training_stop is True
    assert cb.anomaly == "GRAD_EXPLOSION"
    live_err = json.loads((tmp_path / "live_metrics.json").read_text(encoding="utf-8"))
    assert live_err["status"] == "ANOMALY:GRAD_EXPLOSION"

    # 4. Immediate stop on NaN loss
    cb_nan = ModelHealthMonitorCallback(out_dir=tmp_path / "nan")
    ctrl_nan = SimpleNamespace(should_training_stop=False)
    cb_nan.on_log(None, state, ctrl_nan, logs={"loss": float("nan"), "grad_norm": 1.0})
    assert ctrl_nan.should_training_stop is True
    assert cb_nan.anomaly == "NAN_LOSS"

    # 5. Sustained loss divergence after step 60
    cb_div = ModelHealthMonitorCallback(out_dir=tmp_path / "div")
    ctrl_div = SimpleNamespace(should_training_stop=False)
    for step in (65, 70, 75):
        cb_div.on_log(None, SimpleNamespace(global_step=step, max_steps=200, epoch=0.7), ctrl_div, logs={"loss": 5.1, "grad_norm": 3.0})
    assert ctrl_div.should_training_stop is True
    assert cb_div.anomaly == "LOSS_DIVERGENCE"


def test_colab_sft_v3_watchdog_anomaly_and_v3_options():
    from scripts.colab_qwen35_sft import (
        _asset_specs,
        _build_remote_eval_cmd,
        assess_status,
        detect_log_anomaly,
    )

    assert detect_log_anomaly(["HEALTH_ANOMALY: GRAD_EXPLOSION detected at step 45"]) == "HEALTH_ANOMALY"
    assert detect_log_anomaly(["WebSocketClosedError: closed by remote"]) == "WebSocketClosedError"

    hf_ok = {"latest_step": 80, "train_done": False, "all_done": False}
    remote_anom = {
        "colab_alive": True,
        "train_alive": True,
        "gpu_util": 92,
        "disk_free_gb": 22.0,
        "live_metrics": {"status": "ANOMALY:GRAD_EXPLOSION", "anomaly": "GRAD_EXPLOSION", "step": 85},
    }
    res = assess_status(hf_ok, remote_anom, {"last_observed_step": 80}, now_ts=1000.0)
    assert res["status"] == "ANOMALY"
    assert res["wake_agent"] is True
    assert res["next_action"] == "intervene"

    remote_low_disk = {
        "colab_alive": True,
        "train_alive": True,
        "gpu_util": 90,
        "disk_free_gb": 3.2,
    }
    res_disk = assess_status(hf_ok, remote_low_disk, {"last_observed_step": 80}, now_ts=1000.0)
    assert res_disk["status"] == "LOW_DISK"
    assert res_disk["wake_agent"] is True

    # V3 config and fold0-only eval must NOT upload dev_v1.json and must honor --enable-thinking
    assets = _asset_specs(
        data_file="data/sft/qwen35_train_sft_v3_clean.jsonl.gz",
        sft_config="configs/qwen35_sft_v3_clean.yaml",
        eval_splits=["fold0"],
    )
    local_paths = [str(a[0]).replace("\\", "/") for a in assets]
    assert any(p.endswith("/data/sft/qwen35_train_sft_v3_clean.jsonl.gz") for p in local_paths)
    assert any(p.endswith("/configs/qwen35_sft_v3_clean.yaml") for p in local_paths)
    assert all(not p.endswith("/dev_v1.json") for p in local_paths)

    eval_cmd = _build_remote_eval_cmd(
        prefix="exp_qwen35_sft_v3_clean",
        split="fold0",
        transcripts="asr",
        enable_thinking=True,
    )
    assert "--enable-thinking" in eval_cmd
    assert "--no-enable-thinking" not in eval_cmd
    assert "--transcripts asr" in eval_cmd


def test_build_length_bucketed_indices_cuts_padding_and_resumes_cleanly():
    from scripts.train_qwen38_sft import build_length_bucketed_indices

    lengths = [500 + (i * 197) % 8000 for i in range(128)]
    order, start_idx, total = build_length_bucketed_indices(
        lengths, total_epochs=2.0, batch_size=4, window_batches=8, seed=42, epoch_offset=0.0
    )
    assert start_idx == 0 and total == 256 and len(order) == 256
    # Mean within-batch max-min spread must be drastically tighter than random batching
    spreads = [max(lengths[x] for x in order[i : i + 4]) - min(lengths[x] for x in order[i : i + 4]) for i in range(0, len(order), 4)]
    assert sum(spreads) / len(spreads) < 1200

    resumed, r_start, r_total = build_length_bucketed_indices(
        lengths, total_epochs=2.0, batch_size=4, window_batches=8, seed=42, epoch_offset=0.5
    )
    assert r_total == 256
    assert r_start % 4 == 0
    assert resumed == order[r_start:]


def test_zombie_proc_detection_and_eval_banner_reset(tmp_path: Path):
    import inspect
    from scripts.colab_qwen35_sft import detect_log_anomaly, is_proc_status_alive, probe_colab_state
    from scripts.train_qwen38_sft import ModelHealthMonitorCallback

    # 1. Linux zombie ('Z') / dead ('X') processes must NOT be treated as alive
    assert is_proc_status_alive("Name:\tpython3\nState:\tZ (zombie)\nPid:\t3779\n") is False
    assert is_proc_status_alive("Name:\tpython3\nState:\tX (dead)\nPid:\t3779\n") is False
    assert is_proc_status_alive("") is False
    assert is_proc_status_alive("Name:\tpython3\nState:\tR (running)\nPid:\t13301\n") is True
    assert is_proc_status_alive("Name:\tpython3\nState:\tS (sleeping)\nPid:\t13301\n") is True
    assert "'Z', 'X'" in inspect.getsource(probe_colab_state)

    # 2. Eval banner resets prior failures and 80-line tail tolerates noisy kernel warnings
    stale_eval_log = [
        "=== EVAL LAUNCH 2026-11-15 00:10:00 ===",
        "Traceback (most recent call last):",
        "WebSocketClosedError: closed",
        "=== EVAL LAUNCH 2026-11-15 00:20:00 ===",
        "Loading vLLM engine...",
        "Processing 10/250",
    ]
    assert detect_log_anomaly(stale_eval_log) is None

    noisy_oom_log = [
        "=== LAUNCH 2026-11-15 00:59:32 ===",
        "torch.OutOfMemoryError: CUDA out of memory.",
        *["WARNING:kernels.layer.layer: No kernel mapping" for _ in range(50)],
    ]
    assert detect_log_anomaly(noisy_oom_log) == "CUDA out of memory"

    # 3. Resumed ModelHealthMonitorCallback reports monotonic global step/epoch & true speed
    cb = ModelHealthMonitorCallback(out_dir=tmp_path, step_offset=200, epoch_offset=0.5, epoch_scale=1.5)
    ctrl = SimpleNamespace(should_training_stop=False)
    cb.on_log(None, SimpleNamespace(global_step=10, max_steps=100, epoch=0.1), ctrl, logs={"loss": 0.42, "grad_norm": 0.9, "learning_rate": 2.0e-5})
    live = json.loads((tmp_path / "live_metrics.json").read_text(encoding="utf-8"))
    assert live["step"] == 210
    assert live["max_steps"] == 300
    assert live["epoch"] == pytest.approx(0.65)


def test_prepare_merged_checkpoint_flattens_nested_and_fresh_hf_downloads(tmp_path: Path):
    from scripts.colab_qwen35_sft import is_merged_checkpoint_ready, prepare_merged_checkpoint

    class _FakeSyncer:
        def __init__(self) -> None:
            self.downloaded: list[tuple[str, str]] = []

        def exists(self, path_in_repo: str) -> bool:
            return path_in_repo == "exp_v3/merged/config.json"

        def download_dir(self, path_in_repo: str, local_dir: Path) -> Path:
            self.downloaded.append((path_in_repo, str(local_dir)))
            dest = Path(local_dir) / path_in_repo
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "config.json").write_text('{"architectures":["Qwen3_5ForCausalLM"]}', encoding="utf-8")
            (dest / "tokenizer.json").write_text("{}", encoding="utf-8")
            (dest / "model.safetensors").write_bytes(b"weights")
            return dest

    # 1. Incomplete .complete alone must NOT pass readiness check
    out1 = tmp_path / "run1" / "experiments" / "exp_v3"
    merged1 = out1 / "merged"
    nested = merged1 / "exp_v3" / "merged"
    nested.mkdir(parents=True, exist_ok=True)
    (merged1 / ".complete").write_text("ok", encoding="utf-8")
    (nested / "config.json").write_text("{}", encoding="utf-8")
    (nested / "tokenizer.json").write_text("{}", encoding="utf-8")
    (nested / "model.safetensors").write_bytes(b"weights")
    assert is_merged_checkpoint_ready(merged1) is False

    # Recover nested files in-place without calling HF download
    syncer1 = _FakeSyncer()
    assert prepare_merged_checkpoint(syncer1, "exp_v3", out1) is True
    assert syncer1.downloaded == []
    assert is_merged_checkpoint_ready(merged1) is True
    assert not (merged1 / "exp_v3").exists()

    # 2. Fresh download targets out_dir.parent so files land directly at merged_dir
    out2 = tmp_path / "run2" / "experiments" / "exp_v3"
    merged2 = out2 / "merged"
    syncer2 = _FakeSyncer()
    assert prepare_merged_checkpoint(syncer2, "exp_v3", out2) is True
    assert syncer2.downloaded == [("exp_v3/merged", str(out2.parent))]
    assert is_merged_checkpoint_ready(merged2) is True


def test_probe_colab_state_handles_websocket_drop(monkeypatch):
    import scripts.colab_qwen35_sft as sup

    monkeypatch.setattr(sup, "ping_session", lambda session, timeout=25, gpu=None: True)

    def _boom(*args, **kwargs):
        raise RuntimeError("WebSocketClosedError: Connection was lost.")

    monkeypatch.setattr(sup, "run_remote_py", _boom)
    res = sup.probe_colab_state("qwen38-eval-l4", prefix="exp_qwen35_sft_v3_clean", gpu="L4")
    assert res["colab_alive"] is False
    assert "Connection was lost" in res["probe_error"]


def test_free_zombie_vram_if_idle_kills_orphans_then_restarts(monkeypatch):
    import scripts.colab_qwen35_sft as sup

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(sup, "run_remote_py", lambda session, code, timeout=20: calls.append(("py", code)) or "")
    monkeypatch.setattr(sup, "restart_kernel", lambda session: calls.append(("restart", session)))
    monkeypatch.setattr(sup, "ping_session", lambda session, timeout=25: calls.append(("ping", session)) or True)

    # Do NOT touch anything while eval or train is still alive
    sup._free_zombie_vram_if_idle(
        "qwen38-eval-l4",
        {"colab_alive": True, "train_alive": False, "eval_alive": True, "vram_used_mb": 18500},
    )
    assert calls == []

    # When both train and eval are dead and GPU memory is held by orphans, kill orphans then restart kernel
    sup._free_zombie_vram_if_idle(
        "qwen38-eval-l4",
        {"colab_alive": True, "train_alive": False, "eval_alive": False, "vram_used_mb": 4600},
    )
    assert [c[0] for c in calls] == ["py", "restart", "ping"]
    assert "pkill" in calls[0][1]
    assert "EngineCore" in calls[0][1]
    assert calls[1][1] == "qwen38-eval-l4"
    assert calls[2][1] == "qwen38-eval-l4"

