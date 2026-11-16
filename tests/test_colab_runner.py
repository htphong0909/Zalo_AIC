"""Unit tests for colab_runner lost-connection checks, retries, and Qwen3.5 SFT supervisor."""
import subprocess
from unittest.mock import patch

import pytest

from scripts.colab_qwen35_sft import assess_status, detect_log_anomaly
from scripts.colab_runner import LOST_CONNECTION_SIGNATURES, check_lost_connection, upload_file


def test_check_lost_connection_detects_standard_error():
    sample_error = """
    File "C:\\Users\\htpho\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\jupyter_kernel_client\\wsclient.py", line 1073, in execute_interactive
        raise RuntimeError("Connection was lost.")
    RuntimeError: Connection was lost.
    """
    is_lost, sig = check_lost_connection(sample_error)
    assert is_lost is True
    assert "connection was lost" in sig.lower()


def test_check_lost_connection_detects_websocket_variants():
    assert check_lost_connection("WebSocketClosedError: connection reset by peer")[0] is True
    assert check_lost_connection("RemoteDisconnected('Remote end closed connection without response')")[0] is True
    assert check_lost_connection("Read timed out")[0] is True
    assert check_lost_connection("jupyter_kernel_client.wsclient error")[0] is True


def test_check_lost_connection_clean_output():
    clean_text = """
    Loading Whisper backbone: kzap201/phowhisper-large-zaloai
    Evaluation complete: 150 samples processed in 45.2s.
    WER: 0.99%, EM: 92.00%
    """
    is_lost, sig = check_lost_connection(clean_text)
    assert is_lost is False
    assert sig == ""


def test_check_lost_connection_empty():
    assert check_lost_connection("")[0] is False
    assert check_lost_connection(None)[0] is False


def test_upload_file_retries_transient_failure(tmp_path):
    src = tmp_path / "dummy.zip"
    src.write_bytes(b"zipdata")
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        rc = 1 if len(calls) == 1 else 0
        return subprocess.CompletedProcess(cmd, rc, stdout="ok" if rc == 0 else "", stderr="500 Error" if rc else "")

    with (
        patch("scripts.colab_runner.ensure_session", return_value=True),
        patch("scripts.colab_runner.subprocess.run", side_effect=fake_run),
        patch("scripts.colab_runner.time.sleep", return_value=None),
    ):
        res = upload_file(src, "/content/dummy.zip", session="qwen38-work", max_retries=3, gpu="G4")

    assert res.returncode == 0
    assert len(calls) == 2


def test_assess_status_transitions():
    hf_running = {"latest_step": 50, "train_done": False, "all_done": False}
    remote_ok = {"colab_alive": True, "train_alive": True, "eval_alive": False, "gpu_util": 98, "train_log_lines": 40, "train_log_tail": ["step 50 loss=0.4"]}
    v1 = assess_status(hf_running, remote_ok, {"last_observed_step": 0, "recovery_count": 2}, now_ts=1000.0)
    assert v1["status"] == "TRAINING_OK"
    assert v1["wake_agent"] is False
    assert v1["recovery_count"] == 0

    # Session disconnected triggers NEEDS_RECOVERY, waking agent only after >=3 consecutive failures
    v_drop = assess_status(hf_running, {"colab_alive": False}, {**v1, "recovery_count": 1}, now_ts=1200.0)
    assert v_drop["status"] == "NEEDS_RECOVERY"
    assert v_drop["wake_agent"] is False

    v_drop3 = assess_status(hf_running, {"colab_alive": False}, {**v1, "recovery_count": 3}, now_ts=1200.0)
    assert v_drop3["wake_agent"] is True

    # Stall detection after >30 min with 0% GPU util
    v_stall = assess_status(
        hf_running,
        {**remote_ok, "gpu_util": 0},
        {"last_observed_step": 50, "last_train_log_lines": 40, "last_progress_ts": 100.0},
        now_ts=2000.0,
    )
    assert v_stall["status"] == "STALLED"
    assert v_stall["wake_agent"] is True

    # Training done -> READY_FOR_EVAL
    hf_trained = {"latest_step": 465, "train_done": True, "all_done": False}
    v_eval = assess_status(hf_trained, {"colab_alive": True, "train_alive": False, "eval_alive": False}, v1, now_ts=2100.0)
    assert v_eval["status"] == "READY_FOR_EVAL"
    assert v_eval["next_action"] == "launch_eval"

    # All done wakes agent once to finalize report, then skips once finalized
    hf_done = {"latest_step": 465, "train_done": True, "all_done": True}
    assert assess_status(hf_done, {"colab_alive": True}, v1, now_ts=2200.0, finalized_locally=False)["wake_agent"] is True
    assert assess_status(hf_done, {"colab_alive": True}, v1, now_ts=2200.0, finalized_locally=True)["wake_agent"] is False


def test_detect_log_anomaly():
    assert detect_log_anomaly(["step 10 loss=0.5"]) is None
    assert detect_log_anomaly(["RuntimeError: CUDA out of memory. Tried to allocate"]) == "CUDA out of memory"
    assert detect_log_anomaly(["{'loss': nan, 'epoch': 0.1}"]) == "loss': nan"


def test_run_command_silent_hang_times_out(monkeypatch):
    import subprocess
    import sys
    import pytest
    import scripts.colab_runner as colab_runner

    spawned = []
    real_popen = subprocess.Popen

    def fake_popen(_cmd, **kwargs):
        proc = real_popen(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            **kwargs,
        )
        spawned.append(proc)
        return proc

    monkeypatch.setattr(colab_runner, "ensure_session", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(colab_runner, "restart_kernel", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(colab_runner, "ping_session", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(colab_runner.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(colab_runner.subprocess, "Popen", fake_popen)

    try:
        with pytest.raises(RuntimeError, match="timed out after 1s"):
            colab_runner.run_command(
                "echo should_timeout",
                session="dummy",
                timeout=-29,
                max_retries=1,
                stream=False,
            )
        assert spawned and spawned[0].poll() is not None
    finally:
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
