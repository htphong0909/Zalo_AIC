"""Tests for durable HF checkpoint sync (no network: a fake HfApi records calls)."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from src.core.hf_sync import (
    BackgroundPusher,
    DirWatcher,
    HFSyncError,
    HFSyncer,
    make_checkpoint_callback,
    resolve_repo_id,
    snapshot_adapter,
)


class FakeApi:
    def __init__(self, *, fail_whoami=False, fail_upload_times=0, fail_probe=False):
        self.fail_whoami = fail_whoami
        self.fail_upload_times = fail_upload_times
        self.fail_probe = fail_probe
        self.folder_calls: list[dict] = []
        self.file_calls: list[dict] = []
        self.created: list[dict] = []
        self.gate: threading.Event | None = None

    def whoami(self):
        if self.fail_whoami:
            raise RuntimeError("401 invalid token")
        return {"name": "tester"}

    def create_repo(self, **kw):
        self.created.append(kw)

    def upload_file(self, **kw):
        if self.fail_probe and kw.get("path_in_repo") == ".sync_probe.json":
            raise RuntimeError("403 read-only token")
        self.file_calls.append(kw)

    def upload_folder(self, **kw):
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail_upload_times > 0:
            self.fail_upload_times -= 1
            raise ConnectionError("network blip")
        self.folder_calls.append(kw)

    def file_exists(self, **kw):
        return kw["filename"] == "present.json"


def _syncer(api, **kw):
    return HFSyncer("user/repo", api=api, token="t", sleep=lambda _s: None, **kw)


def test_resolve_repo_id_prefers_explicit(monkeypatch):
    monkeypatch.setenv("HF_SYNC_REPO", "env/repo")
    assert resolve_repo_id("x/y") == "x/y"
    assert resolve_repo_id(None) == "env/repo"
    monkeypatch.delenv("HF_SYNC_REPO")
    assert resolve_repo_id(None) is None


def test_invalid_repo_id_rejected():
    with pytest.raises(HFSyncError):
        HFSyncer("no-slash", api=FakeApi())


def test_preflight_ok_creates_private_repo_and_probes_write():
    api = FakeApi()
    assert _syncer(api).preflight() == "tester"
    assert api.created[0]["private"] is True and api.created[0]["exist_ok"] is True
    assert api.file_calls[0]["path_in_repo"] == ".sync_probe.json"


def test_preflight_fails_fast_on_bad_token():
    with pytest.raises(HFSyncError, match="invalid"):
        _syncer(FakeApi(fail_whoami=True)).preflight()


def test_preflight_fails_fast_when_token_is_read_only():
    with pytest.raises(HFSyncError, match="write access"):
        _syncer(FakeApi(fail_probe=True)).preflight()


def test_push_retries_transient_failures_then_succeeds(tmp_path: Path):
    (tmp_path / "a.bin").write_bytes(b"x")
    api = FakeApi(fail_upload_times=2)
    s = _syncer(api)
    assert s.push(tmp_path, "ckpt/latest", "m") is True
    assert s.failures == 2 and len(api.folder_calls) == 1


def test_push_gives_up_quietly_but_raises_when_asked(tmp_path: Path):
    (tmp_path / "a.bin").write_bytes(b"x")
    s = _syncer(FakeApi(fail_upload_times=99), retries=2)
    assert s.push(tmp_path, "p", "m") is False
    with pytest.raises(HFSyncError):
        s.push(tmp_path, "p", "m", raise_on_fail=True)


def test_push_missing_path_is_noop(tmp_path: Path):
    assert _syncer(FakeApi()).push(tmp_path / "nope", "p", "m") is False


def test_exists_uses_file_exists():
    s = _syncer(FakeApi())
    assert s.exists("present.json") and not s.exists("absent.json")


def test_background_pusher_is_latest_wins_and_cleans_superseded(tmp_path: Path):
    api = FakeApi()
    api.gate = threading.Event()  # hold the first upload so later submits queue up
    s = _syncer(api)
    pusher = BackgroundPusher(s, latest_wins=True)
    dirs = []
    for i in range(3):
        d = tmp_path / f"snap{i}"
        d.mkdir()
        (d / "w.bin").write_bytes(b"x")
        dirs.append(d)
    done: dict[int, bool] = {}
    pusher.submit(dirs[0], "latest", "s0", on_done=lambda ok: done.__setitem__(0, ok))
    time.sleep(0.2)  # worker picks up snap0 and blocks on the gate
    pusher.submit(dirs[1], "latest", "s1", on_done=lambda ok: done.__setitem__(1, ok))
    pusher.submit(dirs[2], "latest", "s2", on_done=lambda ok: done.__setitem__(2, ok))
    api.gate.set()
    assert pusher.flush(timeout=10)
    pusher.close(timeout=5)
    pushed = [c["commit_message"] for c in api.folder_calls]
    assert pushed == ["s0", "s2"]  # s1 was superseded in latest_wins mode
    assert done == {0: True, 1: False, 2: True}


def test_background_pusher_queues_all_checkpoints(tmp_path: Path):
    """Default mode (latest_wins=False) queues and uploads every checkpoint."""
    api = FakeApi()
    s = _syncer(api)
    pusher = BackgroundPusher(s, latest_wins=False)
    dirs = []
    for i in range(3):
        d = tmp_path / f"snap{i}"
        d.mkdir()
        (d / "w.bin").write_bytes(b"x")
        dirs.append(d)
    done: dict[int, bool] = {}
    for i, d in enumerate(dirs):
        pusher.submit(d, f"checkpoints/checkpoint-{i * 10}", f"step {i * 10}", on_done=lambda ok, idx=i: done.__setitem__(idx, ok))
    assert pusher.flush(timeout=10)
    pusher.close(timeout=5)
    pushed = [c["commit_message"] for c in api.folder_calls]
    assert pushed == ["step 0", "step 10", "step 20"]
    assert done == {0: True, 1: True, 2: True}


def test_make_checkpoint_callback_saves_every_checkpoint(tmp_path: Path):
    """Callback saves each checkpoint under checkpoint-<step> and mirrors to latest."""
    api = FakeApi()
    s = _syncer(api)
    pusher = BackgroundPusher(s, latest_wins=False)
    cb = make_checkpoint_callback(
        pusher,
        work_dir=tmp_path / "work",
        interval_s=100.0,
        save_steps=10,
        path_in_repo="test_exp/checkpoints",
        save_latest_pointer=True,
    )

    class DummyModel:
        def save_pretrained(self, p):
            (Path(p) / "adapter_model.safetensors").write_bytes(b"fake_adapter")

    class DummyState:
        is_world_process_zero = True
        global_step = 10
        max_steps = 100
        epoch = 0.5
        log_history = [{"loss": 0.42}]

    model = DummyModel()
    state = DummyState()

    # Step end triggers save because save_steps=10 matches global_step=10
    cb.on_step_end(None, state, None, model=model)
    assert pusher.flush(timeout=10)

    # Now step 20 triggers final on_train_end
    state.global_step = 20
    state.log_history.append({"loss": 0.15})
    cb.on_train_end(None, state, None, model=model)
    assert pusher.flush(timeout=10)
    pusher.close(timeout=5)

    paths = [c["path_in_repo"] for c in api.folder_calls]
    assert "test_exp/checkpoints/checkpoint-10" in paths
    assert "test_exp/checkpoints/latest" in paths
    assert "test_exp/checkpoints/checkpoint-20" in paths


def test_dir_watcher_pushes_only_on_change_and_at_stop(tmp_path: Path):
    api = FakeApi()
    d = tmp_path / "exp"
    d.mkdir()
    (d / "log.txt").write_text("a")
    w = DirWatcher(_syncer(api), [(d, "exp")], interval_s=3600)
    assert w.sync_once() == 1
    assert w.sync_once() == 0  # unchanged -> no redundant upload
    (d / "log.txt").write_text("a-changed-and-longer")
    assert w.sync_once() == 1
    assert w.stop(final=True) == 1  # final push is forced
    assert len(api.folder_calls) == 3


def test_snapshot_adapter_writes_weights_and_progress(tmp_path: Path):
    class M:
        def save_pretrained(self, p):
            (Path(p) / "adapter_model.safetensors").write_bytes(b"w")

    d = snapshot_adapter(M(), tmp_path / "s", {"global_step": 7})
    assert (d / "adapter_model.safetensors").exists()
    assert '"global_step": 7' in (d / "progress.json").read_text(encoding="utf-8")


def test_wrapper_aborts_without_running_command_when_preflight_fails(monkeypatch, tmp_path: Path):
    import scripts.hf_sync_run as w

    marker = tmp_path / "ran.txt"

    class BadSyncer:
        def __init__(self, *a, **k):
            pass

        def preflight(self):
            raise HFSyncError("token invalid")

    monkeypatch.setattr(w, "HFSyncer", BadSyncer)
    rc = w.main(["--repo", "u/r", "--watch", str(tmp_path), "--", sys.executable, "-c",
                 f"open(r'{marker}','w').write('x')"])
    assert rc == 3 and not marker.exists()


def test_wrapper_runs_command_and_pushes_final_sync(monkeypatch, tmp_path: Path):
    import scripts.hf_sync_run as w

    api = FakeApi()
    out = tmp_path / "exp"
    out.mkdir()

    monkeypatch.setattr(w, "HFSyncer", lambda repo, **k: HFSyncer(repo, api=api, token="t", sleep=lambda _s: None))
    rc = w.main(["--repo", "u/r", "--watch", f"{out}=results", "--interval-min", "60", "--", sys.executable, "-c",
                 f"open(r'{out / 'metrics.json'}','w').write('{{}}')"])
    assert rc == 0
    assert any(c["path_in_repo"] == "results" for c in api.folder_calls)  # final sync happened


def test_wrapper_propagates_exit_code_and_still_syncs(monkeypatch, tmp_path: Path):
    import scripts.hf_sync_run as w

    api = FakeApi()
    out = tmp_path / "exp"
    out.mkdir()
    (out / "partial.txt").write_text("keep me")
    monkeypatch.setattr(w, "HFSyncer", lambda repo, **k: HFSyncer(repo, api=api, token="t", sleep=lambda _s: None))
    rc = w.main(["--repo", "u/r", "--watch", str(out), "--", sys.executable, "-c", "import sys; sys.exit(7)"])
    assert rc == 7 and len(api.folder_calls) == 1  # crash still persists partial results


def test_preflight_supports_public_repo_via_arg_and_env(monkeypatch):
    api_arg = FakeApi()
    s_arg = HFSyncer("u/public-rl", api=api_arg, token="t", private=False, sleep=lambda _s: None)
    s_arg.preflight()
    assert api_arg.created == [
        {"repo_id": "u/public-rl", "repo_type": "model", "private": False, "exist_ok": True}
    ]

    monkeypatch.setenv("HF_SYNC_PRIVATE", "0")
    api_env = FakeApi()
    s_env = HFSyncer("u/public-rl-env", api=api_env, token="t", sleep=lambda _s: None)
    s_env.preflight()
    assert api_env.created == [
        {"repo_id": "u/public-rl-env", "repo_type": "model", "private": False, "exist_ok": True}
    ]

    class ExistingPublicRepoApi(FakeApi):
        def repo_info(self, *, repo_id, repo_type):
            return {"id": repo_id, "repo_type": repo_type, "private": False}

    monkeypatch.delenv("HF_SYNC_PRIVATE", raising=False)
    api_existing = ExistingPublicRepoApi()
    s_existing = HFSyncer("u/existing-public", api=api_existing, token="t", sleep=lambda _s: None)
    s_existing.preflight()
    assert api_existing.created == []


def test_push_sanitizes_local_base_model_in_peft_readme(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    readme = ckpt / "README.md"
    readme.write_text(
        "---\nbase_model: /content/models/qwen35_sft_v2_merged\nlibrary_name: peft\n---\n# Model Card\n",
        encoding="utf-8",
    )
    api = FakeApi()
    assert _syncer(api).push(ckpt, "exp/checkpoints/checkpoint-10", "save step 10") is True
    text = readme.read_text(encoding="utf-8")
    assert "/content/models/qwen35_sft_v2_merged" not in text
    assert "library_name: peft" in text
    assert text.startswith("---\n") and "# Model Card" in text



