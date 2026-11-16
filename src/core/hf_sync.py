"""Durable checkpoint sync to the Hugging Face Hub (see AGENTS.md section 4.3).

Why this exists: remote GPU sessions (Colab/Kaggle) die without warning (timeouts, WebSocket drops,
server restarts). Anything that only lives on the VM disk is lost. Every long-running job must therefore
push its checkpoints/results to the Hub roughly every 10 minutes and again when it finishes.

Design rules:
- Fail fast: ``HFSyncer.preflight()`` proves the token is valid and the repo is writable BEFORE any GPU
  time is spent (a dead token discovered 90 minutes in is the failure mode this module prevents).
- Never block training: ``BackgroundPusher`` uploads asynchronously and keeps only the newest snapshot.
- Never crash training on a transient network error: ``push`` retries with backoff and returns ``False``;
  only ``raise_on_fail=True`` (used for final artifacts) raises.
- The token is never logged.
"""
from __future__ import annotations

import json
import logging
import os
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("hf_sync")

DEFAULT_INTERVAL_S = 600.0  # ~10 minutes
ENV_REPO = "HF_SYNC_REPO"
ENV_TOKEN = "HF_TOKEN"
ENV_PRIVATE = "HF_SYNC_PRIVATE"
_IPV4_FALLBACK_LOCK = threading.Lock()
_IPV4_FALLBACK_ACTIVE = False
_LOCAL_BASE_MODEL_RE = re.compile(
    r"^(base_model:\s*(?:/|[A-Za-z]:\\|\./).*|-\s*(?:/|[A-Za-z]:\\|\./).*)\r?\n?",
    re.MULTILINE,
)


def _sanitize_peft_readmes(root: Path) -> None:
    """Strip local filesystem `base_model` paths from PEFT README frontmatter before HF upload."""
    for readme in root.rglob("README.md"):
        try:
            txt = readme.read_text(encoding="utf-8")
        except Exception:
            continue
        if not txt.startswith("---"):
            continue
        parts = txt.split("---", 2)
        if len(parts) < 3:
            continue
        fm = parts[1]
        cleaned_fm = _LOCAL_BASE_MODEL_RE.sub("", fm)
        cleaned_fm = re.sub(r"\nbase_model:\s*(?=\n[a-zA-Z_-]+:|\n*$)", "\n", cleaned_fm)
        if cleaned_fm != fm:
            readme.write_text(f"---{cleaned_fm}---{parts[2]}", encoding="utf-8")


def _maybe_enable_ipv4_fallback(exc: Exception) -> bool:
    """Switch socket resolution to prefer IPv4 on Windows when IPv6 TLS resets (WinError 10054)."""
    global _IPV4_FALLBACK_ACTIVE
    if os.name != "nt":
        return False
    msg = f"{type(exc).__name__}: {exc}"
    if "10054" not in msg and "ConnectError" not in msg and "ConnectionResetError" not in msg:
        return False
    with _IPV4_FALLBACK_LOCK:
        if _IPV4_FALLBACK_ACTIVE:
            return False
        orig_getaddrinfo = socket.getaddrinfo

        def _ipv4_first_getaddrinfo(host, port, family=0, *args, **kwargs):
            if family == 0:
                try:
                    res = orig_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)
                    if res:
                        return res
                except OSError:
                    pass
            return orig_getaddrinfo(host, port, family, *args, **kwargs)

        socket.getaddrinfo = _ipv4_first_getaddrinfo  # type: ignore[assignment]
        _IPV4_FALLBACK_ACTIVE = True
        logger.warning("hf_sync: IPv6 connection reset detected; enabling IPv4-first fallback for HF Hub.")
        return True


class HFSyncError(RuntimeError):
    """Sync cannot be guaranteed (bad token, no write access, missing repo id, final push failed)."""


def resolve_repo_id(explicit: str | None = None) -> str | None:
    """Explicit value wins, then the ``HF_SYNC_REPO`` environment variable."""
    return explicit or os.environ.get(ENV_REPO) or None


class HFSyncer:
    """Synchronous, retrying uploader/downloader bound to one Hub repo."""

    def __init__(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        repo_type: str = "model",
        private: bool | None = None,
        api: Any | None = None,
        retries: int = 4,
        backoff_s: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not repo_id or "/" not in repo_id:
            raise HFSyncError(
                f"Invalid HF repo id {repo_id!r}; expected '<user-or-org>/<name>'. "
                f"Pass --hf-repo or set {ENV_REPO}."
            )
        self.repo_id = repo_id
        self.repo_type = repo_type
        if private is None:
            raw_private = os.environ.get(ENV_PRIVATE)
            if raw_private is None or not raw_private.strip():
                self.private = True
            else:
                self.private = raw_private.strip().lower() not in {"0", "false", "no", "off"}
        else:
            self.private = bool(private)
        self.token = token or os.environ.get(ENV_TOKEN) or None
        self.retries = max(1, int(retries))
        self.backoff_s = float(backoff_s)
        self._sleep = sleep
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)
        self.api = api
        self._lock = threading.Lock()
        self.last_ok_ts: float | None = None
        self.failures = 0

    def _call_with_retry(self, fn: Callable[[], Any]) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                _maybe_enable_ipv4_fallback(exc)
                if attempt < self.retries:
                    self._sleep(min(self.backoff_s, 2.0) * attempt)
        assert last_exc is not None
        raise last_exc

    def preflight(self) -> str:
        """Verify token + create repo + prove write access. Raises ``HFSyncError`` so jobs abort early."""
        try:
            who = self._call_with_retry(lambda: self.api.whoami())
        except Exception as exc:  # noqa: BLE001 - any auth/network failure must abort the job
            raise HFSyncError(
                f"HF token is missing/invalid ({type(exc).__name__}). Export a valid {ENV_TOKEN} "
                f"(or run `hf auth login --force`) before starting any GPU job."
            ) from exc
        try:
            repo_exists = False
            if hasattr(self.api, "repo_info"):
                try:
                    self.api.repo_info(repo_id=self.repo_id, repo_type=self.repo_type)
                    repo_exists = True
                except Exception:  # noqa: BLE001 - missing/private repo will be created/verified below
                    repo_exists = False
            if not repo_exists:
                self._call_with_retry(
                    lambda: self.api.create_repo(
                        repo_id=self.repo_id, repo_type=self.repo_type, private=self.private, exist_ok=True
                    )
                )
            self._call_with_retry(
                lambda: self.api.upload_file(
                    path_or_fileobj=json.dumps({"probe_ts": time.time()}).encode("utf-8"),
                    path_in_repo=".sync_probe.json",
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    commit_message="sync preflight",
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise HFSyncError(
                f"No write access to {self.repo_id} ({type(exc).__name__}). Use a token with write scope."
            ) from exc
        return str(who.get("name") or "unknown") if isinstance(who, dict) else "unknown"

    def push(
        self,
        local_path: str | Path,
        path_in_repo: str,
        message: str,
        *,
        raise_on_fail: bool = False,
        ignore_patterns: list[str] | None = None,
    ) -> bool:
        """Upload a file or folder. Retries with backoff. Returns True on success."""
        p = Path(local_path)
        if not p.exists():
            logger.warning("hf_sync: nothing to push, %s does not exist", p)
            return False
        if p.is_dir():
            _sanitize_peft_readmes(p)
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with self._lock:
                    if p.is_file():
                        self.api.upload_file(
                            path_or_fileobj=str(p),
                            path_in_repo=path_in_repo,
                            repo_id=self.repo_id,
                            repo_type=self.repo_type,
                            commit_message=message,
                        )
                    else:
                        self.api.upload_folder(
                            repo_id=self.repo_id,
                            folder_path=str(p),
                            path_in_repo=path_in_repo,
                            repo_type=self.repo_type,
                            commit_message=message,
                            ignore_patterns=ignore_patterns,
                        )
                self.last_ok_ts = time.time()
                logger.info("hf_sync: pushed %s -> %s/%s", p, self.repo_id, path_in_repo)
                return True
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self.failures += 1
                _maybe_enable_ipv4_fallback(exc)
                logger.warning(
                    "hf_sync: push attempt %d/%d failed for %s (%s: %s)",
                    attempt, self.retries, path_in_repo, type(exc).__name__, exc,
                )
                if attempt < self.retries:
                    self._sleep(self.backoff_s * attempt)
        logger.error("hf_sync: GIVING UP pushing %s after %d attempts", path_in_repo, self.retries)
        if raise_on_fail:
            raise HFSyncError(f"Failed to push {path_in_repo} to {self.repo_id}") from last_exc
        return False

    def exists(self, path_in_repo: str) -> bool:
        try:
            return bool(
                self._call_with_retry(
                    lambda: self.api.file_exists(
                        repo_id=self.repo_id, filename=path_in_repo, repo_type=self.repo_type
                    )
                )
            )
        except Exception:  # noqa: BLE001
            return False

    def download_dir(self, path_in_repo: str, local_dir: str | Path) -> Path:
        """Download everything under ``path_in_repo``; returns ``local_dir/path_in_repo``."""
        from huggingface_hub import snapshot_download

        self._call_with_retry(
            lambda: snapshot_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                allow_patterns=f"{path_in_repo.rstrip('/')}/*",
                local_dir=str(local_dir),
                token=self.token,
            )
        )
        return Path(local_dir) / path_in_repo


class BackgroundPusher:
    """Async uploader: slow uploads never block training.

    By default (latest_wins=False), it maintains a FIFO queue so EVERY submitted checkpoint
    is safely pushed to the Hub in sequence without losing any intermediate steps.
    If latest_wins=True, a newer snapshot supersedes an unstarted queued one.
    """

    def __init__(self, syncer: HFSyncer, *, latest_wins: bool = False) -> None:
        self._syncer = syncer
        self.latest_wins = latest_wins
        self._cv = threading.Condition()
        self._queue: list[tuple[Path, str, str, Callable[[bool], None] | None]] = []
        self._busy = False
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="hf-bg-pusher", daemon=True)
        self._thread.start()

    def submit(
        self,
        local_dir: str | Path,
        path_in_repo: str,
        message: str,
        on_done: Callable[[bool], None] | None = None,
    ) -> None:
        with self._cv:
            if self._closed:
                logger.warning("hf_sync: BackgroundPusher is closed, ignoring submit for %s", path_in_repo)
                if on_done is not None:
                    on_done(False)
                return
            if self.latest_wins and self._queue:
                old = self._queue.pop(0)
                if old[3] is not None:
                    old[3](False)  # superseded snapshot: let owner clean it up
            self._queue.append((Path(local_dir), path_in_repo, message, on_done))
            self._cv.notify_all()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._closed:
                    self._cv.wait()
                if not self._queue and self._closed:
                    return
                job = self._queue.pop(0)
                self._busy = True
            ok = False
            try:
                ok = self._syncer.push(job[0], job[1], job[2])
            except Exception:  # noqa: BLE001
                logger.exception("hf_sync: background push crashed")
            finally:
                if job[3] is not None:
                    try:
                        job[3](ok)
                    except Exception:  # noqa: BLE001
                        logger.exception("hf_sync: on_done callback crashed")
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def flush(self, timeout: float | None = None) -> bool:
        """Block until all submitted snapshots have been attempted. Returns False on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while self._queue or self._busy:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cv.wait(remaining)
        return True

    def close(self, timeout: float | None = 600.0) -> None:
        self.flush(timeout)
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)


def _fingerprint(path: Path) -> tuple[int, int, int]:
    n = size = newest = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                st = os.stat(os.path.join(root, name))
            except OSError:
                continue
            n += 1
            size += st.st_size
            newest = max(newest, st.st_mtime_ns)
    return n, size, newest


class DirWatcher:
    """Pushes watched directories every ``interval_s`` (only when their content changed) and once at stop()."""

    def __init__(
        self,
        syncer: HFSyncer,
        watch: list[tuple[Path, str]],
        interval_s: float = DEFAULT_INTERVAL_S,
        message_prefix: str = "periodic sync",
    ) -> None:
        self._syncer = syncer
        self._watch = [(Path(p), repo_path) for p, repo_path in watch]
        self._interval_s = float(interval_s)
        self._prefix = message_prefix
        self._seen: dict[str, tuple[int, int, int]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hf-dir-watcher", daemon=True)

    def start(self) -> "DirWatcher":
        self._thread.start()
        return self

    def sync_once(self, *, force: bool = False) -> int:
        pushed = 0
        for path, repo_path in self._watch:
            if not path.exists():
                continue
            fp = _fingerprint(path)
            if fp[0] == 0 or (not force and self._seen.get(repo_path) == fp):
                continue
            if self._syncer.push(path, repo_path, f"{self._prefix} {time.strftime('%Y-%m-%d %H:%M:%S')}"):
                self._seen[repo_path] = fp
                pushed += 1
        return pushed

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self.sync_once()
            except Exception:  # noqa: BLE001
                logger.exception("hf_sync: periodic sync crashed")

    def stop(self, *, final: bool = True) -> int:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        return self.sync_once(force=True) if final else 0


def snapshot_adapter(model: Any, dest: str | Path, info: dict[str, Any]) -> Path:
    """Save a PEFT adapter (weights only, no optimizer state) plus ``progress.json`` into ``dest``."""
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(d))
    (d / "progress.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return d


def make_checkpoint_callback(
    pusher: BackgroundPusher,
    work_dir: str | Path,
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    save_steps: int | None = None,
    path_in_repo: str = "checkpoints",
    save_latest_pointer: bool = True,
    step_offset: int = 0,
    epoch_offset: float = 0.0,
    epoch_scale: float = 1.0,
) -> Any:
    """Build a ``transformers.TrainerCallback`` that snapshots and uploads EVERY checkpoint.

    Instead of overwriting a single 'latest' folder, every checkpoint is permanently stored
    under ``<repo_base>/checkpoint-<step>``. Optionally (default True), it also mirrors the
    newest checkpoint to ``<repo_base>/latest`` for fast discovery and resume convenience.
    Snapshots trigger periodically every ``interval_s`` seconds (and optionally every ``save_steps``
    optimizer steps) and once at the end of training.
    """
    import shutil

    from transformers import TrainerCallback

    work = Path(work_dir)
    repo_base = path_in_repo.rstrip("/")
    if repo_base.endswith("/latest"):
        repo_base = repo_base[:-7].rstrip("/")

    class HFCheckpointCallback(TrainerCallback):
        def __init__(self) -> None:
            self._last = time.monotonic()
            self._last_step: int = -1
            self.snapshots = 0

        def _snap(self, state: Any, model: Any, tag: str) -> None:
            if model is None or not getattr(state, "is_world_process_zero", True):
                return
            local_step = int(getattr(state, "global_step", 0))
            step = step_offset + local_step
            local_epoch = getattr(state, "epoch", None)
            abs_epoch = (
                (epoch_offset + float(local_epoch) * float(epoch_scale))
                if local_epoch is not None
                else (epoch_offset if epoch_offset > 0 else None)
            )
            dest = work / f"snap-{tag}-{step}"
            log_hist = getattr(state, "log_history", []) or []
            last_loss = None
            last_grad_norm = None
            last_lr = None
            recent_losses: list[dict[str, Any]] = []
            for entry in reversed(log_hist):
                if last_loss is None and "loss" in entry:
                    last_loss = entry["loss"]
                if last_grad_norm is None and "grad_norm" in entry:
                    last_grad_norm = entry["grad_norm"]
                if last_lr is None and "learning_rate" in entry:
                    last_lr = entry["learning_rate"]
                if last_loss is not None and last_grad_norm is not None and last_lr is not None:
                    break
            for entry in log_hist:
                if "loss" in entry:
                    recent_losses.append(
                        {
                            "step": entry.get("step"),
                            "loss": entry.get("loss"),
                            "grad_norm": entry.get("grad_norm"),
                            "learning_rate": entry.get("learning_rate"),
                        }
                    )
            snapshot_adapter(
                model,
                dest,
                {
                    "global_step": step,
                    "max_steps": step_offset + int(getattr(state, "max_steps", 0) or 0),
                    "epoch": abs_epoch,
                    "last_loss": last_loss,
                    "last_grad_norm": last_grad_norm,
                    "last_lr": last_lr,
                    "recent_losses": recent_losses[-10:],
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            self.snapshots += 1
            self._last_step = step

            ckpt_repo_path = f"{repo_base}/checkpoint-{step}"
            latest_repo_path = f"{repo_base}/latest"

            if save_latest_pointer:
                # 1. Upload permanently to checkpoint-<step>
                pusher.submit(
                    dest,
                    ckpt_repo_path,
                    f"checkpoint {tag} step {step}",
                    on_done=None,
                )
                # 2. Mirror to latest/ and cleanup local snapshot after both uploads succeed
                pusher.submit(
                    dest,
                    latest_repo_path,
                    f"checkpoint latest (step {step})",
                    on_done=lambda _ok, d=dest: shutil.rmtree(d, ignore_errors=True),
                )
            else:
                pusher.submit(
                    dest,
                    ckpt_repo_path,
                    f"checkpoint {tag} step {step}",
                    on_done=lambda _ok, d=dest: shutil.rmtree(d, ignore_errors=True),
                )

        def on_step_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
            local_step = int(getattr(state, "global_step", 0))
            step = step_offset + local_step
            time_due = (time.monotonic() - self._last >= interval_s)
            step_due = (
                save_steps is not None
                and save_steps > 0
                and local_step > 0
                and local_step % save_steps == 0
            )
            if (time_due or step_due) and step != self._last_step:
                self._last = time.monotonic()
                self._snap(state, model, "periodic")
            return control

        def on_train_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
            local_step = int(getattr(state, "global_step", 0))
            step = step_offset + local_step
            if step != self._last_step:
                self._snap(state, model, "final")
            pusher.flush(timeout=900)
            return control

    return HFCheckpointCallback()
