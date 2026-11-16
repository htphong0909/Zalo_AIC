#!/usr/bin/env python3
"""Run ANY long command while syncing output folders to the Hugging Face Hub every ~10 minutes.

Use this to wrap every GPU/long job that does not already sync itself (evals, data generation, custom
training scripts). It pushes on a timer, and once more when the command exits (success, failure or signal).

    python scripts/hf_sync_run.py --repo user/zalo-aic-ckpt \
        --watch /content/exp_branchA=branchA --interval-min 10 -- python scripts/run_agent.py ...

``--watch LOCAL_DIR[=PATH_IN_REPO]`` is repeatable (default repo path = directory name).
Token comes from HF_TOKEN; repo defaults to HF_SYNC_REPO. Exit code = the wrapped command's exit code
(3 = HF preflight failed, nothing was run).
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.hf_sync import (  # noqa: E402
    DEFAULT_INTERVAL_S,
    DirWatcher,
    HFSyncError,
    HFSyncer,
    resolve_repo_id,
)


def _parse_watch(items: list[str]) -> list[tuple[Path, str]]:
    """``LOCAL_DIR`` or ``LOCAL_DIR=PATH_IN_REPO`` (``=`` avoids clashing with Windows drive colons)."""
    out: list[tuple[Path, str]] = []
    for item in items:
        local, _, repo_path = item.partition("=")
        out.append((Path(local), (repo_path or Path(local).name).strip("/")))
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: hf_sync_run.py [options] -- <command ...>", file=sys.stderr)
        return 2
    split = argv.index("--")
    opts, cmd = argv[:split], argv[split + 1 :]
    if not cmd:
        print("no command given after --", file=sys.stderr)
        return 2

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None, help="HF repo '<user>/<name>' (default: $HF_SYNC_REPO)")
    ap.add_argument("--watch", action="append", default=[], help="LOCAL_DIR[=PATH_IN_REPO], repeatable")
    ap.add_argument("--interval-min", type=float, default=DEFAULT_INTERVAL_S / 60.0)
    ap.add_argument("--repo-type", default="model", choices=("model", "dataset"))
    args = ap.parse_args(opts)

    logging.basicConfig(level=logging.INFO, format="[hf_sync %(asctime)s] %(message)s", datefmt="%H:%M:%S")
    repo = resolve_repo_id(args.repo)
    if not repo or not args.watch:
        print("ERROR: --repo (or HF_SYNC_REPO) and at least one --watch are required.", file=sys.stderr)
        return 2

    try:
        syncer = HFSyncer(repo, repo_type=args.repo_type)
        user = syncer.preflight()
    except HFSyncError as exc:
        print(f"ERROR (preflight): {exc}\nNothing was run; fix HF access first.", file=sys.stderr)
        return 3
    print(f"[hf_sync] preflight OK as '{user}' -> {repo}; syncing every {args.interval_min:g} min", flush=True)

    watcher = DirWatcher(syncer, _parse_watch(args.watch), interval_s=args.interval_min * 60.0).start()
    proc = subprocess.Popen(cmd)

    def _forward(signum, _frame):  # noqa: ANN001
        proc.send_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _forward)
        except (ValueError, OSError):
            pass

    try:
        rc = proc.wait()
    finally:
        t0 = time.time()
        pushed = watcher.stop(final=True)
        print(f"[hf_sync] final sync pushed {pushed} folder(s) in {time.time() - t0:.0f}s", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
