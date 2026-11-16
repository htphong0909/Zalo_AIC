---
name: checkpoint-sync-hf
description: Use when running ANY training, fine-tuning, evaluation, data-generation or other long (>5 min) job on Colab, Kaggle or any remote GPU — before launching, while it runs, and when recovering after a crash. Enforces preflight + persisting every checkpoint (checkpoint-<step>) to Hugging Face so no training step is lost.
---

# Durable Checkpoints on Hugging Face (MANDATORY)

## Why this skill exists
A Colab run lost ~4 hours of GPU time and **every checkpoint** because (1) results only lived on the VM disk and
were to be downloaded at the very end, (2) one blocking `colab exec` of ~105 min exceeded its 90 min timeout,
and (3) the retry logic re-ran the whole job from scratch. Remote sessions die without warning (timeouts,
WebSocket drops, server restarts). **Anything not on the Hub is already lost.**

## Iron Law
> **No GPU/long job starts without a passed HF preflight, and no training job runs without persisting
> EVERY checkpoint (`checkpoints/checkpoint-<step>`) and results to Hugging Face.**

Violating the letter of this rule violates its spirit. "It's only a quick run" is not an exception: runs are
quick right up until they aren't.

## The 6 rules
1. **Preflight first (fail fast).** Prove the token is valid *and* writable before spending GPU time.
   `HFSyncer(repo).preflight()` does this (whoami + create private repo + tiny write probe). Abort the job if it fails.
2. **Persist EVERY checkpoint and push every ≤10 minutes AND at the end, including on failure.**
   Store every checkpoint snapshot under `<prefix>/checkpoints/checkpoint-<step>/` (and mirror the newest to
   `<prefix>/checkpoints/latest`). Never overwrite or discard historical checkpoints.
3. **Push each stage's output before starting the next stage** (train → push adapter → merge → eval → push metrics).
   Push the adapter *before* the slow merge.
4. **Make jobs resumable and idempotent.** Write a completion marker **last** (e.g. `<prefix>/eval/metrics.json`).
   On start, skip any stage whose marker already exists on the Hub. Retries must resume, never restart from zero.
5. **Never put a >~60 min job inside one blocking `colab exec`.** Split into per-stage execs, or launch detached
   (`nohup python ... > /content/run.log 2>&1 &` in one short exec) and poll the log in later short execs.
   `colab restart-kernel` **kills the running job**, so auto-retry after it is only safe when rule 4 holds.
6. **Secrets stay in env.** `HF_TOKEN` is passed with `colab exec --env HF_TOKEN=...` (see
   `run_command(..., env=[...])` in `scripts/colab_runner.py`). Never write it to a file, a log, a notebook or git.

## Setup (once per session)
```powershell
$env:HF_TOKEN     = "<write token>"            # https://huggingface.co/settings/tokens (write scope)
$env:HF_SYNC_REPO = "<user>/zalo-aic-ckpt"     # private model repo; created automatically by preflight
```
If `hf auth whoami` says the token is invalid, **stop and ask the user for a fresh token** — do not work around it.

## Three ways to comply (pick the first that fits)
| Situation | Use |
|---|---|
| Training with `scripts/train_minicpm5_sft.py` | Built in: requires `--hf-repo`/`HF_SYNC_REPO`, runs preflight, snapshots and uploads EVERY checkpoint to `<prefix>/checkpoints/checkpoint-<step>` (and mirrors to `latest`), supports `--hf-interval-min` (default 10) and optional `--save-steps`, pushes `<prefix>/adapter` before merging. `--no-hf-sync` exists for throwaway smoke tests only and prints a warning. |
| Any other command (eval, datagen, custom trainer) | Wrap it: `python scripts/hf_sync_run.py --watch <dir>=<repo_path> --interval-min 10 -- <command ...>`. Pushes on a timer and once more on exit, even if the command crashes (exit code is propagated; preflight failure exits 3 without running anything). |
| Your own Python code | `from src.core.hf_sync import HFSyncer, BackgroundPusher, DirWatcher, make_checkpoint_callback` (see `src/core/hf_sync.py`). `BackgroundPusher` queues all submissions by default (`latest_wins=False`). |

Repo layout convention: `<prefix>/checkpoints/checkpoint-<step>/` (all steps preserved), `<prefix>/checkpoints/latest` (mirror pointer), `<prefix>/adapter`, `<prefix>/preds`, `<prefix>/eval/metrics.json`
(written last = branch complete), `<prefix>/training_summary.json`.

## Recovery after a crash
1. Look at what survived: `HfApi().list_repo_files("<user>/<repo>")`.
2. Finished stage (`.../eval/metrics.json` present) → skip it. Final adapter present → `syncer.download_dir(...)` and re-merge.
3. Checkpoints present under `checkpoints/checkpoint-<step>` or `checkpoints/latest` → select the desired step and warm-start with `--init-adapter <downloaded dir>` (optimizer/scheduler restart; weights are kept). Checkpoints can also be combined via model souping.
4. Partial `preds/` → re-run `run_agent.py` with `--resume` after downloading it.
5. Fix the root cause *before* relaunching (e.g. the timeout that killed it).

## Before you stop the session or claim success
Verify with evidence, not assumption: list the repo files and confirm every checkpoint and final metrics are there
**before** running `colab stop`. Report the repo path to the user.

## Red flags — stop and fix
| Thought | Reality |
|---|---|
| "Overwriting only latest is enough" | Overwriting loses historical trajectory, makes checkpoint souping and rollback impossible. Persist every `checkpoint-<step>`. |
| "I'll download everything at the end" | That is exactly what lost the checkpoints. Push periodically. |
| "Short run, no need to sync" | Preflight + one push costs seconds; a dead VM costs the whole run. |
| "The token probably works" | Run preflight; a dead token found at minute 90 is the failure we are preventing. |
| "Retry will just re-run it" | Without a completion marker + skip logic a retry repeats the whole job. |
| "Put it all in one exec to keep it simple" | One blocking exec >60 min is the single point of failure. Split or detach. |
| "Pass the token inside the script file" | Secrets go via `--env` only; scripts in `scripts/scratch/` can reach git. |
