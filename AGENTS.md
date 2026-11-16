# AGENTS.md — AI Agent Guidelines & Project Operating Manual

> **Scope:** This document is the mandatory entry point for all AI agents working in this repository. Any agent (regardless of model or role) must read and strictly adhere to these instructions, constraints, and project conventions before taking any action.

---

## 1. Project Mission & Overview

- **Competition:** Zalo AI Challenge 2026 — Track: **LifeMate (Voice Assistant Tool-calling Agent)**
- **Objective:** Build an autonomous voice-driven AI agent that transforms Vietnamese spoken requests into precise, multi-step actions across simulated digital apps (smart home, banking, ride-hailing, food delivery, email, reminders, etc.).
- **Input per sample:**
  1. Audio file (`.wav`, 16 kHz, mono, 16-bit) containing the spoken request in Vietnamese (with occasional English loanwords), covering diverse regional accents (North, Central, South).
  2. User context: `current_time`, `timezone`, location, and `user_profile` (home/work address, preferences).
  3. Installed apps & tools: schema catalog (`<split>_tools_catalog.json`) exposing available tool definitions.
  4. Conversation history: previous dialogue turns (if any).
  5. Simulated app environment: accessible strictly via the official `zaic_tools.ToolEnv`.
- **Expected Output:**
  - An ordered sequence of `tool_calls` (`name`, `arguments`) executed in the environment.
  - Final `state_hash` computed by `env.get_state_hash()`.
  - Handle three scenarios:
    - Normal actions: call required tools sequentially to fulfill the request.
    - Missing information: call `sys.ask_missing_info` with tool name and missing parameter names.
    - Out-of-scope: return empty list `[]` if no installed app can handle the request.

---

## 2. Hard Constraints & Red Lines (DO NOT VIOLATE)

Any solution violating these rules will receive a score of **0** or be disqualified:

1. **NO DIRECT APP STATE ACCESS:**
   - **Strict prohibition:** You MUST NOT open, read, parse, copy, or bypass any file under `<split>_environment/` (including `state.json`).
   - App data can ONLY be read and updated via official tool calls executed inside `zaic_tools.ToolEnv`.
   - At inference time, only `env.validate()`, `env.call()`, `env.get_tool_calls()`, and `env.get_state_hash()` are permitted.
2. **MODEL SIZE LIMITS:**
   - Total system size (ASR + LLM + any auxiliary models) must be strictly **< 6 Billion parameters**.
   - The LLM component alone must be strictly **< 5 Billion parameters** with a maximum context length of **16,384 tokens** (16K).
3. **LATENCY LIMIT:**
   - Processing time per sample must be **$\le$ 30 seconds**.
   - Target benchmark hardware: 1x NVIDIA A30 GPU (24GB VRAM), 16-core Intel Xeon Gold 6442Y CPU, 64GB RAM.
4. **OFFLINE INFERENCE:**
   - The final submission is packaged in a Docker container and evaluated completely offline (no internet access, no external APIs).
5. **DETERMINISTIC REPRODUCIBILITY:**
   - Fix all random seeds (`random`, `numpy`, `torch`).
   - Use deterministic decoding (`temperature = 0.0` or greedy search) for both ASR and LLM components.

---

## 3. Directory Map & File Responsibilities

All files must live in their designated module. Keep boundaries clean:

| Directory | Scope & Responsibility |
|---|---|
| `src/core/` | Common schemas, data models (`Pydantic`), structured logging, and utilities. |
| `src/asr/` | Audio preprocessing, speech recognition engines (e.g. Whisper, PhoWhisper), ASR wrappers. |
| `src/agent/` | Prompt engineering, tool-calling strategies, JSON parsing, and ReAct loop execution. |
| `src/env/` | Safe interaction wrapper for `zaic_tools.ToolEnv` preventing illegal operations. |
| `src/eval/` | Local evaluation harness: Tool-call F1, token-F1 for fuzzy fields, and State Hash matching. |
| `configs/` | Declarative YAML/JSON configuration files for models, prompts, and experiments. |
| `scripts/` | Runnable operational scripts (`run_eval.py`, `make_submission.py`, `verify_submission.py`). |
| `experiments/` | Local experiment outputs, run logs, and metrics snapshots (ignored by Git). |
| `tests/` | Automated unit tests, integration tests, and smoke test suites. |
| `docs/` | Architectural specs (`docs/specs/`), plans (`docs/plans/`), and competition notes. |

---

## 4. Remote GPU Execution

Local developer machines may lack 24GB VRAM GPUs for evaluating 3B LLMs or fine-tuning ASR models. Agents can utilize either Google Colab or Kaggle Batch Kernels remotely.

### 4.1. Google Colab CLI (`colab`)

1. **Session Provisioning:**
   - Check existing sessions: `colab sessions`
   - Create a new GPU session: `colab new -s <name> --gpu <L4|T4>`
   - Check session status: `colab status`
2. **Code & Environment Setup:**
   - Upload code or lightweight configs: `colab upload <local_file_or_dir> <remote_path>`
   - Install packages: `colab install <package_name>` or `colab exec "pip install -r requirements.txt huggingface_hub"`
3. **Data & Model Weight Retrieval from Hugging Face:**
   - Download datasets and model checkpoints directly from Hugging Face into the Colab VM:
     `colab exec "hf download <hf_repo_id> --local-dir /content/models/<model_name>"`
   - *(Note: Do NOT mount or store anything to Google Drive).*
4. **Remote Execution:**
   - Execute single commands or scripts: `colab exec -s <name> -f <script_path>`
5. **Checkpoint Upload & Cleanup:**
   - Upload fine-tuned model checkpoints or logs directly from Colab to Hugging Face:
     `colab exec "hf upload <hf_repo_id> /content/experiments/<exp_name>/checkpoint <path_in_repo>"`
   - Download summary metrics locally: `colab download <remote_results_path> experiments/exp_YYYYMMDD_<name>/`
   - Terminate session when finished: `colab stop -s <session_id>`

### 4.2. Kaggle Batch GPU Execution (`kaggle` CLI — Alternative for Colab Outages)

When Google Colab runs out of compute units or suffers from 503 Service Unavailable, Kaggle provides **30 hours/week of free GPU (2x NVIDIA T4 or 1x P100)** in headless batch mode.

Detailed Guide: see [`docs/KAGGLE_GPU_GUIDE.md`](docs/KAGGLE_GPU_GUIDE.md).

- **Authentication:** Credentials in `~/.kaggle/kaggle.json` (account: `xunthanhh`). Verify via `python -m kaggle kernels list --mine`.
- **Workflow:**
  1. Package clean codebase: `python scripts/kaggle_runner.py package`
  2. Test GPU Ping: `python scripts/kaggle_runner.py ping`
  3. Push batch job: `python -m kaggle kernels push -p <staging_dir>`
  4. Poll status: `python -m kaggle kernels status <kernel_slug>`
  5. Download outputs: `python -m kaggle kernels output <kernel_slug> -p <output_dir>`
- **No Google Drive:** Synchronize all weights and datasets directly via Hugging Face Hub.

---

### 4.3. System-Prompt Mandate: Durable Checkpoints & Auto-Resume (MANDATORY FOR EVERY AGENT REQUEST)

> **CRITICAL DIRECTIVE (TREAT AS SYSTEM PROMPT):** Remote GPU sessions (Colab/Kaggle) die without warning; anything that only lives on the VM disk is lost. Checkpoint saving and automatic resumption are non-negotiable hard rules.

1. **Preflight before GPU time:** `HF_TOKEN` (write scope) and `HF_SYNC_REPO=<user>/<repo>` must be set and verified (`src/core/hf_sync.py: HFSyncer.preflight()`). If the token is invalid, STOP and ask the user — never run without sync.
2. **Persist EVERY checkpoint (all steps preserved, never overwrite):**
   - Save and upload every checkpoint to its own distinct folder `<prefix>/checkpoints/checkpoint-<step>/` on Hugging Face Hub (e.g. `checkpoint-10`, `checkpoint-20`, `checkpoint-30`).
   - Do NOT overwrite historical checkpoints; every step is retained to enable model souping, step-by-step metric evaluation, and rollback.
   - A convenience mirror `<prefix>/checkpoints/latest/` is kept up to date pointing to the latest snapshot for instant recovery.
   - Periodic sync occurs every $\le$ 10 minutes (and optionally on step boundaries via `--save-steps <N>`), plus once at the end of training (including on failure).
   - Training via `scripts/train_minicpm5_sft.py` and `scripts/train_minicpm5_dpo.py` syncs all checkpoints automatically; wrap any other long command with `python scripts/hf_sync_run.py --watch <dir>=<repo_path> --interval-min 10 -- <command>`.
3. **MANDATORY AUTO-RESUME ON RETRY:**
   - Any remote training script or orchestrator MUST implement auto-resume logic: before starting training, check if `<prefix>/checkpoints/latest/` exists on Hugging Face Hub.
   - If a checkpoint exists, the job MUST automatically download and resume from that checkpoint. **NEVER restart training from step 0 when a partial checkpoint exists.**
4. **Push each stage's output before the next stage**, write a completion marker last (`eval/metrics.json`), and make jobs resumable so retries skip finished work.
5. **Keep each blocking `colab exec` under ~60 min** (split stages or run detached with `nohup` and poll). Pass secrets only with `colab exec --env`, never in files, logs or git.
6. **Verify on the Hub (list repo files) before `colab stop`** and before reporting success.

Full procedure and recovery steps: skill `.agents/skills/checkpoint-sync-hf/SKILL.md`.

---

## 5. Model Weights & Data Management via Hugging Face Hub (NO GOOGLE DRIVE)

All external artifacts (model weights, fine-tuned checkpoints, audio datasets, synthetic data) are strictly managed via **Hugging Face Hub**. **Google Drive must NOT be used.**

### Core Directives for Agents:
1. **Agent Responsibility:**
   - The AI agent is responsible for automating file management on Hugging Face: pushing checkpoints, pulling required weights, verifying hashes, and organizing repository paths.
2. **Authentication:**
   - Authenticate using stored tokens (`hf auth login` / `HF_TOKEN` environment variable).
   - Check authenticated accounts with `hf auth whoami` or `hf auth list`.
3. **CLI Usage (`hf`):**
   - Download model/data: `hf download <repo_id> [filename] --local-dir <local_dir>`
   - Upload checkpoint/data: `hf upload <repo_id> <local_path> [path_in_repo] [--repo-type model|dataset]`
   - Copy files: `hf cp <src> <dest>`
4. **Python API Usage (`huggingface_hub`):**
   ```python
   from huggingface_hub import HfApi, hf_hub_download, snapshot_download

   # Download weights or data
   local_dir = snapshot_download(repo_id="username/model_or_dataset", local_dir="models/checkpoint")

   # Upload artifact or checkpoint
   api = HfApi()
   api.upload_folder(
       folder_path="experiments/exp_YYYYMMDD_name/checkpoints",
       repo_id="username/model_name",
       repo_type="model",
   )
   ```
5. **Local Git Hygiene:**
   - Model weights (`*.safetensors`, `*.bin`, `*.pt`, `*.pth`) are strictly excluded from git. Always store and pull them from Hugging Face.

---

## 6. File Naming Conventions & Root Hygiene

1. **Python Source Files (`src/`, `tests/`):**
   - Use `snake_case` reflecting single responsibility (e.g., `tool_validator.py`, `whisper_engine.py`, `eval_metrics.py`).
2. **Operational Scripts (`scripts/`):**
   - Use `snake_case` prefixed with an action verb (e.g., `run_eval.py`, `make_submission.py`).
3. **Configurations (`configs/`):**
   - Use `snake_case` or `kebab-case` with descriptive names (e.g., `base.yaml`, `whisper_base.yaml`).
4. **Experiment Runs (`experiments/`):**
   - Format: `exp_YYYYMMDD_<descriptive_slug>/` (e.g., `exp_20261103_baseline_react/`).
5. **Root Guard:**
   - **NEVER** write scratch scripts (`test.py`, `check.py`), temporary JSONs, audio files, or log files in the project root.
   - All temporary scripts belong in `.cache/` or `experiments/`.

---

## 7. Development Workflow & Quality Gates

- **Decompose before implementing:** Keep units small and focused.
- **Test-Driven / Verification first:** Write unit tests in `tests/` for every parser, wrapper, or metric function.
- **Metric Verification:** Always benchmark against local validation data before declaring an agent improvement successful.
- **Git Hygiene:** Verify `git status` before committing. Never commit large binary model checkpoints (`*.pth`, `*.safetensors`) to Git.

---

## 8. Mandatory Experiment Logging & History Protocol

For EVERY experiment run (ASR, LLM SFT, DPO, RL, or evaluation), the agent MUST record and persist:

1. **Local Experiment Report (`experiments/<exp_id>/REPORT.md`):**
   - Goal, hypothesis, and dataset distribution.
   - Hyperparameters, runtime, and training loss.
   - Hugging Face repository links to all stored checkpoints and artifacts.
   - Full evaluation metrics (Tool-call F1, State Match Rate, Precision, Recall, breakdown by `act`, `ask`, `oos`, `history`).
   - Per-sample qualitative diagnostics (improved vs regressed samples).
2. **Central Dashboard Update:**
   - Update `reports/LLM_AGENT_EXPERIMENTS_DASHBOARD.md` (for LLM tool-calling agent experiments) or `reports/AGENT_LOOP_DASHBOARD.md` (for ASR experiments) with the new row and delta against the active baseline.
3. **Transparent User Reporting:**
   - In all user communications, explicitly cite the exact Hugging Face commits verified and provide clickable links to the report and artifacts.
