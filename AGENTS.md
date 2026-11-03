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

## 4. Remote GPU Execution via Colab CLI (`colab`)

Local developer machines may lack 24GB VRAM GPUs for evaluating 3B LLMs or fine-tuning ASR models. Agents can utilize Google Colab remotely via the installed `colab` CLI (`colab.exe`).

### Colab CLI Operating Guide:

1. **Session Provisioning:**
   - Check existing sessions: `colab sessions`
   - Create a new GPU session: `colab new --gpu`
   - Check session status: `colab status`
2. **Code & Data Syncing:**
   - Upload code or lightweight configs: `colab upload <local_file_or_dir> <remote_path>`
   - Mount Google Drive if dataset is stored there: `colab drivemount /content/drive`
3. **Remote Execution:**
   - Execute single commands or scripts: `colab exec "python scripts/run_eval.py --config configs/base.yaml"`
   - Install required packages: `colab install <package_name>` or `colab exec "pip install -r requirements.txt"`
4. **Result Retrieval & Cleanup:**
   - Download evaluation outputs: `colab download <remote_results_path> experiments/exp_YYYYMMDD_<name>/`
   - Terminate session when finished: `colab stop <session_id>`

---

## 5. File Naming Conventions & Root Hygiene

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

## 6. Development Workflow & Quality Gates

- **Decompose before implementing:** Keep units small and focused.
- **Test-Driven / Verification first:** Write unit tests in `tests/` for every parser, wrapper, or metric function.
- **Metric Verification:** Always benchmark against local validation data before declaring an agent improvement successful.
- **Git Hygiene:** Verify `git status` before committing. Never commit large binary model checkpoints (`*.pth`, `*.safetensors`) to Git.
