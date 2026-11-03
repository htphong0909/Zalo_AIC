# Design Spec: Project Foundation & AI Agent Standards

- **Date:** 2026-11-03
- **Author:** Antigravity & User
- **Status:** Approved
- **Target Repository:** `ZALO_AIC`
- **Competition:** Zalo AI Challenge 2026 - LifeMate (Audio Voice Assistant Tool-calling Agent)

---

## 1. Overview & Objective

Establish a clean, professional, and robust project foundation for the Zalo AI Challenge 2026 (LifeMate track). This foundation provides:
1. A clear directory hierarchy separating code, configuration, scripts, experiments, documentation, and tests.
2. Strict file naming and repository hygiene rules to prevent clutter in the root folder.
3. An `AGENTS.md` guide at the repository root that orients any AI assistant entering the workspace, enforcing competition constraints and developer workflow standards.
4. Clean directory scaffolding without cluttering the project with dummy/placeholder code files.

---

## 2. Directory Structure

The repository root must maintain a strict separation of concerns. Only directories and essential top-level context files are created at this stage:

```text
ZALO_AIC/
├── AGENTS.md                          # Mandatory instructions and hard constraints for all AI agents
├── README.md                          # Project introduction and workspace structure guide
├── .gitignore                         # Ignore data/, experiments/, models/, system/Python cache
├── configs/                           # Configuration files (.yaml, .json)
├── src/                               # Core Python source packages
│   ├── core/                          # Common schemas, logging, general utilities
│   ├── asr/                           # Speech-to-Text audio processing and models
│   ├── agent/                         # Prompt templates, function calling, reasoning loop
│   ├── env/                           # Safe wrapper around zaic_tools.ToolEnv
│   └── eval/                          # Local evaluation metrics (Tool-call F1, State Hash match)
├── scripts/                           # Runnable operational scripts (evaluation, submission, verification)
├── experiments/                       # Local experiment logs, artifacts, and test outputs (git-ignored)
├── tests/                             # Unit tests and validation test cases
└── docs/                              # Project documentation
    ├── overview/                      # Official competition documentation (Data, Description, Evaluation, Rule)
    └── specs/                         # Architecture specifications and system design documents
```

---

## 3. Naming Conventions & Repository Hygiene

### 3.1 File Naming Conventions
- **Python Source Code (`src/`, `tests/`)**: `snake_case` reflecting single responsibility (e.g., `tool_caller.py`, `eval_metrics.py`, `test_env_wrapper.py`).
- **Scripts (`scripts/`)**: `snake_case` prefixed with an action verb (e.g., `run_eval.py`, `make_submission.py`, `verify_submission.py`).
- **Configuration Files (`configs/`)**: `snake_case` or `kebab-case` describing the configuration role (e.g., `base.yaml`, `whisper_base.yaml`).
- **Experiment Runs (`experiments/`)**: Prefixed with date and descriptive slug: `exp_YYYYMMDD_<name>/` (e.g., `exp_20261103_baseline_asr/`).
- **Documentation (`docs/`)**: Markdown `.md` format. Specifications use `YYYY-MM-DD-<topic>-design.md`.

### 3.2 Root Guard & Git Hygiene
- No temporary scripts (`test.py`, `temp.json`, `run.log`, etc.) in the workspace root.
- Heavy model weights, checkpoints, raw experiment outputs, and environment states are stored under `experiments/` or `.cache/` and are strictly excluded via `.gitignore`.
- Only keep `.gitkeep` files where necessary to preserve empty directories in git.

---

## 4. `AGENTS.md` Specification

The root `AGENTS.md` must be read by all AI agents upon session initialization. It contains:

1. **Mission & Context**: Audio request + user context + installed apps $\rightarrow$ tool calls sequence via `zaic_tools.ToolEnv` $\rightarrow$ final `tool_calls` + `state_hash`.
2. **Hard Constraints (Red Lines)**:
   - **Data Access:** NEVER read or inspect files inside `<split>_environment/` or `state.json` directly. Interacting with app data is only permitted via `zaic_tools.ToolEnv`.
   - **Model Size:** Total system (ASR + LLM + others) must be strictly under 6 Billion parameters. The LLM must be strictly under 5 Billion parameters with context length $\le 16\text{K}$ tokens.
   - **Latency:** Maximum 30 seconds processing time per sample on benchmark hardware (NVIDIA A30 24GB, 16-core CPU, 64GB RAM).
   - **Execution Environment:** Fully offline Docker container without external network access at inference time.
   - **Reproducibility:** Fixed random seeds across all libraries; deterministic decoding (`temperature = 0` / greedy decoding).
3. **Directory Map & Scopes:** Clear guide indicating where code, configs, and experiments must be placed.
4. **Development Workflow:**
   - Write modular code with single responsibility.
   - Verify every new component with unit tests in `tests/`.
   - Never commit large weights or dirty temporary logs to git.

---

## 5. Verification & Acceptance Criteria

1. Directory scaffolding matches the structure specified in Section 2 without dummy code files.
2. `.gitignore` is updated to exclude `experiments/`, `*.pth`, `*.bin`, `*.safetensors`, and local virtual environments.
3. `AGENTS.md` exists at repository root, containing all hard constraints, directory mappings, and naming rules.
4. `README.md` exists at repository root with a clean overview of the project structure and setup.
5. All new files are committed to git clean and ready for development.
