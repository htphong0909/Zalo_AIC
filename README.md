# Zalo AI Challenge 2026 — LifeMate

> **Track:** Voice Assistant Tool-calling Agent  
> **Repository:** `ZALO_AIC`

---

## 📌 Project Overview

**LifeMate** challenges teams to develop an autonomous AI agent capable of converting Vietnamese spoken requests into accurate, multi-step actions across various simulated applications (Smart Home, Banking, Ride-Hailing, Food Delivery, Email, Reminders, Contacts, etc.).

The agent does not just converse; it is evaluated on **what it does**:
- Calling the right tools with valid arguments in the proper sequence via `zaic_tools.ToolEnv`.
- Requesting missing arguments via `sys.ask_missing_info` when information is incomplete.
- Returning an empty tool call list `[]` when the request is out-of-scope.
- Producing a verifiable final app state hash (`state_hash`).

---

## ⚙️ Core Technical Constraints

All solutions must strictly comply with official competition rules:

| Constraint | Limit / Requirement |
|---|---|
| **Total Model Size** | **< 6 Billion parameters** (ASR + LLM + any other modules combined) |
| **LLM Model Size** | **< 5 Billion parameters**, context length $\le$ 16,384 tokens |
| **Per-sample Latency** | **$\le$ 30 seconds** on NVIDIA A30 GPU (24GB VRAM), 16-core CPU, 64GB RAM |
| **Environment Access** | **Strictly prohibited** from reading `state.json` directly. All actions go via `zaic_tools.ToolEnv` |
| **Inference Mode** | 100% offline Docker container (no network access at evaluation) |
| **Reproducibility** | Fixed random seeds, deterministic decoding (`temperature = 0`) |

---

## 📂 Repository Structure

The project follows a clean, modular architecture:

```text
ZALO_AIC/
├── AGENTS.md                  # Mandatory guidelines and operational constraints for AI agents
├── README.md                  # Project overview and setup guide
├── .gitignore                 # Exclusion rules for data, experiments, and model checkpoints
├── configs/                   # Declarative YAML/JSON configuration files for models and runs
├── src/                       # Core Python packages
│   ├── core/                  # Schemas, logging, and data structures
│   ├── asr/                   # Speech-to-Text audio processing and model engines
│   ├── agent/                 # Prompt engineering, tool calling strategy, ReAct loop
│   ├── env/                   # Safe execution wrapper around zaic_tools.ToolEnv
│   └── eval/                  # Internal evaluation harness (Tool-call F1 & State Hash match)
├── scripts/                   # Operational entrypoints (run_eval.py, make_submission.py)
├── experiments/               # Local experiment outputs, run logs, and metric dumps (git-ignored)
├── tests/                     # Unit tests and automated test suites
└── docs/                      # Official competition rules, data guides, and system design specs
    ├── overview/              # Official competition documentation
    ├── specs/                 # Architectural specifications
    └── plans/                 # Detailed implementation plans
```

---

## 🚀 Remote GPU Execution via Colab CLI

For training and heavy inference runs requiring 24GB VRAM GPUs, developers and agents can execute scripts remotely on Google Colab using `colab` CLI:

```bash
# 1. Provision a remote GPU runtime
colab new --gpu

# 2. Check active sessions
colab sessions

# 3. Upload code or configuration
colab upload configs/ /content/configs/
colab upload src/ /content/src/

# 4. Pull model weights or dataset from Hugging Face (No Google Drive)
colab exec "hf download <hf_repo_id> --local-dir /content/models/<model_name>"

# 5. Run remote inference or evaluation
colab exec "python scripts/run_eval.py --config configs/base.yaml"

# 6. Push trained checkpoints to Hugging Face Hub
colab exec "hf upload <hf_repo_id> /content/experiments/<exp_name>/checkpoint"

# 7. Fetch experiment summary metrics locally
colab download /content/experiments/ experiments/

# 8. Release resources when finished
colab stop <session_id>
```

---

## 🤗 Model Weights & Data Management (Hugging Face Hub)

All heavy binary artifacts (ASR models, LLM checkpoints, datasets) are versioned and stored on **Hugging Face Hub**. **Google Drive is not used.**

- **Download:** `hf download <repo_id> --local-dir <local_dir>`
- **Upload:** `hf upload <repo_id> <local_dir> [path_in_repo] --repo-type [model|dataset]`
- **Python Integration:** Automated via `huggingface_hub` (`snapshot_download`, `upload_folder`).

---

## 🛠️ For AI Assistants

All AI agents working on this project must consult [AGENTS.md](file:///C:/WORKSPACE/ZALO_AIC/AGENTS.md) before writing code, modifying pipelines, or generating submissions.
