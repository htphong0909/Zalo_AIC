# Project Foundation & Agent Standards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the clean directory scaffolding, updated `.gitignore`, root `AGENTS.md`, and `README.md` for the Zalo AI Challenge 2026 LifeMate project.

**Architecture:** Pure scaffolding and repository standardization without boilerplate code files: create empty directory hierarchy with `.gitkeep`, update `.gitignore` for experiment logs and model artifacts, write comprehensive `AGENTS.md` for AI assistant orientation, and write `README.md`.

**Tech Stack:** Git, Markdown, PowerShell/Bash.

**Spec:** [docs/specs/2026-11-03-project-foundation-and-agent-standards-design.md](file:///C:/WORKSPACE/ZALO_AIC/docs/specs/2026-11-03-project-foundation-and-agent-standards-design.md)

## Global Constraints

- Never add dummy or placeholder Python code files into `src/` or `tests/`.
- All directories must be tracked in Git via `.gitkeep`.
- `AGENTS.md` must clearly specify all hard rules from the competition: no direct access to `<split>_environment/` or `state.json`, system < 6B params, LLM < 5B with <= 16K context, <= 30s latency, offline Docker on NVIDIA A30 GPU, and deterministic decoding.
- Root guard: zero scratch, temporary, or log files in the repository root.

---

### Task 1: Directory Scaffolding & Git Hygiene

**Files:**
- Create:
  - `configs/.gitkeep`
  - `src/core/.gitkeep`
  - `src/asr/.gitkeep`
  - `src/agent/.gitkeep`
  - `src/env/.gitkeep`
  - `src/eval/.gitkeep`
  - `scripts/.gitkeep`
  - `experiments/.gitkeep`
  - `tests/.gitkeep`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Spec directory layout.
- Produces: Clean directory tree tracked in git, ignoring `experiments/` contents and heavy model weights.

- [ ] **Step 1: Create directories and `.gitkeep` files**

Create all required directory paths with `.gitkeep` inside:
- `configs/.gitkeep`
- `src/core/.gitkeep`
- `src/asr/.gitkeep`
- `src/agent/.gitkeep`
- `src/env/.gitkeep`
- `src/eval/.gitkeep`
- `scripts/.gitkeep`
- `experiments/.gitkeep`
- `tests/.gitkeep`

- [ ] **Step 2: Update `.gitignore` to ignore experiments and weights**

Ensure `.gitignore` contains rules for:
```gitignore
# Data & Experiments
data/
/data/
experiments/*
!experiments/.gitkeep

# Model weights & large checkpoints
*.pth
*.pt
*.bin
*.safetensors
*.onnx

# Cache & artifacts
.cache/
runs/
```

- [ ] **Step 3: Verify git status reflects untracked `.gitkeep` and clean directory structure**

Run:
```powershell
git status
```
Expected: Only `.gitignore` and `.gitkeep` files appear in untracked/modified list.

- [ ] **Step 4: Commit directory scaffolding**

Run:
```powershell
git add configs/ src/ scripts/ experiments/ tests/ .gitignore
git commit -m "chore: scaffold clean directory hierarchy with gitkeep and updated gitignore"
```

---

### Task 2: AI Agent Instructions (`AGENTS.md`)

**Files:**
- Create: `AGENTS.md`

**Interfaces:**
- Consumes: Competition specifications and hard constraints from `docs/specs/2026-11-03-project-foundation-and-agent-standards-design.md` and `docs/overview/`.
- Produces: Root agent orientation document that any LLM/agent must follow.

- [ ] **Step 1: Write `AGENTS.md`**

Write `AGENTS.md` containing:
1. **Project Mission**: Audio request in Vietnamese -> Agent reasoning & tool calls -> Output `tool_calls` and `state_hash`.
2. **Hard Constraints & Red Lines**:
   - Zero direct file access to `<split>_environment/` or `state.json`. All reads/writes go through `zaic_tools.ToolEnv`.
   - Total model size < 6B parameters. LLM < 5B parameters, context length <= 16K tokens.
   - Latency limit: <= 30 seconds per sample on NVIDIA A30 GPU, 16-core CPU, 64GB RAM.
   - Offline container: 100% offline at inference, no external APIs.
   - Deterministic execution: Fixed random seeds, temperature = 0 / greedy decoding.
3. **Directory Map & Scopes**: Table of where each file type lives (`src/core`, `src/asr`, `src/agent`, `src/env`, `src/eval`, `configs/`, `scripts/`, `experiments/`, `tests/`, `docs/`).
4. **File Naming & Root Guard Rules**: `snake_case` for code, action verbs for `scripts/`, `exp_YYYYMMDD_<name>/` for `experiments/`, zero temporary files at root.
5. **Development Workflow**: Single-responsibility units, unit testing before completion, local evaluation metric verification.

- [ ] **Step 2: Verify `AGENTS.md` content and formatting**

Check that:
- File exists at `C:\WORKSPACE\ZALO_AIC\AGENTS.md`.
- No placeholders, TODOs, or unresolved sections exist.
- All 5 hard rules from the official documentation are explicitly included.

- [ ] **Step 3: Commit `AGENTS.md`**

Run:
```powershell
git add AGENTS.md
git commit -m "docs: add AGENTS.md guidelines and hard constraints for AI agents"
```

---

### Task 3: Project Documentation & Final Verification (`README.md`)

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: Project foundation layout and specs.
- Produces: Clear human-facing project overview at repository root.

- [ ] **Step 1: Write `README.md`**

Write `README.md` containing:
- Project title: Zalo AI Challenge 2026 - LifeMate (Audio Voice Assistant Tool-calling Agent).
- Project Overview & Target Track.
- Repository structure tree.
- Core technical requirements summary (hardware, latency, model size).
- Quick guide on where code, configs, and experiments belong.

- [ ] **Step 2: Verify workspace integrity and git status**

Run:
```powershell
git status
```
Verify working tree is clean after staging.

- [ ] **Step 3: Commit and push changes to remote repository**

Run:
```powershell
git add README.md
git commit -m "docs: add project README.md overview and directory guide"
git push origin main
```
Verify push succeeds.
