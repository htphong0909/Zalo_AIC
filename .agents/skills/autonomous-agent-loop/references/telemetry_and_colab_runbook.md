# Telemetry CLI & Colab GPU Dual-Referee Runbook

## 1. Deterministic Telemetry Commands (`scripts/loop_logger.py`)

All permanent run records live under `reports/agent_loops/` (tracked outside `src/` so `git restore src/` rollbacks NEVER delete rejected experiment patches or KPIs).

```powershell
# 1. Read <=25-line anti-amnesia brief (Main Orchestrator start of loop)
python scripts/loop_logger.py brief --max-recent 5

# 2. Initialize Loop N folder & update Live HTML Monitor
python scripts/loop_logger.py start --loop 1 --hypothesis "Whisper initial_prompt entity biasing" --agent loop-researcher

# 3. Save Phase 1 Forensic Diagnosis Markdown
python scripts/loop_logger.py forensic --loop 1 --file scripts/scratch/temp_forensic.md

# 4. Snapshot git diff src/ configs/ -> 02_code_changes.patch BEFORE Referee/Rollback!
python scripts/loop_logger.py snapshot --loop 1 `
  --optimization "Added tool catalog entity prompt biasing to WhisperASREngine" `
  --pytest-passed --pytest-summary "10/10 passed" `
  --audit-passed --audit-notes "Verified by sop-auditor"

# 5. Finalize Pareto Gate, commit or rollback src/, render Dashboards
python scripts/loop_logger.py finalize --loop 1 `
  --hypothesis "Whisper initial_prompt entity biasing" `
  --train-json scripts/scratch/temp_train_res.json `
  --dev-json scripts/scratch/temp_dev_res.json `
  --raw-log scripts/scratch/temp_referee_raw.log

# 6. Re-render Markdown & HTML Dashboards manually anytime
python scripts/loop_logger.py render
```

---

## 2. Permanent Files Produced Per Loop (`reports/agent_loops/loop_NNN_<timestamp>/`)

| File | Content & Purpose |
| :--- | :--- |
| `00_loop_meta.json` | Loop ID, hypothesis, start timestamp |
| `01_forensic_brief.md` | Acoustic root-cause analysis on Train Glass-Box errors |
| `02_code_changes.patch` | Full `git diff src/ configs/` captured **before** Pareto rollback |
| `02_modification_summary.json` | Modified files, optimization summary, Pytest & SOP audit verdict |
| `03_kpi_metrics.json` | Train KPIs + Sanitized macro-only Dev KPIs + Deltas |
| `03_train_error_diff.json` | Detailed error samples and phoneme confusion shifts on Train |
| `03_raw_execution.log` | Complete raw stdout/stderr from GPU evaluator run |
| `LOOP_REPORT.md` | 1-page human-readable scientific summary linked from `AGENT_LOOP_DASHBOARD.md` |

---

## 3. GPU Dual-Referee Execution Pattern

When running Phase 3:
Execute the unified, self-healing pipeline with hardcoded auto-check for 'Connection was lost' and auto-kernel restart:
```powershell
python scripts/colab_runner.py referee --loop <N> --hypothesis "<HYPOTHESIS>"
```
This single command automatically:
1. Packages the codebase and uploads to Colab `/content/ZALO_AIC`.
2. Runs Referee 1 on Colab GPU with auto-retry on WebSocket drop.
3. Downloads Train evaluation metrics.
4. Runs Referee 2 on Colab GPU with zero-leakage and auto-retry on WebSocket drop.
5. Downloads Dev evaluation metrics.
6. Calls `loop_logger.py finalize` and outputs the Pareto Gate summary.

---

## 4. Durable Checkpoints on Hugging Face (Gate 7 — MANDATORY)

Colab sessions die without warning; never leave work only on the VM. Full rules: skill `checkpoint-sync-hf`.

```powershell
$env:HF_TOKEN = "<write token>"; $env:HF_SYNC_REPO = "<user>/zalo-aic-ckpt"

# Training: built in (preflight + snapshots every 10 min stored under checkpoints/checkpoint-<step> + adapter pushed before merge)
python scripts/train_minicpm5_sft.py ... --hf-repo $env:HF_SYNC_REPO --hf-prefix exp_name --hf-interval-min 10

# Any other long command: wrap it (pushes every 10 min and once more on exit, even on failure)
python scripts/hf_sync_run.py --watch /content/out=exp_name --interval-min 10 -- python <script> <args>
```
- `colab_runner.run_command(..., env=["HF_TOKEN=...", "HF_SYNC_REPO=..."])` passes the secrets to the kernel (never print or store them).
- Keep each blocking `colab exec` under ~60 min (split stages or launch detached with `nohup` and poll); a kernel restart kills the job, so retries must resume from HF markers, not restart.
- Short referee evals whose outputs are tiny and logged by `loop_logger.py` are exempt; anything that trains, or runs > 10 min, is not.
