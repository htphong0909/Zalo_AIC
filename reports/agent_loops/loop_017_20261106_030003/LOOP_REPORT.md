# Loop Report #017 — ACCEPTED

**Hypothesis:** Calibrated bilateral acoustic collar expansion (symmetric 60ms) and margin safety buffering (150ms) with relaxed autoregressive repetition penalty (1.03)  
**Decision Reason:** Dev WER improved (-0.01%) or EM improved (+0.00%) without train collapse  
**Timestamp:** 2026-11-06 03:19:29  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **0.99%** | **1.55%** | `-0.01%` | **$\le$ 1.50%** |
| **CER** | **0.86%** | **1.09%** | `+0.04%` | **$\le$ 1.00%** |
| **Sentence EM** | **92.00%** | **86.96%** | `+0.00%` | **$\ge$ 95.00%** |
| **Latency** | 0.363s | 0.348s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
