# Loop Report #018 — REJECTED

**Hypothesis:** Asymmetric acoustic collar geometry (60ms pre-roll, 90ms post-roll buffer) combined with zero-mean centering (DC offset elimination)  
**Decision Reason:** Dev did not improve (delta WER=+0.00%, delta EM=+0.00%) or Train regressed  
**Timestamp:** 2026-11-06 03:42:49  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **0.99%** | **1.55%** | `+0.00%` | **$\le$ 1.50%** |
| **CER** | **0.86%** | **1.09%** | `+0.00%` | **$\le$ 1.00%** |
| **Sentence EM** | **92.00%** | **86.96%** | `+0.00%` | **$\ge$ 95.00%** |
| **Latency** | 0.364s | 0.346s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
