# Loop Report #011 — ACCEPTED

**Hypothesis:** Symmetric bilateral acoustic collar padding (40ms pre + 40ms post), prompt entity de-biasing, and financial dialect ITN recovery  
**Decision Reason:** Dev WER improved (-0.01%) or EM improved (-0.33%) without train collapse  
**Timestamp:** 2026-11-06 00:09:18  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **1.14%** | **1.60%** | `-0.01%` | **$\le$ 1.50%** |
| **CER** | **0.96%** | **1.08%** | `+0.01%` | **$\le$ 1.00%** |
| **Sentence EM** | **91.33%** | **86.96%** | `-0.33%` | **$\ge$ 95.00%** |
| **Latency** | 0.365s | 0.352s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
