# Loop Report #003 — ACCEPTED

**Hypothesis:** Zalo-specialized PhoWhisper Large (1.54B) with Vietnamese ITN and domain prompt conditioning  
**Decision Reason:** Dev WER improved (-10.68%) or EM improved (+60.87%) without train collapse  
**Timestamp:** 2026-11-05 05:10:34  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **1.44%** | **1.74%** | `-10.68%` | **$\le$ 3.00%** |
| **CER** | **1.24%** | **1.20%** | `-7.79%` | **$\le$ 3.00%** |
| **Sentence EM** | **88.67%** | **85.62%** | `+60.87%` | **$\ge$ 80.00%** |
| **Latency** | 1.492s | 1.398s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
