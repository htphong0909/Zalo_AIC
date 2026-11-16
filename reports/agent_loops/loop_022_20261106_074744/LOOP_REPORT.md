# Loop Report #022 — REJECTED

**Hypothesis:** Southern conversational negative particle normalization standardizing pre-predicate debuccalized [hɔŋ] to formal written không  
**Decision Reason:** Dev did not improve (delta WER=+0.01%, delta EM=-0.34%) or Train regressed  
**Timestamp:** 2026-11-06 08:03:56  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **0.96%** | **1.56%** | `+0.01%` | **$\le$ 1.50%** |
| **CER** | **0.86%** | **1.09%** | `+0.00%` | **$\le$ 1.00%** |
| **Sentence EM** | **92.00%** | **86.62%** | `-0.34%` | **$\ge$ 95.00%** |
| **Latency** | 0.364s | 0.357s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
