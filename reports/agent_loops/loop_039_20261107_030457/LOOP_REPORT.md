# Loop Report #039 — REJECTED

**Hypothesis:** Colloquial Dialectal Negation Normalization: Southern Vietnamese 'hông' before action/modal predicates mapped to standard 'không'  
**Decision Reason:** Dev did not improve (delta WER=+0.03%, delta EM=-1.01%) or Train regressed  
**Timestamp:** 2026-11-07 03:12:00  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **0.80%** | **1.53%** | `+0.03%` | **$\le$ 1.50%** |
| **CER** | **0.62%** | **1.04%** | `+0.01%` | **$\le$ 1.00%** |
| **Sentence EM** | **92.67%** | **85.95%** | `-1.01%` | **$\ge$ 95.00%** |
| **Latency** | 0.365s | 0.351s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
