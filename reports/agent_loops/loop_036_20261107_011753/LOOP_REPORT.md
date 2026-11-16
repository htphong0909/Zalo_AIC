# Loop Report #036 — REJECTED

**Hypothesis:** Audio-Tower-Only LoRA adaptation (CLAnSR ICASSP 2025) restricting PEFT to acoustic encoder projections with 100% frozen decoder  
**Decision Reason:** Dev did not improve (delta WER=+0.00%, delta EM=+0.00%) or Train regressed  
**Timestamp:** 2026-11-07 01:53:23  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **0.99%** | **1.51%** | `+0.00%` | **$\le$ 1.50%** |
| **CER** | **0.86%** | **1.07%** | `+0.00%` | **$\le$ 1.00%** |
| **Sentence EM** | **92.00%** | **87.63%** | `+0.00%` | **$\ge$ 95.00%** |
| **Latency** | 0.367s | 0.350s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
