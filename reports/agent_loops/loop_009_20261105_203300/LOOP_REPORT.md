# Loop Report #009 — REJECTED

**Hypothesis:** Acoustic domain LoRA adaptation (r=16, alpha=32) on PhoWhisper Large targeting attention projections  
**Decision Reason:** Dev did not improve (delta WER=+0.07%, delta EM=-0.33%) or Train regressed  
**Timestamp:** 2026-11-05 21:16:21  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **0.59%** | **1.68%** | `+0.07%` | **$\le$ 1.50%** |
| **CER** | **0.47%** | **1.16%** | `+0.09%` | **$\le$ 1.00%** |
| **Sentence EM** | **92.67%** | **86.96%** | `-0.33%` | **$\ge$ 95.00%** |
| **Latency** | 0.361s | 0.344s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
