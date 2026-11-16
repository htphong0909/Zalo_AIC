# Loop Report #040 — REJECTED

**Hypothesis:** Multi-Dialect Cross-Attention & Encoder LoRA adaptation (r=8, alpha=16) aligns acoustic latent representations to standard Vietnamese text  
**Decision Reason:** Dev did not improve (delta WER=+0.00%, delta EM=+0.00%) or Train regressed  
**Timestamp:** 2026-11-07 09:20:08  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **0.77%** | **1.50%** | `+0.00%` | **$\le$ 1.50%** |
| **CER** | **0.59%** | **1.03%** | `+0.00%` | **$\le$ 1.00%** |
| **Sentence EM** | **92.67%** | **86.96%** | `+0.00%` | **$\ge$ 95.00%** |
| **Latency** | 0.366s | 0.355s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
