# Loop Report #042 — REJECTED

**Hypothesis:** Hyper-compact Dual-Tower Acoustic LoRA (r=4, alpha=8, dropout=0.1, lr=8e-5, 3 epochs, weight_decay=0.01) regularizes phonetic cross-attention to prevent speaker memorization while grounding regional dialect shifts  
**Decision Reason:** Dev did not improve (delta WER=+0.00%, delta EM=+0.00%) or Train regressed  
**Timestamp:** 2026-11-07 10:16:22  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **0.82%** | **1.50%** | `+0.00%` | **$\le$ 1.50%** |
| **CER** | **0.63%** | **1.03%** | `+0.00%` | **$\le$ 1.00%** |
| **Sentence EM** | **92.67%** | **86.96%** | `+0.00%` | **$\ge$ 95.00%** |
| **Latency** | 0.370s | 0.359s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
