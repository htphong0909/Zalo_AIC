# Loop Report #012 — ACCEPTED

**Hypothesis:** Linear peak amplitude normalization to 0.95 combined with conversational vocative prompt de-biasing  
**Decision Reason:** Dev WER improved (-0.03%) or EM improved (+0.33%) without train collapse  
**Timestamp:** 2026-11-06 00:59:25  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **0.99%** | **1.57%** | `-0.03%` | **$\le$ 1.50%** |
| **CER** | **0.88%** | **1.09%** | `+0.01%` | **$\le$ 1.00%** |
| **Sentence EM** | **92.00%** | **87.29%** | `+0.33%` | **$\ge$ 95.00%** |
| **Latency** | 0.477s | 0.453s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
