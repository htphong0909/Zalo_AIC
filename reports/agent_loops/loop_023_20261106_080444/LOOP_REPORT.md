# Loop Report #023 — REJECTED

**Hypothesis:** Constrained 2-beam search decoding (num_beams=2, early_stopping=True) with repetition penalty (1.03) over calibrated acoustic preprocessing  
**Decision Reason:** Dev did not improve (delta WER=+0.06%, delta EM=+0.00%) or Train regressed  
**Timestamp:** 2026-11-06 08:22:59  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **1.11%** | **1.61%** | `+0.06%` | **$\le$ 1.50%** |
| **CER** | **0.94%** | **1.17%** | `+0.08%` | **$\le$ 1.00%** |
| **Sentence EM** | **91.33%** | **86.96%** | `+0.00%` | **$\ge$ 95.00%** |
| **Latency** | 0.824s | 0.787s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
