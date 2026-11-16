# Loop Report #002 — REJECTED

**Hypothesis:** Deterministic Vietnamese Inverse Text Normalization (ITN) and orthographic diacritic harmonization  
**Decision Reason:** Dev did not improve (delta WER=+15.82%, delta EM=-22.74%) or Train regressed  
**Timestamp:** 2026-11-05 04:26:31  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **29.32%** | **28.24%** | `+15.82%` | **$\le$ 3.00%** |
| **CER** | **22.71%** | **19.88%** | `+10.89%` | **$\le$ 3.00%** |
| **Sentence EM** | **2.00%** | **2.01%** | `-22.74%` | **$\ge$ 80.00%** |
| **Latency** | 0.402s | 0.333s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
