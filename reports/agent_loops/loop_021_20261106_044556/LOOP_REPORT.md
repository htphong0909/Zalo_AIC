# Loop Report #021 — REJECTED

**Hypothesis:** Isolated neutral topic-noun prompt conditioning while strictly preserving traditional tone harmonization in ITN to capture the +1.33% EM gain without orthographic penalties.  
**Decision Reason:** Dev did not improve (delta WER=+0.19%, delta EM=-1.68%) or Train regressed  
**Timestamp:** 2026-11-06 05:22:27  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **0.95%** | **1.74%** | `+0.19%` | **$\le$ 1.50%** |
| **CER** | **0.86%** | **1.24%** | `+0.15%` | **$\le$ 1.00%** |
| **Sentence EM** | **92.00%** | **85.28%** | `-1.68%` | **$\ge$ 95.00%** |
| **Latency** | 0.367s | 0.359s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
