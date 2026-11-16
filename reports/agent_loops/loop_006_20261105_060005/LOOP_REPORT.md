# Loop Report #006 — REJECTED

**Hypothesis:** Orthographic tone harmonization (hoà/hoá/thuỷ) with hallucination suppression (condition_on_prev_tokens=False) and acoustic onset padding  
**Decision Reason:** Dev did not improve (delta WER=+0.42%, delta EM=-6.36%) or Train regressed  
**Timestamp:** 2026-11-05 06:34:18  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **1.72%** | **2.16%** | `+0.42%` | **$\le$ 3.00%** |
| **CER** | **1.50%** | **1.63%** | `+0.43%` | **$\le$ 3.00%** |
| **Sentence EM** | **82.67%** | **79.26%** | `-6.36%` | **$\ge$ 80.00%** |
| **Latency** | 1.879s | 1.810s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
