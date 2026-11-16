# Loop Report #016 — REJECTED

**Hypothesis:** Domain-conditioned Vietnamese ITN and orthographic harmonization engine resolving colloquial contracted compound numerals (hai lăm, bốn lăm), expanded clock kém and financial ranges, modern nucleus tone harmonization (khoá, toà), and canonical acronym casing.  
**Decision Reason:** Dev did not improve (delta WER=+0.15%, delta EM=-2.01%) or Train regressed  
**Timestamp:** 2026-11-06 02:58:32  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **1.04%** | **1.71%** | `+0.15%` | **$\le$ 1.50%** |
| **CER** | **0.92%** | **1.16%** | `+0.11%` | **$\le$ 1.00%** |
| **Sentence EM** | **91.33%** | **84.95%** | `-2.01%` | **$\ge$ 95.00%** |
| **Latency** | 0.370s | 0.348s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
