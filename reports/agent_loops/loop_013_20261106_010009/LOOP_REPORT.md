# Loop Report #013 — REJECTED

**Hypothesis:** Grounded conversational command prompting with conditional syntax anchors and comprehensive multi-domain entity coverage eliminates prefix verb distortion and sentence-final polite particle divergence.  
**Decision Reason:** Dev did not improve (delta WER=+0.12%, delta EM=-0.67%) or Train regressed  
**Timestamp:** 2026-11-06 01:24:48  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **1.22%** | **1.69%** | `+0.12%` | **$\le$ 1.50%** |
| **CER** | **1.05%** | **1.21%** | `+0.12%` | **$\le$ 1.00%** |
| **Sentence EM** | **87.33%** | **86.62%** | `-0.67%` | **$\ge$ 95.00%** |
| **Latency** | 0.487s | 0.488s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
