# Loop Report #004 — REJECTED

**Hypothesis:** PhoWhisper Large 80-channel feature alignment with SDPA and batched inference latency acceleration  
**Decision Reason:** Dev did not improve (delta WER=+0.00%, delta EM=+0.00%) or Train regressed  
**Timestamp:** 2026-11-05 05:30:29  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **1.44%** | **1.74%** | `+0.00%` | **$\le$ 3.00%** |
| **CER** | **1.24%** | **1.20%** | `+0.00%` | **$\le$ 3.00%** |
| **Sentence EM** | **88.67%** | **85.62%** | `+0.00%` | **$\ge$ 80.00%** |
| **Latency** | 0.358s | 0.342s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
