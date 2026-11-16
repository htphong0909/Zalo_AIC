# Loop Report #010 — REJECTED

**Hypothesis:** Bilateral acoustic temporal padding (100ms pre-roll + 200ms post-roll buffer) on PhoWhisper Large to eliminate speech boundary truncation and final syllable dropouts  
**Decision Reason:** Dev did not improve (delta WER=+0.04%, delta EM=+0.00%) or Train regressed  
**Timestamp:** 2026-11-05 23:23:18  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **1.17%** | **1.65%** | `+0.04%` | **$\le$ 1.50%** |
| **CER** | **0.96%** | **1.11%** | `+0.04%` | **$\le$ 1.00%** |
| **Sentence EM** | **90.67%** | **87.29%** | `+0.00%` | **$\ge$ 95.00%** |
| **Latency** | 0.375s | 0.366s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
