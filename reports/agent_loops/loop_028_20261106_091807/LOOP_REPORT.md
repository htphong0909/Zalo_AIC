# Loop Report #028 — REJECTED

**Hypothesis:** Asymmetric VAD margin geometry (200ms pre-speech margin, 120ms post-speech margin at -35dB threshold) captures low-energy voice onsets without capturing trailing exhalations  
**Decision Reason:** Dev did not improve (delta WER=+0.07%, delta EM=+0.34%) or Train regressed  
**Timestamp:** 2026-11-06 09:28:01  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **1.03%** | **1.59%** | `+0.07%` | **$\le$ 1.50%** |
| **CER** | **0.90%** | **1.10%** | `+0.02%` | **$\le$ 1.00%** |
| **Sentence EM** | **92.00%** | **87.63%** | `+0.34%` | **$\ge$ 95.00%** |
| **Latency** | 0.370s | 0.356s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
