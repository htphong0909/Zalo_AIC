# Loop Report #032 — ACCEPTED

**Hypothesis:** Symmetric 170ms VAD collar (+1 encoder frame buffer at -35dB clean SNR) captures speech onset VOT and trailing vowel release  
**Decision Reason:** Dev WER improved (-0.01%) or EM improved (+0.34%) without train collapse  
**Timestamp:** 2026-11-06 11:01:53  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **0.99%** | **1.51%** | `-0.01%` | **$\le$ 1.50%** |
| **CER** | **0.86%** | **1.07%** | `-0.01%` | **$\le$ 1.00%** |
| **Sentence EM** | **92.00%** | **87.63%** | `+0.34%` | **$\ge$ 95.00%** |
| **Latency** | 0.365s | 0.353s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
