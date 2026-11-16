# Loop Report #015 — REJECTED

**Hypothesis:** Dynamic boundary silence trimming (-28.0 dB with 150ms safety collar) combined with calibrated autoregressive repetition penalty (1.10) suppresses background microphone hum hallucinations and breaks entity KV-cache attention latching.  
**Decision Reason:** Dev did not improve (delta WER=+0.00%, delta EM=-0.67%) or Train regressed  
**Timestamp:** 2026-11-06 02:19:27  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **0.99%** | **1.56%** | `+0.00%` | **$\le$ 1.50%** |
| **CER** | **0.87%** | **1.04%** | `-0.01%` | **$\le$ 1.00%** |
| **Sentence EM** | **92.00%** | **86.29%** | `-0.67%` | **$\ge$ 95.00%** |
| **Latency** | 0.350s | 0.336s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
