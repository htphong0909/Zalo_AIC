# Loop Report #034 — REJECTED

**Hypothesis:** Generic Levenshtein phonetic-bounded alignment and flanked single-syllable recovery with strict entity immunity (PY-GEC ICASSP 2025)  
**Decision Reason:** Dev did not improve (delta WER=+6.88%, delta EM=-65.56%) or Train regressed  
**Timestamp:** 2026-11-07 00:37:18  

## 1. Dual-Referee Evaluation Results

| Metric | Glass-Box Train (150 samples) | Sealed Blind Dev (299 samples) | Delta vs Best | Target DoD |
| :--- | :---: | :---: | :---: | :---: |
| **WER** | **6.70%** | **8.39%** | `+6.88%` | **$\le$ 1.50%** |
| **CER** | **4.40%** | **5.24%** | `+4.17%` | **$\le$ 1.00%** |
| **Sentence EM** | **25.33%** | **22.07%** | `-65.56%` | **$\ge$ 95.00%** |
| **Latency** | 0.370s | 0.353s | - | $\le$ 0.500s |

## 2. Permanent Artifacts
- Forensic Brief: [`01_forensic_brief.md`](./01_forensic_brief.md)
- Code Patch: [`02_code_changes.patch`](./02_code_changes.patch)
- Modification Summary: [`02_modification_summary.json`](./02_modification_summary.json)
- Metrics Snapshot: [`03_kpi_metrics.json`](./03_kpi_metrics.json)
- Train Error Diff: [`03_train_error_diff.json`](./03_train_error_diff.json)
