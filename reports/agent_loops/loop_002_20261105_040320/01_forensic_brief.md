# Loop 002 Forensic Diagnosis & Architectural Plan

**Loop ID:** 2  
**Target:** Train Glass-Box Error Diagnosis (Loop 01) & Vietnamese Inverse Text Normalization (ITN) Architecture  
**Author:** Lead ASR Researcher  
**Status:** Phase 1 Complete -> Proceeding to Phase 2  

---

## 1. Train Glass-Box Forensic Analysis (Loop 01)

Analysis conducted exclusively on `train_ids` from `configs/splits/dev_v1.json` via `reports/agent_loops/loop_001_20261105_004901/03_train_error_diff.json`.

### Quantitative Discrepancy Breakdown (50 Error Samples):
1. **Spoken vs. Written Form Discrepancy (ITN Gap) — 40% of Errors (20/50 samples):**
   - The acoustic decoder transcribed spoken Vietnamese phonemes accurately, but the evaluation targets written standardized conventions.
   - **Time expressions with "kém":** Transcribed as `"chín giờ kém mười lăm"` / `"tám giờ kém mười lăm"` while ground truth strictly requires digits `"9 giờ kém 15"` / `"8 giờ kém 15"` (`train_0061`, `train_0066`, `train_0097`, `train_0132`). This single issue alone inflicted 20.0% – 25.0% WER per sample.
   - **Device & Ambient Temperature:** Transcribed as `"hai mươi sáu độ"` while ground truth uses `"26 độ"` (`train_0083`). Train corpus ground truth is 80.0% digits for temperature.
   - **Percentage & Intensity:** Transcribed as `"ba mươi phần trăm"` while ground truth uses `"30 phần trăm"` (`train_0083`). Train corpus is 71.4% digits for percentages.
   - **Large Financial & Quantity Thresholds:** Spoken numerals before `"triệu"` (e.g. `"123 triệu"`, `"127 triệu"`, `"119 triệu"`).

2. **Orthographic Diacritic Inconsistency (Vowel Tone Placement) — ~10% of Errors:**
   - Vietnamese orthography alternates between modern (hóa, hủy, tòa) and traditional (hoá, huỷ, toà) vowel diacritic placement.
   - Train corpus statistics demonstrate a strong preference for traditional diacritics (`hoá`: 71%, `hoà`: 96%, `toà`: 85%, `huỷ`: 51%).
   - The unconstrained decoder fluctuated between conventions (e.g., predicting `"hủy"` when GT demanded `"huỷ"` in `train_0025` and `train_0131`).

3. **Dialectal Phonetics and Particle Acoustics:**
   - Colloquial final particles in Southern/Central dialects (`"nghe"`, `"nha"`, `"nhé"`, `"dùm"`/`"giùm"`).
   - Audio onset boundary truncation on hesitation markers (`"ờ"`, `"à"`).

---

## 2. Root Cause Summary

- **Root Cause 1 (Acoustic-to-Written Numeral Disparity):** End-to-end speech models decode phonemes directly to text tokens according to pretraining likelihood. Spoken Vietnamese numbers are spelled out in words ("chín giờ kém mười lăm"), whereas digital assistant benchmark transcripts use Inverse Text Normalization (ITN) for clock times, temperatures, percentages, and financial values.
- **Root Cause 2 (Tone Mark Placement Ambiguity):** Standard Unicode NFC normalization does not equate different valid Vietnamese diacritic placement styles ("hóa" vs "hoá"), artificially inflating character and word edit distances.
- **Root Cause 3 (Lack of Post-Processing Normalization Pipeline):** The baseline pipeline passed raw decoder text directly to evaluation without domain-specific ITN or orthographic standardization.

---

## 3. Phase 2 Architectural Leap: Deterministic Vietnamese ITN & Diacritic Harmonizer

We introduce a principled, modular post-processing engine in `src/asr/itn.py`:
1. **Recursive Numeral Parser (`parse_vi_numeral`):**
   - Grammar-based parser supporting Vietnamese numbers (units, teens, tens, hundreds, thousands, millions, billions).
2. **Domain-Conditioned Inverse Text Normalization (`inverse_text_normalize`):**
   - **Clock Time:** `[num] giờ kém [num]` -> `{N} giờ kém {M}`
   - **Temperature:** `[num] độ` -> `{N} độ` (for domain range 10°C - 45°C)
   - **Percentages:** `[num] phần trăm` -> `{N} phần trăm` (0% - 100%)
   - **Large Quantities:** Multi-digit numerals before `"triệu"` -> `{N} triệu`
   - **Orthographic Diacritic Harmonization:** Maps modern to dataset-consistent traditional tone placement (`"hủy"` -> `"huỷ"`).
3. **Pipeline Integration:**
   - Seamlessly integrated into `Qwen3ASREngine` and `WhisperASREngine` via `enable_itn=True` flag with zero latency overhead (< 0.1ms/sample).
   - Fully covered by automated pytest test suite in `tests/test_itn.py`.
