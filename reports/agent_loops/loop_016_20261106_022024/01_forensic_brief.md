# Loop #16 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Executive Summary & Baseline State
- **Prior Best Accepted Loop:** Loop #14 (Dev WER: 1.56%, Dev CER: 1.05%, Dev EM: 86.96%, Train WER: 1.02%, Train EM: 92.00%, Latency: 0.338s).
- **Target Definition of Done (DoD):** Dev WER $\le$ 1.50%, Dev CER $\le$ 1.00%, Dev EM $\ge$ 95.00%, Latency $\le$ 0.50s.
- **Current Gap to Primary Milestone:** Exactly 0.06% Dev WER (~1-2 word corrections out of 299 Dev utterances).
- **Loop #15 Post-Mortem:** In Loop #15, tightening silence trim to -28 dB and increasing repetition penalty to 1.10 clipped subtle phonemes and penalized valid repeats on Dev (Dev EM dropped to 86.29%). Loop #14 remains the verified golden acoustic baseline.

---

## 2. Train Glass-Box Forensic Diagnosis (12 Remaining Errors & Normalization Discrepancies)

A systematic forensic analysis of the 12 remaining errors in `reports/agent_loops/loop_014_20261106_012521/03_train_error_diff.json` and the 1,246 Train split transcripts revealed three key architectural and text normalization bottlenecks in `src/asr/itn.py`:

### Root Cause 1: Spoken Colloquial Numbering & Contracted Tens Omission in Numeral Parsing
- **Manifestation:**
  - In conversational Vietnamese (especially Southern dialects), compound numerals frequently elide the formal particle `"mươi"` before units: `"hai lăm"` (25), `"ba lăm"` (35), `"bốn lăm"` (45), `"hai mốt"` (21), `"hai tư"` (24).
  - In `src/asr/itn.py`, `_parse_under_thousand` and numeral regexes (`_NUM_DO`, `_NUM_PCT`, `_NUM_KEM_AFTER`) strictly expected the formal form `"mươi"` (`"hai mươi lăm"`, `"ba mươi"`).
  - As a result, `parse_vi_numeral("hai lăm")` returned `None`.
  - Spoken assistant commands like `"hai lăm độ"`, `"bốn lăm phần trăm"`, and `"chín giờ kém hai lăm"` or `"kém năm"` failed inverse text normalization entirely and remained in spoken words, conflicting with ground truth digit formatting (`25 độ`, `45 phần trăm`, `9 giờ kém 25`, `9 giờ kém 5`).

### Root Cause 2: Incomplete Modern Orthographic Tone Harmonization & Truncated Financial Million Ranges
- **Manifestation:**
  - Modern Vietnamese orthography (QĐ 1983 Bộ Giáo dục & SGK cải cách) harmonizes diphthongs with open glides (`oa`, `oe`, `uy`) by placing tones on the nucleus vowel (`hoà`, `hoá`, `xoá`, `khoá`, `toà`, `huỷ`).
  - Loop #7 introduced `_TONE_HARMONIZATION_MAP` for `hòa`, `hóa`, `xóa`, `hủy`, but omitted `khóa -> khoá` (34 khoá vs 25 khóa in Train) and `tòa -> toà` (11 toà vs 2 tòa).
  - In financial quantities before `"triệu"`, `_NUM_TRIEU` strictly restricted conversion to `val >= 10`. However, ground truth analysis reveals `"2 triệu"` is formatted as digits in 12 out of 15 instances in Train (80% digit preference), and `_NUM_TRIEU` completely omitted teen quantities (`mười một triệu` through `mười chín triệu`).

### Root Cause 3: Punctuation Spacing Glitches & Entity Casing Variance
- **Manifestation:**
  - Autoregressive beam/greedy decoding can occasionally produce spurious leading whitespace before punctuation marks (`"cửa nhé ."` -> `"cửa nhé."`), and lowercase/mixed casing of core assistant entities (`"sjc"` -> `"SJC"`, `"sms"` -> `"SMS"`, `"zalo"` -> `"Zalo"`).
  - Across the Train corpus, `SJC` (20/20, 100%), `SMS` (61/61, 100%), and `Zalo` (50/50, 100%) follow strict canonical casing. Standardizing canonical acronym casing in ITN ensures robust entity extraction and sentence formatting.

---

## 3. Phase 2 Architectural & Normalization Implementation Plan

1. **`src/asr/itn.py`:**
   - **Colloquial Numeral Parsing:** Expand `_parse_under_thousand` for 2-token sequences `(t0, t1)` where `t0 in _DIGITS` and `t1 in ("mốt", "hai", "ba", "bốn", "tư", "lăm", "nhăm", "năm", "sáu", "bảy", "bẩy", "tám", "chín")` to return `_DIGITS[t0] * 10 + _DIGITS[t1]` (`"hai lăm"` $\to$ 25, `"bốn lăm"` $\to$ 45, `"hai tư"` $\to$ 24).
   - **Time Expressions with `kém`:** Add `năm`, `lăm`, and contracted forms (`hai lăm`) to `_NUM_KEM_AFTER` so `"chín giờ kém năm"` $\to$ `"9 giờ kém 5"` and `"chín giờ kém hai lăm"` $\to$ `"9 giờ kém 25"`.
   - **Ambient Temperature & Intensity Percentage:** Expand `_NUM_DO` and `_NUM_PCT` to support contracted tens (`hai lăm độ` $\to$ `25 độ`, `bốn lăm phần trăm` $\to$ `45 phần trăm`).
   - **Financial Quantities with `triệu`:** Expand `_NUM_TRIEU` to cover teens (`mười một` ... `mười chín`) and single-digit `hai` (`2 triệu` $\to$ `2 triệu`).
   - **Comprehensive Orthographic Tone Harmonization:** Add `khóa -> khoá`, `tòa -> toà`, `thỏa -> thoả`, `tỏa -> toả`, `thủy -> thuỷ`, `tùy -> tuỳ`, `họa -> hoạ`, `khỏe -> khoẻ`, `lòe -> loè`, `lóa -> loá` to `_TONE_HARMONIZATION_MAP`.
   - **Canonical Brand & Acronym Casing:** Add `SJC`, `SMS`, `Zalo` casing normalizer.
   - **Punctuation Spacing Cleanup:** Remove space before punctuation `re.sub(r"\s+([,.:?!])", r"\1", s)` and collapse repeated whitespace.

2. **`tests/test_itn.py`:**
   - Add comprehensive regression unit tests covering contracted tens (`hai lăm`, `bốn lăm`, `ba tư`), clock `kém` with single digits, colloquial temperatures, financial millions, tone harmonization, and acronym casing.

---

## 4. Verification Gate
- Unit tests (`python -m pytest tests/ -q`): Must pass 100% green.
- Anti-snoop invariant: Zero Dev access.
- Zero local compute mandate: Strictly lightweight unit tests only.
