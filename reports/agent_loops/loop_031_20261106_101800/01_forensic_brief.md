# Loop #31 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Executive Summary & Baseline State
- **Prior Best Accepted Loop:** Loop #25 (Dev WER: 1.52%, Dev CER: 1.08%, Dev EM: 87.29%, Train WER: 0.99%, Train EM: 92.00%, Latency: ~0.355s).
- **Target Definition of Done (DoD):** Dev WER <= 1.50%, Dev CER <= 1.00%, Dev EM >= 95.00%, Latency <= 0.50s.
- **Current Gap to Milestone 1:** Only 0.02% Dev WER (a single word error across the entire blind evaluation corpus separates current performance from the <= 1.50% milestone).
- **Acoustic vs Lexical Boundary Analysis:**
  - Signal-level acoustic preconditioning (60ms bilateral collar padding, -35 dB silence trimming with 150ms safety margin, peak normalization to 0.95, 1.02 repetition penalty) has minimized acoustic boundary truncation.
  - The remaining error budget is dominated by subword lexical fragmentation in foreign loanwords and ride-hailing syntactic numeral mismatches.

---

## 2. Train Glass-Box Forensic Diagnosis: Lexical Compound Splitting & Vehicular Collocations

Detailed examination of the LifeMate vocabulary distribution and Train Glass-Box diagnostic errors reveals two systematic sources of avoidable word substitutions and deletions:

1. **Foreign Loanword Compound Fragmentation:**
   - In standard Vietnamese ASR normalization (`asr_metrics.normalize_vi_text`), punctuation is stripped via `re.compile(r"[^\w\s]")`, replacing hyphens with spaces (`e-mail` -> `e mail`, `wi-fi` -> `wi fi`).
   - The LifeMate ground-truth corpus establishes a 100% solid compound convention for common technology loanwords:
     - `email`: 17 occurrences vs `e-mail`: 0 occurrences (100% solid compound)
     - `wifi`: 2 occurrences vs `wi-fi`: 0 occurrences (100% solid compound)
     - `online`: 23 occurrences vs `on-line`: 0 occurrences (100% solid compound)
     - `offline`: 3 occurrences vs `off-line`: 0 occurrences (100% solid compound)
     - `autopay` and `coworking` follow the identical solid compound standard.
   - When PhoWhisper's English/multilingual subword tokenizer decodes acoustic loanwords, it frequently outputs hyphenated or space-separated forms (`e-mail`, `e mail`, `wi-fi`). Under evaluation, an utterance predicted as `e mail` against reference `email` incurs a double penalty (1 substitution + 1 insertion = 2 word errors) and unconditionally destroys Sentence Exact Match.

2. **Ride-Hailing Vehicular Collocation Numeral Discrepancy:**
   - LifeMate domain commands frequently trigger vehicle bookings ("đặt xe", "gọi xe").
   - In the ground-truth corpus, `xe bốn chỗ` occurs 12 times while `xe 4 chỗ` occurs 0 times (100% written numeral collocation).
   - In contrast, expressions with `ô tô` exhibit both digit and word variants. Unconditioned Whisper inference frequently produces digit `4` in `xe 4 chỗ`, inducing a recurring substitution error against `xe bốn chỗ`.

---

## 3. Phase 2 Architectural & Decoding Refactoring: Loanword Fusion & Entity Standardization

In `src/asr/itn.py`, we augment the deterministic Inverse Text Normalization engine with loanword compound fusion and vehicular collocation standardization passes:

### Code Modifications:
1. **Foreign Loanword Solid Compound Fusion (`src/asr/itn.py`):**
   - Implemented regex patterns targeting hyphenated and spaced compound variants:
     - `\be[\s\-]+mail\b` -> `email` (and `\bE[\s\-]+mail\b` -> `Email`)
     - `\bwi[\s\-]+fi\b` -> `wifi` (and `\bWi[\s\-]+fi\b` -> `Wifi`)
     - `\bon[\s\-]+line\b` -> `online` (and `\bOn[\s\-]+line\b` -> `Online`)
     - `\boff[\s\-]+line\b` -> `offline` (and `\bOff[\s\-]+line\b` -> `Offline`)
     - `\bauto[\s\-]+pay\b` -> `autopay`
     - `\bco[\s\-]+working\b` -> `coworking`
   - Added case-preserving substitution helper `_sub_loanword` preserving uppercase acronyms (`EMAIL`, `WIFI`), capitalized titles (`Email`, `Wifi`), and lowercase defaults.

2. **Ride-Hailing Entity Standardization (`src/asr/itn.py`):**
   - Implemented targeted vehicular collocation standardization:
     - `\bxe\s+4\s+chỗ\b` -> `xe bốn chỗ`
     - `\bXe\s+4\s+chỗ\b` -> `Xe bốn chỗ`
   - Preserves non-matching vehicle types (`xe 7 chỗ`, `ô tô 4 chỗ`) without disruption.

3. **Golden Rule Invariant Preservation:**
   - Strictly preserved all existing rules:
     - `_TONE_PATTERNS` (`hủy` -> `huỷ`, `hòa` -> `hoà`, etc.)
     - `_RE_KEM` (`chín giờ kém mười lăm` -> `9 giờ kém 15`)
     - `_RE_DO` (`hai mươi sáu độ` -> `26 độ`)
     - `_RE_PCT` (`ba mươi phần trăm` -> `30 phần trăm`)
     - `_RE_TRIEU` (`một trăm hai mươi ba triệu` -> `123 triệu`)
     - `_RE_DIALECT_RUT` (`gút tiết kiệm` -> `rút tiết kiệm`)

4. **Unit Test Suite Gate (`tests/test_asr.py`):**
   - Added `test_itn_loanword_compound_fusion_and_vehicle_standardization` verifying all compound variations, casing, vehicle entities, and golden rule preservation.
   - Pytest status: 69/69 passed (100% green).

### Strict Protocol Compliance:
- **Sealed Blind Dev Invariant:** Zero access or snooping of Dev set.
- **Local Zero-Compute Mandate:** Lightweight unit testing only (`pytest tests/ -q` = 69 passed in 23.55s). No local GPU or heavy inference.
