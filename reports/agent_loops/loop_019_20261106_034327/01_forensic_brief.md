# Loop #19 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Executive Summary & Baseline State
- **Prior Best Accepted Loop:** Loop #17 (Dev WER: 1.55%, Dev CER: 1.09%, Dev EM: 86.96%, Train WER: 0.99%, Train EM: 92.00%, Latency: 0.348s).
- **Target Definition of Done (DoD):** Dev WER <= 1.50%, Dev CER <= 1.00%, Dev EM >= 95.00%, Latency <= 0.50s.
- **Current Gap to Primary Milestone:** Exactly 0.05% Dev WER (~1 word correction across the 299-sample Sealed Dev set).
- **Loop #18 Post-Mortem:**
  - Loop #18 tested asymmetric collar padding (60ms pre-roll / 90ms post-roll) and DC offset removal.
  - Evaluation confirmed identical performance to Loop #17 (Dev WER 1.55%, Dev EM 86.96%, Train WER 0.99%, Train EM 92.00%).
  - The Pareto Gate correctly rejected Loop #18 to prevent unnecessary complexity. Loop #17 remains our gold baseline.

---

## 2. Train Glass-Box Forensic Diagnosis (12 Remaining Errors in Loop #17)

Detailed inspection of the 12 error utterances in `reports/agent_loops/loop_017_20261106_030003/03_train_error_diff.json` and corpus-wide quantitative analysis on the Train Glass-Box revealed three actionable root causes:

### Root Cause 1: Spurious Punctuation and Enclitic Comma Artifacts
- **Empirical Corpus Evidence:**
  - In conversational Vietnamese LifeMate assistant commands, polite and modal particles (`nhé`, `nha`, `ạ`, `đi`, `nhỉ`, `nè`, `hen`) are enclitics.
  - Quantitative analysis of the 1,246 Train split transcripts revealed an overwhelming **852 to 2 ratio (>99.7%)** of polite particles appearing preceded by a space **without** a comma.
  - However, Whisper's multilingual BPE pretraining frequently introduces European-style commas before vocatives and tags (e.g. `", nhé."`, `", nha."`, `", ạ."`).
  - Furthermore, quotation marks (`"`, `“`, `”`, `'`, `‘`, `’`) appear **0 times** across ground truth annotations, yet Whisper occasionally introduces quotation mark tokens around app names, entity slots, or message bodies.
- **Remediation:**
  - Strip quotation marks in `_sanitize_hyp`: `clean_text = re.sub(r'["“”\'‘’]', "", clean_text)`.
  - Strip spurious commas before polite/modal particles in `_sanitize_hyp`: `clean_text = re.sub(r",\s*(nhé|nha|ạ|đi|nhỉ|nè|hen)\b", r" \1", clean_text, flags=re.IGNORECASE)`.

### Root Cause 2: Temperature Unit Symbol Discrepancy (`độ C` / `°C` vs `độ`)
- **Empirical Corpus Evidence:**
  - Across the 1,246 Train split utterances, ambient and appliance temperature is referenced 59 times.
  - **All 59 occurrences strictly use `"độ"`** (e.g. `"26 độ"`, `"27 độ"`). There are **0 occurrences** of `"độ C"` or `"°C"`.
  - Multilingual Whisper models frequently transcribe temperature expressions as `"26 độ C"` or `"26°C"`. Under ASR evaluation (`normalize_vi_text`), `"26 độ C"` decomposes into five words (`"26 độ c"`), leaving an extraneous `"c"` error.
- **Remediation:**
  - In `src/asr/itn.py`, standardize all temperature unit representations (`\d+\s*(?:°\s*[cC]|°|độ\s*[cC])\b` -> `\1 độ`).
  - Expand `_RE_DO` numeral matching to recognize spoken temperature followed by `độ C` / `°C` and map to canonical `<val> độ`.

### Root Cause 3: Autoregressive Decoder Latching & Attention Drift
- **Acoustic Profiling & Decoding Cause:**
  - In `train_0105`, ground truth `"...30 phút... trước lúc xe đón 15 phút."` was decoded as `"...30 phút... trắc lúc đón xe 30 phút."`, latching onto the earlier numerical token `"30 phút"`.
  - Unconstrained greedy generation without explicit segment-level constraints permits cross-attention KV-cache latching.
- **Remediation:**
  - In `_build_gen_kwargs`:
    - Set `"condition_on_prev_tokens": False` to strictly isolate single-utterance decoding from prior context bleed.
    - Set `"temperature": 0.0` to enforce strictly deterministic decoding.
    - Set `"compression_ratio_threshold": 2.4` to activate standard Whisper repetition ratio monitoring.

---

## 3. Phase 2 Architectural & Decoding Implementation Plan

1. **`src/asr/whisper_engine.py`:**
   - Update `_sanitize_hyp`:
     - Strip quotation marks (`"`, `“`, `”`, `'`, `‘`, `’`).
     - Standardize comma-separated polite particles: `, <particle>` -> ` <particle>`.
   - Update `_build_gen_kwargs`:
     - Add `"condition_on_prev_tokens": False`.
     - Add `"temperature": 0.0`.
     - Add `"compression_ratio_threshold": 2.4`.

2. **`src/asr/itn.py`:**
   - Add temperature unit normalization in `inverse_text_normalize`:
     `s = re.sub(r"(\d+)\s*(?:°\s*[cC]|°|độ\s*[cC])\b", r"\1 độ", s)`
   - Expand `_RE_DO` regex to match optional Celsius suffix:
     `_RE_DO = re.compile(rf"\b({_NUM_DO})\s*(?:°\s*[cC]|°|độ\s*[cC]|độ)\b", re.IGNORECASE)`

3. **`tests/test_asr.py` & `tests/test_itn.py`:**
   - Add unit tests for polite particle comma removal, quote stripping, and temperature unit normalization.
   - Verify 100% green on `pytest tests/ -q`.

---

## 4. Verification Gate
- Unit tests (`python -m pytest tests/ -q`): Must pass 100% green.
- Anti-snoop invariant: Zero Dev access.
- Local zero-compute mandate: Quick mock unit tests only.
