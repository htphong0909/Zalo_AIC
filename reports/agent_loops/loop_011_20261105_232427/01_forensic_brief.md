# Loop #11 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Executive Summary & Baseline State
- **Prior Best Accepted Loop:** Loop #07 (Dev WER: 1.61%, Dev CER: 1.07%, Dev EM: 87.29%, Train WER: 1.16%, Train EM: 90.67%)
- **Target Definition of Done (DoD):** Dev WER $\le$ 1.50%, Dev CER $\le$ 1.00%, Dev EM $\ge$ 95.00%, Latency $\le$ 0.50s
- **Optimization Objective:** Eliminate the remaining ~0.11% WER gap to cross the $\le$ 1.50% DoD threshold while preserving deterministic reproducibility and zero dev set leakage.

---

## 2. Train Glass-Box Forensic Diagnosis (150 Samples)

Through systematic waveform and transcript analysis on the 150 Train Glass-Box samples (`train_ids` in `configs/splits/dev_v1.json`), three distinct root causes were identified:

### Root Cause 1: Asymmetric Acoustic Boundary Truncation
- **Acoustic Phenomenon:** In Loop #07, 40ms pre-roll padding was introduced, which successfully stabilized leading consonant onsets. However, post-roll padding was left at 0.0ms.
- **Waveform Evidence:** In samples like `train_0083` and `train_0006`, trailing silence before the audio boundary is only 11.9ms (190 samples at 16kHz). Because Whisper's STFT feature extraction applies a 25ms Hann window that tapers to 0 at the window boundaries, final syllables spoken close to the file boundary suffer from severe acoustic attenuation.
- **Why Loop #10 Failed:** Loop #10 attempted 100ms pre-roll + 200ms post-roll padding. 200ms of dead silence (3,200 samples) created an unnaturally wide silence collar that caused Whisper's autoregressive decoder attention to drift and hallucinate.
- **Solution:** A tight, symmetric bilateral collar of **40ms pre-roll + 40ms post-roll padding** (640 samples / 4 frames each), providing edge taper protection without creating decoder drift zones.

### Root Cause 2: Prompt Prefix Bleed and Name Collision
- **Error Manifestation:** In `train_0019`:
  - Ground Truth: `"Chị ơi, giúp tôi đặt phòng khách sạn ở Đà Lạt với nhé."`
  - Loop #07 Prediction: `"Chị Huyết, tôi cần đặt phòng khách sạn ở Đà Lạt với."`
- **Mechanism:** The prompt string contained `"Huy"`. In Vietnamese speech with acoustic reduction, Whisper's self-attention over `<|startofprev|>` tokens collided with the phoneme sequence, hallucinating `"Chị Huyết"`. Furthermore, meta-narrative prefix text `"Hệ thống trợ lý..."` can act as an autoregressive prior distorting initial utterance tokens.
- **Solution:** Cleaned and pruned the prompt prefix to focus strictly on operational intent keywords and core characters while removing collision-prone tokens (`Huy`).

### Root Cause 3: Spoken Southern Phonological Compound Deviation
- **Error Manifestation:** In `train_0087`:
  - Ground Truth: `"...không đủ cả hai khoản thì rút tiết kiệm ra nhé."`
  - Loop #07 Prediction: `"...không đủ cả hai khoản thì gút tiết kiệm ra nhé."`
- **Linguistic Root Cause:** In Southern Vietnamese colloquial phonology, initial rhotic /r/ is systematically merged into velar [g] (pronounced "gút tiết kiệm", "gút tiền"). In standard written Vietnamese, "gút" is only used for medical gout ("bệnh gút"); across the 4,042 train samples, "gút" appears exactly 0 times in valid transcripts.
- **Solution:** Integrated principled Inverse Text Normalization (ITN) compound recovery for financial dialect collocations: `r"\bgút\s+(tiết\s+kiệm|tiền)\b" -> r"rút \1"`.

---

## 3. Architectural & Decoding Implementations (Phase 2)

1. **`src/asr/whisper_engine.py`:**
   - Added `post_roll_pad_ms: float = 40.0` parameter to `WhisperASREngine.__init__`.
   - Updated `_pad_waveform()` to apply symmetric bilateral padding:
     ```python
     if len(waveform) > 0 and (self.pre_roll_pad_ms > 0 or self.post_roll_pad_ms > 0):
         pad_pre = int(round((self.pre_roll_pad_ms / 1000.0) * sr)) if self.pre_roll_pad_ms > 0 else 0
         pad_post = int(round((self.post_roll_pad_ms / 1000.0) * sr)) if self.post_roll_pad_ms > 0 else 0
         return np.pad(waveform, (pad_pre, pad_post), mode="constant", constant_values=0.0)
     ```
   - Streamlined `DEFAULT_WHISPER_PROMPT` to eliminate prefix narrative bleeding and token collision.

2. **`src/asr/itn.py`:**
   - Added regex pattern `_RE_DIALECT_RUT = re.compile(r"\bgút\s+(tiết\s+kiệm|tiền)\b", re.IGNORECASE)` to `inverse_text_normalize()` to recover dialectal financial compounds into standard orthography.

3. **`tests/test_itn.py` & `tests/test_asr.py`:**
   - Added unit test coverage for dialect compound recovery and bilateral padding waveform calculation.

---

## 4. Verification & Unit Test Gate Results
- **Unit Test Suite:** `python -m pytest tests/ -q` -> **74/74 passed (100% green)** in 61.8s.
- **Train Glass-Box (150 samples) Benchmark:**
  - Train WER: **1.14%** (improved from 1.16%, $\Delta = -0.02\%$)
  - Train CER: **0.96%** (improved from 0.97%, $\Delta = -0.01\%$)
  - Sentence EM: **91.33%** (improved from 90.67%, $\Delta = +0.66\%$)
  - Total Train Errors: Reduced from 14 to 13 (`train_0087` perfectly resolved to exact match).
  - Average Latency: **0.359s** per sample ($\le$ 0.500s DoD target satisfied).
