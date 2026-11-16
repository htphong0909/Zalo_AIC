# Loop 006 Forensic Diagnosis & Architectural Refactoring Brief

**Loop ID:** 6  
**Hypothesis:** Orthographic tone harmonization (hoà/hoá/thuỷ) with hallucination suppression (condition_on_prev_tokens=False) and acoustic onset padding  
**Author:** Lead ASR & Deep Learning Researcher  
**Target:** Train Glass-Box Forensic Analysis & Precision Recovery  

---

## 1. Train Glass-Box Forensic Diagnosis (Analysis of Loop 003 Residual Errors)

Detailed inspection of the 17 remaining Train errors from Loop 003 (`reports/agent_loops/loop_003_20261105_043154/03_train_error_diff.json`) reveals three clear, systemic error mechanisms:

### A. Prefix Hallucination & Prompt Bleeding (`train_0115`):
- **Ground Truth:** `Nhắn Zalo cho Minh giúp chị là mai gửi mình file nhé.`
- **Prediction:** `thêm vào danh bạ. Nhắn Zalo cho Minh giúp chị là mai gửi mình file nhé.`
- **Acoustic & Decoder Origin:** When `initial_prompt` is supplied via `prompt_ids` with `condition_on_prev_tokens=True` (or model default), the decoder conditions subsequent decoding on the prompt context ("danh bạ, gọi điện..."). On short utterances or leading silences, this induces autoregressive bleeding of prompt keywords ("thêm vào danh bạ.") directly into the output hypothesis.
- **Remedy:** Explicitly enforce `condition_on_prev_tokens=False` in `gen_kwargs` to suppress continuation artifacts from preceding context or prompt tokens.

### B. Acoustic Onset Syllable Boundary Clipping (`train_0207`, `train_0006`):
- **`train_0207`:**
  - **Ground Truth:** `Tắt đèn phòng khách giúp em, rồi nhắn SMS cho Hương là tối nay ăn cơm nhà nhé.`
  - **Prediction:** `Phòng khách giúp em, rồi nhắn SMS cho Hương là tối nay ăn cơm nhà nhé.`
  - **Error:** Complete deletion of leading onset words `"Tắt đèn"`.
- **`train_0006`:**
  - **Ground Truth:** `Nếu giá mua vào vàng SJC hôm nay dưới 123 triệu thì 9 giờ sáng mai nhắc tôi đi mua nhé.`
  - **Prediction:** `Điêu giá thu mua vào vàng SJC...` (onset `"Nếu"` corrupted to `"Điêu"`).
- **Acoustic Origin:** Fast plosives and alveolar nasals at the absolute start of audio clips suffer from STFT/Hann window roll-off and filterbank boundary attenuation, leading to lost initial speech energy.
- **Remedy:** Prepend a 50ms leading silence padding buffer (800 samples at 16 kHz) prior to Mel spectrogram computation. Empirical validation on `train_0207` confirms 100% recovery of the missing `"Tắt đèn"` onset.

### C. Orthographic Tone Placement Divergence & Numeral Harmonization:
- **Tone Placement Divergence:**
  - Modern diacritic orthography places tones on the main vowel (e.g. `hòa`, `hóa`, `thủy`, `tòa`, `thỏa`, `xóa`, `khóa`), whereas Vietnamese conversational corpora overwhelmingly follow traditional orthography (`hoà`, `hoá`, `thuỷ`, `toà`, `thoả`, `xoá`, `khoá`).
  - Analysis on Train Glass-Box: `hoà` (54) vs `hòa` (2), `hoá` (69) vs `hóa` (28), `xoá` (22) vs `xóa` (1), `khoá` (35) vs `khóa` (25).
  - Normalizing to traditional orthography eliminates synthetic orthographic mismatches.
- **Ride-Hailing Seat Numeral Harmonization (`train_0105`, `train_0149`):**
  - Spoken ride-hailing requests interchangeably transcribe `"ô tô bốn chỗ"` and `"ô tô 4 chỗ"`.
  - Harmonizing `"ô tô bốn chỗ"` / `"ô tô 4 chỗ"` -> `"ô tô 4 chỗ"` aligns with standard ITN numeral formatting.

---

## 2. Root Cause Summary

1. **Acoustic Boundary Roll-Off:** Windowing functions at time $t=0$ attenuate early speech transients, causing onset phoneme deletion ("tắt đèn") or substitution ("nếu" -> "điêu").
2. **Decoder Prompt Bleeding:** Autoregressive language modeling in Whisper's cross-attention conditions generation on prompt prefix tokens when condition_on_prev_tokens is not suppressed.
3. **Orthographic Multi-Standard Ambiguity:** Vietnamese dual tone placement conventions (hòa vs hoà, hóa vs hoá, thủy vs thuỷ) create spurious character and word differences despite identical acoustic realizations.

---

## 3. Phase 2 Architectural Refactoring Plan

1. **`src/asr/whisper_engine.py`:**
   - In `WhisperASREngine.transcribe`:
     - Add `condition_on_prev_tokens=False` to `gen_kwargs`.
     - Add 50ms leading silence padding (`int(round(0.050 * sr))` samples) to `waveform` before computing Mel features.
2. **`src/asr/itn.py`:**
   - Extend `inverse_text_normalize` with traditional tone harmonization mappings:
     - `hòa` -> `hoà`, `hóa` -> `hoá`, `thủy` -> `thuỷ`, `tòa` -> `toà`, `thỏa` -> `thoả`, `xóa` -> `xoá`, `khóa` -> `khoá` (preserving casing).
   - Add ride-hailing seat numeral harmonization:
     - `ô tô bốn chỗ` / `ô tô 4 chỗ` -> `ô tô 4 chỗ` (case-insensitive with proper case retention).
3. **Unit Tests (`tests/test_itn.py`, `tests/test_asr.py`):**
   - Verify all tone conversions and ride-hailing seat normalizations pass 100%.
