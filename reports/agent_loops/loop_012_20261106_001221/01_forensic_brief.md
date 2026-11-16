# Loop #12 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Executive Summary & Baseline State
- **Prior Best Accepted Loop:** Loop #11 (Dev WER: 1.60%, Dev CER: 1.07%, Dev EM: 86.96%, Train WER: 1.14%, Train EM: 91.33%)
- **Target Definition of Done (DoD):** Dev WER $\le$ 1.50%, Dev CER $\le$ 1.00%, Dev EM $\ge$ 95.00%, Latency $\le$ 0.50s
- **Optimization Objective:** Bridge the final 0.10% WER gap to cross the primary $\le$ 1.50% DoD threshold through linear acoustic waveform normalization, prompt de-biasing, and conversational vocative grounding without dev set leakage.

---

## 2. Train Glass-Box Forensic Diagnosis (150 Samples)

Systematic acoustic and linguistic investigation into the remaining 13 Train Glass-Box errors (`train_ids` in `configs/splits/dev_v1.json`) revealed three specific root causes:

### Root Cause 1: Dynamic Range Attenuation on Low-Gain Acoustic Waveforms
- **Acoustic Phenomenon:** Analysis of the error samples revealed that audio files have low gain (mean RMS: -25.0 dBFS, mean peak: 0.448, with quietest samples peaking at only 0.311).
- **STFT Filterbank Clamping Mechanism:** Whisper's log-Mel spectrogram feature extractor applies dynamic range clamping: $\text{log\_spec} = \max(\text{log\_spec}, \text{log\_spec.max()} - 8.0)$. When a speaker vocalizes quiet consonants or onsets (e.g., initial /z/ in `train_0147` "Dạ Zalo" or /n/ in `train_0006` "Nếu giá..."), their energy falls into the bottom 8.0 log-clamp floor, drowning out acoustic cues and causing the autoregressive decoder to substitute clearer phonemes ("Nhạc Zalo", "Điều giá").
- **Solution:** Integrated linear waveform peak normalization to 0.95 ($s_{\text{norm}} = s \cdot \frac{0.95}{\max(|s|, 10^{-6})}$) directly in the preprocessing pipeline before bilateral collar padding. This maximizes STFT dynamic range across all samples linearly without non-linear spectral distortion.

### Root Cause 2: Proper Name Hallucination Prior from Prompt Prefix
- **Error Manifestation:** In `train_0019`:
  - Ground Truth: `"Chị ơi, giúp tôi đặt phòng khách sạn ở Đà Lạt với nhé."`
  - Loop #11 Prediction: `"Chị Hương, tôi cần đặt phòng khách sạn ở Đà Lạt với."`
- **Autoregressive Bias Mechanism:** `DEFAULT_WHISPER_PROMPT` explicitly listed person names: `"Các nhân vật: Minh, Lan, Tuấn, Hương, Sếp Dũng, Bác Ba, Mai, Bố, Mẹ, An, Thảo, Nam, Bình, Phượng, Phạm Hùng."` Because Whisper's prefix conditioning attends back to prompt tokens, kinship address terms ("Chị...", "Anh...", "Em...") faced intense LM competition from the primed names ("Hương", "Tuấn", "Lan"). When acoustic clarity on the vocative particle "ơi" is weak, the model hallucinates a primed name ("Chị Hương").
- **Solution:** Completely pruned arbitrary proper names from the prompt, replacing them with natural conversational address starters and honorifics: `"Dạ, vâng, trợ lý ơi, em ơi, anh ơi, chị ơi."`

### Root Cause 3: Missing Domain Anchors for Smart Home Numerical Quantities
- **Error Manifestation:** In `train_0083`:
  - Ground Truth: `"...quạt thì khỏi bật, điều hoà phòng khách để 26 độ."`
  - Loop #11 Prediction: `"...quạt thì khỏi bật, điều hoà phòng khách để 26."` (dropped `"độ"`).
- **Mechanism:** In colloquial Vietnamese, climate control commands frequently end with numerical setpoints. In `train.jsonl`, all climate commands consistently include "độ". However, the prompt lacked grounding examples for climate temperature or percentage targets.
- **Solution:** Enriched prompt with domain-specific numerical examples: `"điều hoà 26 độ, đèn 30 phần trăm"`, reinforcing the probability of trailing unit tokens.
- **Decoding Parameter Tuning:** Explicitly bounded `max_new_tokens=128` in generation kwargs to eliminate HuggingFace deprecation warnings and prevent runaway decoding.

---

## 3. Architectural & Decoding Implementations (Phase 2)

1. **`src/asr/whisper_engine.py`:**
   - Added `peak_normalize: bool = True` to `WhisperASREngine.__init__`.
   - Implemented `_preprocess_waveform()` which scales waveform amplitude to peak 0.95 prior to applying 40ms symmetric bilateral collar padding.
   - Refactored `DEFAULT_WHISPER_PROMPT` to:
     ```python
     DEFAULT_WHISPER_PROMPT = (
         "Trợ lý ảo tiếng Việt LifeMate điều khiển thiết bị thông minh, điều hoà 26 độ, đèn 30 phần trăm, "
         "chuyển khoản ngân hàng, ví điện tử, đặt xe, liên lạc Zalo, SMS, gọi điện, tra cứu thời tiết, nhắc nhở lịch hẹn. "
         "Dạ, vâng, trợ lý ơi, em ơi, anh ơi, chị ơi."
     )
     ```
   - Added `max_new_tokens: 128` to `_build_gen_kwargs()`.

2. **`tests/test_asr.py`:**
   - Added unit test cases for peak amplitude normalization to 0.95, empty/zero waveform protection, and bilateral padding length preservation.

---

## 4. Verification & Unit Test Gate Results
- **Unit Test Suite:** `python -m pytest tests/ -q` -> **74/74 passed (100% green)** in 36.64s.
- **Local Zero-Compute Invariant:** Zero model weights downloaded, zero inference executed locally, full compliance with user mandate.
