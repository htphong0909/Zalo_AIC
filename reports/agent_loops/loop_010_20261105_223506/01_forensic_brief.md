# Loop 010 Forensic Diagnosis Brief

**Loop ID:** 10  
**Hypothesis:** Bilateral acoustic temporal padding (100ms pre-roll + 200ms post-roll buffer) on PhoWhisper Large to eliminate speech boundary truncation and final syllable dropouts  
**Author:** Lead ASR & Deep Learning Researcher  
**Target:** Train Glass-Box Forensic Analysis & Precision Recovery (DoD Target: Dev WER <= 1.50%, CER <= 1.00%, EM >= 95.00%, Latency <= 0.50s)

---

## 1. Train Glass-Box Forensic Diagnosis (Analysis of Loop 007 Residual Errors)

Detailed acoustic and phonetic investigation on the 14 residual errors from the Loop 007 Train evaluation (`reports/agent_loops/loop_007_20261105_095436/03_train_error_diff.json`):

### A. Post-Roll Trailing Silence Deficit (17.3% Affected Utterances):
- **Empirical Evidence:** Boundary RMS audit across all 150 Train evaluation samples revealed that **26 out of 150 clips (17.3%)** have active voicing continuing into the final 100ms of the recording (`end_rms > 0.01`).
- **Critical Failure Cases:**
  - `train_0083` (duration 5.608s): Audio ends abruptly during active voicing of "độ" (RMS=0.0705 at 5.31s-5.41s, RMS=0.0538 at 5.41s-5.51s, RMS=0.0435 at 5.51s-5.61s, Peak=0.1852). In Loop 07, `post_roll_pad_ms` was 0.0ms. Whisper's cross-attention decoder prematurely terminated with `<|endoftranscript|>`, dropping "độ" entirely: GT: `"điều hoà phòng khách để 26 độ."` vs PRED: `"điều hoà phòng khách để 26."`
  - `train_0208` (end_rms=0.0352): Final syllable corrupted: GT: `"tiền tiết kiệm chuyển thay."` vs PRED: `"tiết kiệm chuyển thôi."`
  - `train_0011` & `train_0079`: Sentence-final modal particle corrupted: GT: `"...nhé."` vs PRED: `"...nha."`
  - `train_0015`: Tail particle corrupted: GT: `"...dùm tôi hen."` vs PRED: `"...giùm tôi em."`
- **Acoustic & Architectural Mechanism:** Whisper's 2-stage Conv1D temporal downsampling layers (stride 2 + stride 2 = factor 4 downsampling, kernel size 3) require future acoustic context to compute stable boundary representations. Furthermore, Whisper was pretrained on 30s clips where utterances naturally decay into silence before padding. Without trailing silence (0ms post-roll), an abrupt cutoff at peak voicing produces unnatural spectral transitions, forcing premature `<|endoftranscript|>` or tail substitution errors.
- **Principled Solution:** Prepend 200ms (3200 samples at 16 kHz = 20 STFT hops) of trailing neutral silence padding. This provides a natural acoustic decay buffer that allows the decoder to finish decoding all voiced syllables before emitting the end-of-transcript token.

### B. Pre-Roll Window Margin Deficit:
- **Empirical Evidence:** In Loop 07, 40ms pre-roll pad (640 samples = 4 STFT hops) improved WER from 1.74% to 1.61%, proving the validity of acoustic padding. However, low-energy onset consonants still suffered from boundary distortion:
  - `train_0147`: GT: `"Dạ Zalo đi ạ."` vs PRED: `"Nhạc Zalo đi ạ."` (onset voiced fricative /z/ corrupted into velar nasal /ɲ/).
  - `train_0006`: GT: `"Nếu giá mua vào..."` vs PRED: `"Điều giá thu mua vào..."` (onset alveolar nasal /n/ distorted).
- **Acoustic Mechanism:** Whisper's Conv1D front-end has an effective receptive field of ~110ms across 2 downsampling layers. A 40ms pre-roll margin is narrower than the filter span, leaving the first ~70ms vulnerable to boundary filterbank zero-leakage.
- **Principled Solution:** Expand pre-roll padding from 40ms to 100ms (1600 samples = 10 STFT hops), placing the full onset transient squarely within the receptive field.

### C. Latency and Dimension Invariant:
- Whisper maps all audio inputs into a fixed 30-second Mel spectrogram tensor `[batch, 80, 3000]`. Adding 100ms leading and 200ms trailing padding adjusts waveform indices inside the 30-second window, adding 0 parameters and 0 FLOPS. GPU latency remains strictly $\le 0.50$s per sample.

---

## 2. Refactoring Plan (Phase 2)

1. **`src/asr/whisper_engine.py`:**
   - Update `WhisperASREngine.__init__` with default `pre_roll_pad_ms=100.0` and `post_roll_pad_ms=200.0`.
   - Update `_pad_waveform(waveform, sr)` to perform bilateral padding:
     `np.pad(waveform, (pre_samples, post_samples), mode="constant", constant_values=0.0)`.
2. **`configs/e2e_phowhisper_qwen35_4b.yaml`:**
   - Add explicit declarative fields `pre_roll_pad_ms: 100.0` and `post_roll_pad_ms: 200.0` under `asr`.
3. **`tests/test_asr.py`:**
   - Add unit tests verifying `_pad_waveform` applies correct bilateral pre/post samples.
   - Run `python -m pytest tests/ -q` to guarantee 100% green.
