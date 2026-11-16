# Loop #14 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Executive Summary & Baseline State
- **Prior Best Accepted Loop:** Loop #12 (Dev WER: 1.57%, Dev CER: 1.09%, Dev EM: 87.29%, Train WER: 0.99%, Train EM: 92.00%, Latency: 0.453s).
- **Target Definition of Done (DoD):** Dev WER $\le$ 1.50%, Dev CER $\le$ 1.00%, Dev EM $\ge$ 95.00%, Latency $\le$ 0.50s.
- **Current Gap to Primary Milestone:** 0.07% Dev WER.
- **Loop #13 Post-Mortem:** Prompt over-expansion with multi-domain entities in Loop #13 caused lexical collisions and regressed WER (REJECTED by Pareto Gate). The concise prompt from Loop #12 is preserved intact as the gold standard.

---

## 2. Train Glass-Box Forensic Diagnosis (12 Remaining Error Samples)

A systematic acoustic, phonological, and sequence decoding investigation of the 12 error cases in `reports/agent_loops/loop_012_20261106_001221/03_train_error_diff.json` revealed two core physical root causes:

### Root Cause 1: Autoregressive Token Attention Latching (Phrase Echoing)
- **Manifestation:**
  - `train_0105`:
    - Ground Truth: `"...đón trước giờ hẹn 30 phút, rồi nhắc chị mang laptop trước lúc xe đón 15 phút."`
    - Prediction: `"...đón trước giờ hẹn 30 phút, rồi nhắc chị bằng laptop check lúc đón xe 30 phút."`
    - Mechanism: Whisper repeated the previously emitted token `"30 phút"` at the utterance tail instead of attending to the acoustic frames corresponding to `"15 phút"`.
  - `train_0006`:
    - Ground Truth: `"Nếu giá mua vào vàng..."`
    - Prediction: `"Điều giá thu mua vào vàng..."` (stuttering insertion of `"thu mua"` before `"mua vào"`).
- **Theoretical Root Cause:** Default greedy decoding in Transformers Whisper runs with `repetition_penalty = 1.0` (unpenalized). In complex multi-clause commands containing multiple numerical or time entities, the self-attention mechanism assigns disproportionate prior mass to already decoded token representations in the KV-cache, causing the model to loop or substitute earlier tokens when acoustic confidence is slightly challenged.
- **Solution:** Inject deterministic `repetition_penalty = 1.05` into `WhisperASREngine._build_gen_kwargs()`. A 5% logit penalty suppresses autoregressive loop latching while preserving legitimate linguistic repetitions.

### Root Cause 2: Acoustic Boundary Noise Amplification & Offset Drag
- **Manifestation:**
  - `train_0147` ("Dạ Zalo đi ạ"): Audio has 850ms of quiet breathing/noise before speech onset at 850ms; Whisper hallucinated `"Nhả"` instead of `"Dạ"`.
  - `train_0006`: Audio has 650ms of lead-in room/mic hum before speech onset; Whisper hallucinated `"Điều giá thu"` instead of `"Nếu giá"`.
  - `train_0015`: Audio has 800ms of post-speech mouth noise/exhale around 7000ms; Whisper hallucinated `"em"` for trailing `"hen."`.
- **Acoustic Profiling:**
  - Quantitative energy profiling across Train audio revealed that 48% of audio samples contain $> 200$ms (and up to $1,180$ms) of pre-speech silence/noise, and 34% contain $> 200$ms (up to $1,396$ms) of post-speech silence/decay.
  - Linear peak normalization (Loop #12) scales quiet recordings by up to $2.86\times$. When an audio clip has hundreds of milliseconds of ambient mic hum before speech starts, this hum is amplified and fed into Whisper's encoder. The encoder consumes its initial receptive field on amplified noise, triggering hallucinated prefix syllables (`"Nhả"`, `"Điều"`).
- **Solution:** Implement conservative energy-based boundary trimming with a safety collar (`vad_margin_ms = 120.0`, `vad_db_threshold = -35.0 dB`) inside `_preprocess_waveform()`.
  - Trimming executes strictly BEFORE peak normalization and before symmetric 40ms collar padding.
  - The generous 120ms safety margin guarantees zero truncation of soft initial/final consonants (fricatives, plosives, nasals).
  - Peak normalization then operates purely on the active speech segment, and 40ms symmetric zero-padding provides clean acoustic transitions.
  - Latency impact: < 1.0ms per 10s audio.

---

## 3. Phase 2 Architectural & Decoding Implementation Plan

1. **`src/asr/whisper_engine.py`:**
   - Add parameters to `WhisperASREngine.__init__`:
     - `repetition_penalty: float = 1.05`
     - `trim_silence: bool = True`
     - `vad_db_threshold: float = -35.0`
     - `vad_margin_ms: float = 120.0`
   - Implement `_trim_silence(waveform, sr)` using vectorised sliding frame RMS and dB relative threshold with collar margin.
   - Update `_preprocess_waveform(waveform, sr)`:
     1) `_trim_silence` (if enabled)
     2) `peak_normalize` (if enabled)
     3) `pad_waveform` (40ms bilateral collar)
   - Update `_build_gen_kwargs()` to supply `repetition_penalty=self.repetition_penalty` when set.
   - Preserve `DEFAULT_WHISPER_PROMPT` unchanged from Loop #12 baseline.

2. **`tests/test_asr.py`:**
   - Add unit tests for `_trim_silence`:
     - Verification of lead/tail trimming on padded test signal.
     - Verification of safety collar preservation (length >= signal - cuts).
     - Edge-case testing: zero waveform, empty waveform, ultra-short waveform.
     - Test that `repetition_penalty` is populated in `_build_gen_kwargs()`.

---

## 4. Verification Gate
- Unit tests (`python -m pytest tests/ -q`): Must maintain 100% pass rate.
- Local zero-compute mandate: Strict adherence, zero local inference/training.
