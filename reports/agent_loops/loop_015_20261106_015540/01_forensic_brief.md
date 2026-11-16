# Loop #15 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Executive Summary & Baseline State
- **Prior Best Accepted Loop:** Loop #14 (Dev WER: 1.56%, Dev CER: 1.05%, Dev EM: 86.96%, Train WER: 1.02%, Train EM: 92.00%, Latency: 0.338s).
- **Target Definition of Done (DoD):** Dev WER $\le$ 1.50%, Dev CER $\le$ 1.00%, Dev EM $\ge$ 95.00%, Latency $\le$ 0.50s.
- **Current Gap to Primary Milestone:** Exactly 0.06% Dev WER (~1-2 word corrections out of 299 Dev utterances).
- **Loop #14 Achievement:** The introduction of repetition penalty (1.05) and boundary silence trimming (-35 dB, 120ms safety collar) yielded immediate gains across WER, CER, and reduced latency to 0.338s.

---

## 2. Train Glass-Box Forensic Diagnosis (12 Remaining Error Samples)

A systematic investigation of the 12 error cases in `reports/agent_loops/loop_014_20261106_012521/03_train_error_diff.json` revealed two actionable physical root causes:

### Root Cause 1: Low-to-Medium SNR Microphone Background Hum Penetration
- **Manifestation:**
  - `train_0006`: Ground truth `"Nếu giá mua vào vàng..."` was decoded as `"Điều giá thu mua vào vàng..."`.
  - `train_0105`: 1040ms of ambient room silence preceded speech before `"Đặt cho chị ô tô 4 chỗ..."`.
- **Acoustic Profiling & Physical Cause:**
  - In Loop #14, the energy boundary trimming threshold was set conservatively at `vad_db_threshold = -35.0 dB`.
  - Detailed waveform RMS profiling of `train_0006` revealed that ambient room hum and microphone noise floor sits between -26.0 dB and -29.0 dB (SNR ~22-25 dB).
  - Consequently, at -35.0 dB, trimming only eliminated 10ms of the 600ms pre-speech noise segment in `train_0006`. This hum was subsequently amplified by peak amplitude normalization and ingested by Whisper's audio encoder, triggering hallucinated prefix syllables (`"Điều giá thu"`).
- **Optimization:**
  - Shift `vad_db_threshold` from `-35.0 dB` to `-28.0 dB`.
  - Concurrently expand `vad_margin_ms` from `120.0 ms` to `150.0 ms`.
  - Across all 150 Train samples, `-28.0 dB` with `150.0 ms` margin increases average lead-in trimming from 0.26s to 0.36s (trimming 480ms of dead air/hum in `train_0006` and 1.04s in `train_0105`) and increases trailing silence trimming from 0.16s to 0.25s (eliminating mouth clicks/breaths like the tail of `train_0015`).
  - Strict duration safety analysis confirmed zero clipping of soft initial/final phonemes (speech rates remain constant at 0.18–0.22s/word across all samples).

### Root Cause 2: Persistent Attention Latching on Repeated Entities
- **Manifestation:**
  - `train_0105`: Ground truth `"...đón trước giờ hẹn 30 phút, rồi nhắc chị mang laptop trước lúc xe đón 15 phút."` was decoded as `"...đón trước giờ hẹn 30 phút, rồi nhắc chị bằng laptop check lúc đón xe 30 phút."` (repeating `"30 phút"` at the utterance tail instead of `"15 phút"`).
  - `train_0079`: Ground truth `"...trong năm ngày tới thì trả bằng ngân hàng hết, mà cái nào trên 500 nghìn..."` was decoded as `"...500 ngày..."` (echoing `"ngày"` from earlier `"năm ngày tới"`).
- **Physical Root Cause:**
  - While `repetition_penalty = 1.05` in Loop #14 began mitigating hallucinated loop repeats, a 5% logit discount corresponds to only ~0.5 logit reduction. When self-attention has strongly latched onto an entity representations in the KV-cache, 0.5 logit is insufficient to break the latch when trailing acoustic confidence is moderate.
- **Optimization:**
  - Increase `repetition_penalty` from `1.05` to `1.10`.
  - A 10% logit discount applies an effective ~1.0-1.4 logit penalty against recently decoded tokens, breaking KV-cache attention latching on repeated numbers and nouns while preserving grammatical fluency in Vietnamese.

---

## 3. Phase 2 Architectural & Decoding Implementation Plan

1. **`src/asr/whisper_engine.py`:**
   - Update `WhisperASREngine.__init__`:
     - `vad_db_threshold: float = -28.0` (from -35.0)
     - `vad_margin_ms: float = 150.0` (from 120.0)
     - `repetition_penalty: float = 1.10` (from 1.05)
   - Update method signatures and default parameter fallbacks in `_trim_waveform_silence()` and `_preprocess_waveform()`.
2. **`tests/test_asr.py`:**
   - Synchronize unit test harness (`DummyEngine`) defaults to `vad_db_threshold = -28.0`, `vad_margin_ms = 150.0`, and `repetition_penalty = 1.10`.
   - Verify safety margin preservation and repetition penalty propagation.

---

## 4. Verification Gate
- Unit tests (`python -m pytest tests/ -q`): Must pass 100% green.
- Anti-snoop invariant: Zero Dev access.
- Zero local compute mandate: Strictly lightweight unit tests only.
