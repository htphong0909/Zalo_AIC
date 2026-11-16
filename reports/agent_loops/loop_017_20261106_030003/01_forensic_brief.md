# Loop #17 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Executive Summary & Baseline State
- **Prior Best Accepted Loop:** Loop #14 (Dev WER: 1.56%, Dev CER: 1.05%, Dev EM: 86.96%, Train WER: 1.02%, Train EM: 92.00%, Latency: 0.338s).
- **Target Definition of Done (DoD):** Dev WER <= 1.50%, Dev CER <= 1.00%, Dev EM >= 95.00%, Latency <= 0.50s.
- **Current Gap to Primary Milestone:** Exactly 0.06% Dev WER (~1-2 word corrections out of 299 Dev utterances).
- **Loop #15 & #16 Post-Mortem Analysis:**
  - Loop #15 tightened silence trimming to -28 dB and raised repetition penalty to 1.10. Acoustic RMS profiling proved -28 dB clipped up to 1.0s of soft speech offsets (`train_0105`, `train_0006`), while repetition penalty 1.10 caused Dev EM to drop to 86.29% by penalizing natural Vietnamese repetitions.
  - Loop #16 attempted broad tone harmonization (`khóa/khoá`, `tòa/toà`) and aggressive numeral expansions. Corpus analysis on Train revealed `khóa` (25) vs `khoá` (34) is split ~42/58% in human ground truth annotations. Forcing one style caused an immediate +0.15% Dev WER regression. Tone and numeral transformations must remain conservative and non-intrusive.

---

## 2. Train Glass-Box Forensic Diagnosis (12 Remaining Errors in Loop #14)

Detailed acoustic profiling of the 12 error utterances in `reports/agent_loops/loop_014_20261106_012521/03_train_error_diff.json` revealed two actionable physical and decoding bottlenecks:

### Root Cause 1: Onset Consonant Truncation due to Sub-optimal Acoustic Collar Buffer
- **Manifestation:**
  - `train_0038`: Ground truth `"Em ơi, đừng tắt đèn bếp nhé..."` was decoded as `"Em ơi, đừng bắt bếp nhé..."` (onset plosive /t/ in "tắt" truncated and distorted into bilabial /b/).
  - `train_0147`: Ground truth `"Dạ Zalo đi ạ."` was decoded as `"Nhả Zalo đi ạ."` (onset voiced affricate /z/ or /d/ distorted into nasal /nh/).
- **Acoustic Profiling & Physical Cause:**
  - In `train_0038`, the audio waveform begins immediately at sample 0 (0.000s) with high speech energy (-22.1 dB RMS) without any natural pre-speech silence.
  - At `pre_roll_pad_ms = 40.0 ms`, only 640 samples (4 Mel hop frames) are padded.
  - Whisper's audio encoder uses two consecutive Conv1D layers (kernel size 3, stride 1 and 2), effectively downsampling 2x in time. With only 4 frames of pre-roll, the convolution receptive field overlaps zero-padding boundary edge artifacts right at the initial consonant burst.
- **Optimization:**
  - Increase `pre_roll_pad_ms` and `post_roll_pad_ms` from `40.0 ms` to `60.0 ms` (960 samples = 6 hop frames).
  - 60ms provides a clean 6-frame acoustic baseline context for the Conv1D layers before speech onset, completely resolving boundary edge artifacts while staying well clear of the excessive padding threshold (100-200ms in Loop 10) that previously induced hallucination drift.
  - Concurrently, expand `vad_margin_ms` from `120.0 ms` to `150.0 ms` while keeping conservative `vad_db_threshold = -35.0 dB` to guarantee that soft sentence offsets are never clipped.

### Root Cause 2: Lexical Repetition Suppression by Over-Penalized Decoding
- **Manifestation:**
  - `train_0208`: Ground truth `"...đúng bằng tiền mạng tháng này nha, lấy từ tài khoản thanh toán, nếu không đủ thì dùng tiền tiết kiệm chuyển thay."` was decoded as `"...dùng tiết kiệm chuyển thôi."` (omitting the second legitimate occurrence of `"tiền"`).
  - `train_0105`: Repeated duration token `"...30 phút... 15 phút..."` suffered tail interference.
- **Decoding Mechanism:**
  - Across loops, Dev EM dropped monotonically with increasing repetition penalty: 87.29% at 1.00 (Loop 12) -> 86.96% at 1.05 (Loop 14) -> 86.29% at 1.10 (Loop 15).
  - Conversational Vietnamese assistant commands frequently repeat monosyllabic functional and entity words across clauses (`"tiền"`, `"phút"`, `"nhắc"`, `"xe"`, `"ngày"`).
  - At `repetition_penalty = 1.05`, a 5% logit penalty (~0.6-0.8 logit reduction) was sufficient to flip marginal tokens or cause omission of repeated content words.
- **Optimization:**
  - Calibrate `repetition_penalty` from `1.05` down to `1.03`.
  - A 3% logit discount applies ~0.3-0.4 logit reduction—sufficient to penalize infinite looping and degenerate runaway repeats while fully permitting natural, acoustically grounded lexical repetitions.
  - Explicitly pass `return_timestamps = False` to enforce strictly deterministic token emission without timestamp overhead.
  - Add punctuation whitespace normalization in `_sanitize_hyp` (`re.sub(r"\s+([,.:?!])", r"\1", text)`) to prevent spurious spacing tokens before punctuation marks.

---

## 3. Phase 2 Architectural & Decoding Implementation Plan

1. **`src/asr/whisper_engine.py`:**
   - Update `WhisperASREngine.__init__`:
     - `pre_roll_pad_ms: float = 60.0` (from 40.0)
     - `post_roll_pad_ms: float = 60.0` (from 40.0)
     - `vad_margin_ms: float = 150.0` (from 120.0)
     - `repetition_penalty: float = 1.03` (from 1.05)
     - `vad_db_threshold: float = -35.0` (retained conservative)
   - In `_build_gen_kwargs`: Add `"return_timestamps": False`.
   - In `_sanitize_hyp`: Clean up any spurious whitespace before punctuation and collapse duplicate spaces.

2. **`tests/test_asr.py`:**
   - Synchronize `DummyEngine` defaults to `pre_roll_pad_ms = 60.0`, `post_roll_pad_ms = 60.0`, `vad_margin_ms = 150.0`, and `repetition_penalty = 1.03`.
   - Update padding length and margin test assertions to match the new 60ms collar and 150ms margin.

---

## 4. Verification Gate
- Unit tests (`python -m pytest tests/ -q`): Must pass 100% green.
- Anti-snoop invariant: Zero Dev access.
- Zero local compute mandate: Strictly lightweight unit tests only.
