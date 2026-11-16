# Loop #18 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Executive Summary & Baseline State
- **Prior Best Accepted Loop:** Loop #17 (Dev WER: 1.55%, Dev CER: 1.09%, Dev EM: 86.96%, Train WER: 0.99%, Train EM: 92.00%, Latency: 0.348s).
- **Target Definition of Done (DoD):** Dev WER <= 1.50%, Dev CER <= 1.00%, Dev EM >= 95.00%, Latency <= 0.50s.
- **Current Gap to Milestone 1:** Exactly 0.05% Dev WER (~1 word correction across the 299-sample Sealed Dev set).
- **Core Objective for Loop #18:** Deliver a principled acoustic preprocessing and collar geometry refinement to eliminate boundary decay truncation and ADC hardware bias, pushing Dev WER past the 1.50% milestone into the 1.4x% territory.

---

## 2. Train Glass-Box Forensic Diagnosis (12 Remaining Train Errors in Loop #17)

Analysis of the 12 remaining error samples in `reports/agent_loops/loop_017_20261106_030003/03_train_error_diff.json` and acoustic waveform profiling across the 1,246 Train split utterances revealed two key physical bottlenecks:

### Root Cause 1: Terminal Syllable & Nasal Coda Decay Truncation Under Symmetric 60ms Collar
- **Manifestation in Train Errors:**
  - `train_0015`: Ground truth `"...lấy từ tiết kiệm dùm tôi hen."` was decoded as `"...lấy từ tiết kiệm giùm tôi em."` — the utterance-final word `"hen"` (Southern modal particle) had its alveolar nasal coda /n/ truncated into bilabial /m/.
  - `train_0208`: Ground truth `"...dùng tiền tiết kiệm chuyển thay."` was decoded as `"...dùng tiết kiệm chuyển thôi."` — the sentence-final off-glide diphthong /aj/ decayed into /oj/.
  - `train_0011`: Ground truth `"...nạp tiền vào ví nhé."` was decoded as `"...nạp tiền vào ví nha."` — terminal polite particle `"nhé"` decayed into `"nha"`.
  - `train_0079`: Ground truth `"...để lại đó chưa trả nhé."` was decoded as `"...để lại đó chưa trả nha."` — sentence-final `"nhé"` decayed into `"nha"`.
- **Acoustic Profiling & Physical Cause:**
  - In conversational Vietnamese, utterance endings naturally exhibit falling pitch, decreased subglottal pressure, and breathy or unreleased terminal codas lasting 80–120ms after active speech drops below the energy threshold.
  - In Loop #17, the collar was expanded symmetrically to 60ms (960 samples = 6 Mel frames). While 60ms pre-roll is optimal for leading onset consonants (preventing Conv1D boundary artifacts without inducing hallucination), 60ms post-roll cuts too close to the decaying trailing speech tail.
  - At 60ms post-roll, the Whisper Transformer encoder's self-attention and cross-attention receptive fields overlap zero-padding borders right at the trailing formant transitions, distorting decaying codas (/n/ -> /m/, /aj/ -> /oj/).
- **Optimization Leap:**
  - Shift from symmetric (60ms / 60ms) to **asymmetric collar geometry**: `pre_roll_pad_ms = 60.0 ms`, `post_roll_pad_ms = 90.0 ms` (1440 samples = 9 Mel frames).
  - Pre-roll stays strictly at 60ms to prevent leading conversational hallucination drift (observed in Loop 10 at 100ms+).
  - Post-roll expanded to 90ms safely buffers the decaying phonetic tail, giving the Transformer encoder 9 full Mel frames of trailing context without boundary cutoff artifacts.
  - Latency impact: 0.000s, because Whisper's feature extractor pads all audio up to 3000 frames (30s) regardless.

### Root Cause 2: ADC / Microphone DC Offset & Mel Filterbank Low-Frequency Leakage
- **Manifestation & Acoustic Measurements:**
  - Waveform profiling across 100 Train utterances revealed non-zero mean values (`mean = -0.00003` to `-0.00011`, max abs `0.00038`).
  - While small in absolute terms, uncentered DC offset causes two subtle distortions:
    1. **Log-Mel Filterbank Leakage:** In the STFT step, non-zero DC energy concentrates in bin 0 (0 Hz) and bleeds across neighboring low-frequency bins (up to 100–200 Hz) due to Hann window spectral leakage.
    2. **Peak Normalization Asymmetry:** Peak normalization `waveform * (0.95 / max(abs(waveform)))` scales the positive and negative lobes unevenly when a DC baseline bias exists.
- **Optimization Leap:**
  - Implement zero-mean centering (DC offset removal) as Step 1 of `_preprocess_waveform`:
    `if getattr(self, "remove_dc_offset", True) and np.std(waveform) > 1e-5: waveform = waveform - float(np.mean(waveform))`
  - Guarantees zero-mean acoustic air-pressure oscillations prior to silence trimming and peak amplitude normalization, yielding clean, symmetric log-Mel filterbank inputs.

---

## 3. Phase 2 Architectural & Decoding Implementation Plan

1. **`src/asr/whisper_engine.py`:**
   - Add `remove_dc_offset: bool = True` parameter to `WhisperASREngine.__init__`.
   - Update default `post_roll_pad_ms: float = 90.0` (asymmetric 60ms pre-roll, 90ms post-roll).
   - In `_preprocess_waveform`:
     - Step 1: Remove DC offset via zero-mean centering (`waveform = waveform - float(np.mean(waveform))` when `np.std(waveform) > 1e-5`).
     - Step 2: Conservative VAD boundary silence trimming (`-35.0 dB`, `150.0 ms` safety margin).
     - Step 3: Peak amplitude normalization to `0.95`.
     - Step 4: Asymmetric bilateral collar padding (`60ms` pre-roll, `90ms` post-roll).
   - Retain golden decoding parameters: `repetition_penalty = 1.03`, `return_timestamps = False`, `max_new_tokens = 128`.

2. **`tests/test_asr.py`:**
   - Synchronize `DummyEngine` defaults: `pre_roll_pad_ms = 60.0`, `post_roll_pad_ms = 90.0`, `remove_dc_offset = True`, `repetition_penalty = 1.03`.
   - Update expected length assertions: `expected_len = 16000 + int(0.06 * 16000) + int(0.09 * 16000)`.
   - Add explicit test case verifying DC offset elimination on audio with simulated ADC hardware bias.

---

## 4. Verification Gate
- Unit tests: `python -m pytest tests/ -q` (must pass 100% green).
- Sealed Blind Dev Invariant: Zero access to Dev set data.
- Local Zero-Compute Mandate: All heavy compute deferred to Colab GPU session.
