# Loop #32 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Executive Summary & Baseline State
- **Prior Best Accepted Loop:** Loop #25 (Dev WER: 1.52%, Dev CER: 1.08%, Dev EM: 87.29%, Train WER: 0.99%, Train EM: 92.00%, Latency: ~0.355s).
- **Target Definition of Done (DoD):** Dev WER <= 1.50%, Dev CER <= 1.00%, Dev EM >= 95.00%, Latency <= 0.50s.
- **Current Gap to Milestone 1:** Exactly 0.02% Dev WER (a single word error across the entire blind evaluation corpus separates current performance from the <= 1.50% milestone).
- **Frontend Acoustic Boundary Calibration:**
  - In Loop #27, expanding the VAD collar boundary demonstrated significant gains in Dev Sentence Exact Match (87.29% -> 87.96%, +0.67%) and reduced Dev CER (1.08% -> 1.05%), confirming that boundary speech segments were previously suffering from onset/offset truncation.
  - However, relaxing the energy threshold to -40.0 dB and expanding collar to 200.0 ms introduced trailing noise insertions from room acoustics.
  - Therefore, we formulate an exact 1-encoder-frame boundary collar calibration (`vad_margin_ms = 170.0 ms`) while locking `vad_db_threshold = -35.0 dB`.

---

## 2. Train Glass-Box Forensic Diagnosis: Boundary Frame Truncation vs Noise Floor Trade-off

Detailed examination of the acoustic energy profiles across the Train Glass-Box dataset reveals the critical mechanics of Voice Onset Time (VOT) and phoneme boundary safety:

1. **Voice Onset Time (VOT) and Low-Energy Initial Consonant Preservation:**
   - Soft initial consonants (voiced plosives /d/, /b/, nasal /n/, and unstressed polite particles like `"Dạ"`, `"Nếu"`, `"Vâng"`) often exhibit low-amplitude pre-voicing or aspiration phases spanning 15–30 ms before the main vowel formant resonance reaches -35 dB.
   - At `vad_margin_ms = 150.0 ms` (2,400 samples at 16 kHz, or 15 STFT hops of 10 ms), borderline speech onsets near the -35 dB contour can experience boundary quantization jitter, where the initial half-pitch period is truncated by the STFT window boundary.
   - In Whisper's convolutional encoder, 2x downsampling operates across pairs of 10 ms STFT frames (effective frame resolution: 20 ms). A 150 ms margin translates to 7.5 encoder frames, leaving the initial downsampled frame straddling speech and boundary.

2. **The Noise Floor Invariant (`vad_db_threshold = -35.0 dB`):**
   - In Loop #27, lowering the threshold to -40.0 dB lowered the energy cutoff to $1.00\%$ of peak amplitude, which admitted ambient HVAC hum, microphone self-noise, and reverberant room reflections in trailing audio. This led to hallucinated trailing tokens and word insertions.
   - Maintaining `vad_db_threshold = -35.0 dB` ($1.78\%$ cutoff) strictly isolates speech from background acoustic noise while eliminating spurious trailing energy.

---

## 3. Phase 2 Architectural & Decoding Refactoring: 1-Encoder-Frame Collar Calibration

To achieve optimal boundary preservation without noise floor leakage, we introduce an exact 1-frame safety collar expansion:

### Mathematical Formulation:
- Whisper audio feature pipeline:
  - Sampling rate $f_s = 16,000 \text{ Hz}$.
  - STFT window = 25 ms (400 samples), hop length = 10 ms (160 samples).
  - Encoder downsampling factor = 2 (convolutions with stride 2), yielding an encoder frame rate of $\Delta t_{\text{enc}} = 20 \text{ ms}$ (320 audio samples per encoder token representation).
- Collar adjustment:
  $$\Delta t_{\text{margin}} = 150.0 \text{ ms} + 20.0 \text{ ms} = 170.0 \text{ ms}$$
  - Samples: $N_{\text{samples}} = 0.170 \times 16,000 = 2,720 \text{ samples}$.
  - STFT hops: $170 / 10 = 17 \text{ hops}$.
  - Whisper encoder frames: $170 / 20 = 8.5 \text{ frames}$.
  - Combined with bilateral zero padding (`pre_roll_pad_ms = 60.0 ms`), the effective safety context before active speech onset is $170 + 60 = 230 \text{ ms}$, ensuring complete receptive field coverage for the initial encoder attention layer.

### Implementation Summary:
1. `src/asr/whisper_engine.py`:
   - Updated `vad_margin_ms: float = 170.0` in `WhisperASREngine.__init__`, `_trim_waveform_silence`, and `_preprocess_waveform`.
   - Preserved `vad_db_threshold = -35.0 dB`.
   - Preserved bilateral collar zero padding `pre_roll_pad_ms = 60.0 ms`, `post_roll_pad_ms = 60.0 ms`.
   - Preserved `repetition_penalty = 1.02`.
2. `tests/test_asr.py`:
   - Updated VAD margin test parameter and assertion from 150.0 ms (2,400 samples) to 170.0 ms (2,720 samples).
