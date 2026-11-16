# Loop #29 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Executive Summary & Baseline State
- **Prior Best Accepted Loop:** Loop #25 (Dev WER: 1.52%, Dev CER: 1.08%, Dev EM: 87.29%, Train WER: 0.99%, Train EM: 92.00%, Latency: ~0.355s).
- **Target Definition of Done (DoD):** Dev WER <= 1.50%, Dev CER <= 1.00%, Dev EM >= 95.00%, Latency <= 0.50s.
- **Current Gap to Milestone 1:** Exactly 0.02% Dev WER (a single word correction across 299 Dev utterances reaches the <= 1.50% milestone).
- **Historical Collar Padding Progression:**
  - *Loop 0 (0 ms padding):* Dev WER = 1.74% (severe boundary clipping on onset/offset consonants).
  - *Loop 7 (40 ms padding):* Dev WER = 1.61% (partial receptive field coverage, 2 downsampled frames).
  - *Loop 17 (60 ms padding):* Dev WER = 1.55% (3 downsampled frames, substantially reduced boundary clipping).
  - *Loop 25 (60 ms padding + repetition penalty 1.02):* Dev WER = 1.52%, Dev EM = 87.29%.

---

## 2. Train Glass-Box Forensic Diagnosis: Boundary Acoustic Attenuation & Conv1D Receptive Field Undercoverage

Detailed acoustic inspection of Whisper's front-end feature extraction architecture explains why 60ms collar padding is still sub-optimal:

### Receptive Field Derivation:
1. **Audio & STFT Parameters:**
   - Sampling rate: $f_s = 16,000\text{ Hz}$.
   - STFT window: 400 samples ($25\text{ ms}$).
   - STFT hop size: 160 samples ($10\text{ ms}$).
2. **Conv1D Encoder Front-end:**
   - Conv1: $\text{kernel}=3, \text{stride}=1, \text{padding}=1 \implies \text{RF}_{\text{conv1}} = 3\text{ STFT frames}$.
   - Conv2: $\text{kernel}=3, \text{stride}=2, \text{padding}=1 \implies \text{RF}_{\text{conv2}} = 1 + (3-1) + (3-1) = 5\text{ STFT frames}$.
   - Temporal span of 5 STFT frames: $(5 - 1) \times 10\text{ ms} + 25\text{ ms} = 40\text{ ms} + 25\text{ ms} = 65.0\text{ ms}$ ($1,040\text{ samples}$).
3. **The 60ms Undercoverage Deficit:**
   - At $60.0\text{ ms}$ padding ($6\text{ hops} = 960\text{ samples}$), the padding window is strictly smaller than the $65.0\text{ ms}$ receptive field ($60\text{ ms} < 65\text{ ms}$).
   - As a direct consequence, the initial downsampled encoder vector at $t=0$ suffers from boundary attenuation and partial convolution edge truncation, truncating the initial 5ms of attack acoustics for leading unvoiced consonants and aspirates.

### Exact Frame-Grid Alignment:
- Whisper downsamples temporal resolution by $2\times$ in Conv2, yielding an encoder frame rate of $20\text{ ms}$ per token ($320\text{ samples}$).
- Symmetric $80.0\text{ ms}$ collar padding corresponds to:
  $$\frac{80.0\text{ ms}}{10.0\text{ ms/hop}} = 8\text{ STFT hops} = 1,280\text{ samples}$$
  $$\frac{80.0\text{ ms}}{20.0\text{ ms/frame}} = 4\text{ downsampled encoder frames}$$
- $80.0\text{ ms}$ is the exact integer multiple of encoder frames that strictly covers the $65.0\text{ ms}$ receptive field ($80\text{ ms} \ge 65\text{ ms}$) with zero fractional phase shift!

---

## 3. Phase 2 Architectural & Decoding Refactoring: Symmetric 80.0 ms Integer 4-Frame Collar

### Changes in `src/asr/whisper_engine.py`:
1. **Default Collar Padding:**
   - `pre_roll_pad_ms: float = 80.0`
   - `post_roll_pad_ms: float = 80.0`
   - Fully covers the 65.0 ms receptive field ($80.0\text{ ms} \ge 65.0\text{ ms}$), preventing boundary edge distortion.
2. **Preserved Golden Parameters:**
   - `repetition_penalty: float = 1.02` (from Loop #25 golden baseline).
   - `vad_db_threshold: float = -35.0` and `vad_margin_ms: float = 150.0` (optimal conservative boundary VAD).
   - `peak_normalize: bool = True` (0.95 scaling for Mel headroom).
3. **Modular Function Exposure:**
   - Expose module-level `preprocess_waveform(waveform: np.ndarray, sr: int = 16000, ...)` for clean modularity, code reuse, and independent unit testability.

### Verification & Compliance:
- **Unit Tests:** All tests in `tests/test_asr.py` updated to 80.0 ms assertions and 100% passing.
- **Sealed Blind Dev Invariant:** Zero Dev data accessed or snooped.
- **Local Zero-Compute Mandate:** Lightweight unit testing only (`pytest tests/ -q`), no local GPU/heavy model inference.
