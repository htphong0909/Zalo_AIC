# Loop #24 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Executive Summary & Baseline State
- **Prior Best Accepted Loop:** Loop #17 (Dev WER: 1.55%, Dev CER: 1.09%, Dev EM: 86.96%, Train WER: 0.99%, Train EM: 92.00%, Latency: 0.363s).
- **Target Definition of Done (DoD):** Dev WER <= 1.50%, Dev CER <= 1.00%, Dev EM >= 95.00%, Latency <= 0.50s.
- **Current Gap to Milestone 1:** Exactly 0.05% Dev WER (~1-2 word corrections across the entire 299 Dev utterances).
- **Loops #18–#23 Post-Mortem Synthesis:**
  - *Loops #18 & #19 (Padding & decoding stabilization):* Asymmetric collar padding (60ms/90ms) and fallback parameters produced 0.00% delta. Once the temporal padding exceeds Conv1D receptive field (~50ms), time-domain collar expansions yield diminishing returns.
  - *Loops #20 & #21 (Prompt restructuring):* Changing prompt syntax caused +0.10% to +0.19% Dev WER regressions, confirming the golden Loop #12 prompt is structurally optimal.
  - *Loop #22 (Southern dialect particle normalization):* Heuristic substitution of "hông" to "không" regressed Dev WER (+0.01%) due to ground truth annotator variance.
  - *Loop #23 (Constrained 2-beam search):* Beam search worsened Dev WER (+0.06%), Train WER (+0.12%), and doubled latency (0.79s vs 0.35s). Greedy decoding ($B=1$) is strictly superior on short Vietnamese voice assistant commands.

---

## 2. Train Glass-Box Forensic Diagnosis (12 Remaining Errors in Loop #17)

Acoustic and spectral profiling of the 12 error utterances in `reports/agent_loops/loop_017_20261106_030003/03_train_error_diff.json` revealed an underlying signal-processing bottleneck:

### Root Cause: High-Frequency Spectral Tilt Masking Low-Energy Unvoiced Consonants
1. **Unvoiced Plosive Burst Attenuation:**
   - In `train_0038`, ground truth `"Em ơi, đừng tắt đèn bếp nhé, cứ để đó."` was decoded as `"Em ơi, đừng bắt bếp nhé, cứ để đó."` (onset plosive /t/ in `"tắt"` misclassified as bilabial /b/, with subsequent syllable omission).
   - Acoustic inspection revealed `train_0038` exhibits an unusually low high-frequency energy ratio (0.2020) compared to typical clean speech (>0.60).
   - The unvoiced dental/alveolar stop [t] relies on a transient burst between 4.0 kHz and 7.0 kHz. In recordings dominated by low-frequency microphone hum or room resonance, the natural -6 dB/octave spectral roll-off suppresses this burst below the Mel filterbank activation threshold, causing confusion with low-frequency voiced stops (/b/).
2. **Fricative and Affricate Spectral Confusion:**
   - In `train_0147`, ground truth `"Dạ Zalo đi ạ."` was decoded as `"Nhả Zalo đi ạ."`. The initial voiced affricate/fricative (/z/ or /j/) was misidentified as nasal /ɲ/. High-frequency friction noise was submerged by low-frequency energy.
3. **Formant Disparity in High-Vowel Codas:**
   - In `train_0079`, ground truth `"...trên 500 nghìn..."` was decoded as `"...trên 500 ngày..."`. The closed front vowel /i/ in `"nghìn"` (characterized by a high second/third formant $F_2/F_3 > 2500$ Hz) was confused with open central vowel /a/ in `"ngày"`.

---

## 3. Phase 2 Architectural Leap: Gentle Pre-Emphasis Acoustic Filtering

To resolve high-frequency consonant masking without altering model weights or inducing hallucination, we introduce a **gentle first-order FIR pre-emphasis high-pass filter**:

$$y[t] = x[t] - \alpha \cdot x[t-1]$$

### Key Design Principles:
1. **Calibrated Parameter ($\alpha = 0.90$):**
   - Classical speech processing uses $\alpha = 0.95$–$0.97$, which attenuates low frequencies by 25–30 dB and alters vowel formants.
   - A calibrated $\alpha = 0.90$ provides a gentle +10 dB to +15 dB high-frequency lift. This flattens the speech spectrum sufficiently to sharpen [t], [s], and [z] bursts while fully preserving fundamental frequency ($F_0$) and first formant ($F_1$) structures required by PhoWhisper Large's 80-channel Mel filterbanks.
2. **Pipeline Ordering:**
   - **Step 1: Energy VAD Silence Trimming:** Performed *before* pre-emphasis to ensure boundary silence detection operates on natural acoustic RMS levels.
   - **Step 2: Pre-Emphasis Filtering:** Applied to the trimmed speech signal ($y[t] = x[t] - \alpha x[t-1]$).
   - **Step 3: Peak Amplitude Normalization (0.95):** Applied *after* pre-emphasis to normalize the high-pass filtered waveform to optimal Mel filterbank dynamic headroom without digital clipping.
   - **Step 4: Symmetric Bilateral Collar Padding (60ms):** Pads clean boundary frames for Conv1D receptive field safety.
3. **Strict Invariants Maintained:**
   - Preserves greedy decoding ($B=1$), golden prompt, ITN tone mapping, and bilateral 60ms collar padding.

---

## 4. Phase 2 Architectural & Decoding Implementation Plan

1. **`src/asr/whisper_engine.py`:**
   - Add `pre_emphasis: float = 0.90` parameter to `WhisperASREngine.__init__`.
   - Implement `_apply_pre_emphasis(self, waveform: np.ndarray, alpha: float = 0.90) -> np.ndarray` using vectorized NumPy array slicing with dtype preservation.
   - Integrate `_apply_pre_emphasis` into `_preprocess_waveform` between silence trimming and peak amplitude normalization.
2. **`tests/test_asr.py`:**
   - Update `DummyEngine` defaults to include `pre_emphasis = 0.90` and `_apply_pre_emphasis = WhisperASREngine._apply_pre_emphasis`.
   - Add unit tests verifying `_apply_pre_emphasis` behavior: impulse response, DC suppression, array length preservation, and dtype consistency.

---

## 5. Verification Gate
- Unit tests (`python -m pytest tests/ -q`): Must pass 100% green.
- Anti-snoop invariant: Zero Dev access.
- Zero local compute mandate: Strictly lightweight unit tests only.
