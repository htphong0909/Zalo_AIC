# Loop #30 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Executive Summary & Baseline State
- **Prior Best Accepted Loop:** Loop #25 (Dev WER: 1.52%, Dev CER: 1.08%, Dev EM: 87.29%, Train WER: 0.99%, Train EM: 92.00%, Latency: ~0.355s).
- **Target Definition of Done (DoD):** Dev WER <= 1.50%, Dev CER <= 1.00%, Dev EM >= 95.00%, Latency <= 0.50s.
- **Current Gap to Milestone 1:** Exactly 0.02% Dev WER (only 1-2 words across 299 Dev utterances separates current performance from the <= 1.50% milestone).
- **Acoustic Optimization Plateau:**
  - Signal-level acoustic conditioning (bilateral 60ms collar padding, -35 dB VAD trimming with 150ms safety margin, peak amplitude normalization to 0.95, and 1.02 repetition penalty) has extracted the maximum zero-shot capability from base PhoWhisper Large (1.54B params).
  - Further signal-level collar shifts (80ms in Loop #29, asymmetric collars in Loop #18, pre-emphasis in Loop #24) yielded no further gains, indicating an asymptotic zero-shot ceiling.

---

## 2. Train Glass-Box Forensic Diagnosis: Lexical Domain Shift & Dialectal Phoneme Variations

Detailed examination of the remaining 12 error utterances on the Train Glass-Box split (`train_ids`, 150 diagnostic samples, WER = 0.99%) reveals that errors are semantic, lexical, and dialectal rather than acoustic boundary artifacts:

1. **Conversational Final Particle Variations & Regional Politeness Markers:**
   - Colloquial spoken Vietnamese frequently alternates between regional particles such as *'nha'* vs *'nhé'*, *'dùm'* vs *'giùm'*, and *'nghen'*.
   - Zero-shot PhoWhisper decoder weights place balanced prior probabilities on both variants, leading to occasional mismatch with the canonical ground truth.

2. **LifeMate In-Domain Named Entities & Banking Syntax:**
   - Spoken commands in the LifeMate domain feature specialized command structures ("chuyển tiền", "đặt xe", "nhiệt độ điều hòa") and Vietnamese proper names.
   - Without task-specific weight adaptation, out-of-vocabulary and rare named entities experience logit dispersion across acoustic neighbours.

3. **Phonetic Ambiguity in Rapid Connected Speech:**
   - Contracted compound numerals and rapid dialectal consonant lenitions require acoustic encoder-decoder attention adaptation to resolve fine phonetic contrasts.

---

## 3. Phase 2 Architectural & Decoding Refactoring: LoRA Domain Adaptation Infrastructure

To break through the 1.50% WER barrier, we implement an end-to-end Parameter-Efficient Fine-Tuning (PEFT / LoRA) adapter infrastructure:

### Code Modifications:
1. **Engine Adapter Support (`src/asr/whisper_engine.py`):**
   - Added `adapter_path: str | Path | None = None` parameter to `WhisperASREngine.__init__`.
   - When `adapter_path` is provided, wraps `self.model` using `PeftModel.from_pretrained(self.model, str(adapter_path)).to(device)`.
   - Exposes module-level helper `preprocess_waveform(waveform: np.ndarray, sr: int = 16000, ...)` with golden defaults (`pre_roll_pad_ms=60.0`, `post_roll_pad_ms=60.0`, `peak_normalize=True`, `trim_silence=True`, `vad_db_threshold=-35.0`, `vad_margin_ms=150.0`).
   - Preserves all golden baseline parameters in `WhisperASREngine` (`pre_roll_pad_ms=60.0`, `post_roll_pad_ms=60.0`, `repetition_penalty=1.02`, `vad_db_threshold=-35.0`, `vad_margin_ms=150.0`).

2. **Dual-Referee Evaluation Integration:**
   - Added `--adapter-path` CLI option to `src/eval/train_asr_evaluator.py`.
   - Added `--adapter-path` CLI option to `src/eval/blind_dev_evaluator.py`.
   - Instantiates `WhisperASREngine(model_id=args.model_id, adapter_path=args.adapter_path)`.

3. **Test Suite Verification (`tests/test_asr.py`):**
   - Updated `test_whisper_engine_import_and_signature` to assert `adapter_path` parameter exists with default `None`.
   - Added unit test verification for standalone `preprocess_waveform` helper and `trim_waveform_silence`.

### Compliance:
- **Sealed Blind Dev Invariant:** Zero Dev data accessed or snooped.
- **Local Zero-Compute Mandate:** Lightweight unit testing only (`pytest tests/ -q` = 68/68 passed in 23.86s). No local GPU training or heavy inference.
