# Loop 003 Forensic Diagnosis & Architectural Plan

**Loop ID:** 3  
**Target:** Train Glass-Box Forensic Analysis & Migration to Zalo-Specialized PhoWhisper Large (1.54B)  
**Author:** Lead ASR Researcher  
**Status:** Phase 1 Complete -> Proceeding to Phase 2  

---

## 1. Train Glass-Box Forensic Analysis (Loops 01 & 02)

Analysis based on Train split error patterns (`train_ids` in `configs/splits/dev_v1.json`) across previous loops:

### Quantitative Performance Trajectory:
- **Base Qwen-1.7B (Zero-Shot Baseline, Loop 000):** Dev WER **28.79%**, Dev EM **2.01%**, Train WER **31.23%**. Severe phonemic hallucinations, tonal confusion, and total breakdown on Vietnamese conversational phrases.
- **Qwen-1.7B LoRA SFT (Loop 001):** Dev WER **12.42%**, Dev EM **24.75%**, Train WER **5.79%**. Although LoRA adapted the model toward assistant commands, a substantial 12.42% WER gap remained.
- **Loop 002 (Base Qwen + ITN):** Evaluated base Qwen instead of specialized weights, proving that post-processing alone cannot bridge acoustic misrecognitions if the acoustic front-end itself lacks high-precision Vietnamese phoneme alignment.

### Acoustic & Phonetic Error Breakdown:
1. **Vietnamese Tonal & Vowel Confusion:**
   - In conversational Vietnamese, subtle vowel length distinctions and tone contours (ngã vs. hỏi, sắc vs. nặng in Southern/Central dialects) are routinely misrecognized by general multilingual models like Qwen-ASR.
   - Examples from Train errors: phonetically adjacent syllables confused due to lack of deep Vietnamese acoustic pretraining.
2. **Entity & Loanword Discrepancy:**
   - App names, domain commands ("Zalo", "SMS", "điều hoà", "chuyển khoản") and compound entities suffer from acoustic dropout or spelling variation without strong domain biasing.
3. **Spoken Numeral & Diacritic Formatting Gap:**
   - Spoken numbers ("chín giờ kém mười lăm", "hai mươi sáu độ", "ba mươi phần trăm") require systematic Inverse Text Normalization (ITN) and traditional tone placement ("huỷ" vs. "hủy").

---

## 2. Root Cause Summary

- **Root Cause 1 (Acoustic Foundation Limitations in Multilingual Qwen):** Qwen3-ASR was primarily trained on massive Chinese/English corpora with limited Vietnamese speech representation. Even with LoRA SFT, the acoustic encoder struggles with Vietnamese regional accents, colloquial tones, and background noise.
- **Root Cause 2 (Acoustic Pre-eminence of PhoWhisper Large):** PhoWhisper Large (`kzap201/phowhisper-large-zaloai`, 1.54B params) was pre-trained and fine-tuned specifically on Vietnamese speech and Zalo AI Challenge conversational audio, achieving state-of-the-art ~4.0% WER on Vietnamese test sets.
- **Root Cause 3 (Decoupled Decisive Solution):** Moving from fine-tuning an under-adapted multilingual base to deploying a proven, domain-specialized 1.54B Vietnamese acoustic model provides the decisive architectural leap required to approach the < 3.0% WER target.

---

## 3. Phase 2 Architectural Leap: PhoWhisper Large + Domain Prompt Biasing + ITN

1. **Acoustic Engine (`WhisperASREngine` in `src/asr/whisper_engine.py`):**
   - Model: `kzap201/phowhisper-large-zaloai` (1.54B parameters, satisfying $\le$ 2.0B constraint).
   - Robust Processor Construction: Gracefully handles `kzap201/phowhisper-large-zaloai` Hugging Face repository metadata quirks by constructing `processor` using `WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")` and `WhisperTokenizerFast.from_pretrained(model_id, extra_special_tokens={})` if `AutoProcessor.from_pretrained` encounters `AttributeError: 'list' object has no attribute 'keys'`.
2. **Domain Prompt Conditioning:**
   - Whisper decoder autoregressive biasing via `initial_prompt` / `prompt_ids` configured with domain vocabulary (smart home, banking, ride-hailing, common Vietnamese names).
3. **Inverse Text Normalization (ITN) Integration:**
   - Enable `enable_itn=True` by default, routing raw transcribed text through `inverse_text_normalize` to standardize clock times, temperatures, percentages, and diacritic tone placement.
4. **Configuration Standardization:**
   - Provide clean configuration in `configs/e2e_phowhisper_qwen35_4b.yaml` and update chunkformer/whisper pipeline configs.
