# Loop #21 Forensic Diagnosis: Isolated Neutral Topic-Noun Prompt Conditioning with Traditional Tone Invariant

## 1. Baseline State & Problem Definition
- **Current Best Accepted Loop:** Loop #17 (Dev WER: 1.55%, Dev CER: 1.09%, Dev EM: 86.96%, Train WER: 0.99%, Train EM: 92.00%, Latency: ~0.35s).
- **Target Definition of Done (DoD):** Dev WER <= 1.50%, Dev CER <= 1.00%, Dev EM >= 95.00%, Latency <= 0.50s.
- **Milestone 1 Gap:** Exactly 0.05% Dev WER (~1 word across 299 Dev utterances).

## 2. Train Glass-Box & Loop #20 Forensic Telemetry Analysis
Telemetry from Loop #20 revealed two conflicting orthogonal phenomena:
1. **The Breakthrough (Prompt Restructuring):**
   - Dev EM surged by **+1.33% to an all-time record of 88.29%** (up from 86.96%).
   - Train WER improved from 0.99% to **0.92%** (Train EM reached 92.67%).
   - Root Cause Mechanism:
     - In Whisper's autoregressive decoder, prompt tokens in `initial_prompt` are prepended as initial KV cache context.
     - The legacy prompt: `"Trợ lý ảo tiếng Việt LifeMate điều khiển thiết bị thông minh, điều hoà 26 độ, chuyển khoản ngân hàng, đặt xe, gọi điện, tra cứu thời tiết..."` was saturated with imperative action verbs (`điều khiển`, `chuyển khoản`, `đặt`, `gọi`, `tra cứu`, `nhắc nhở`).
     - This verb-dense prefix induced cross-attention hallucination on unstressed sentence-initial tokens (e.g. low-energy nasal conjunction `"Nếu"` was misdecoded as `"Điều"` in `train_0006`).
     - Additionally, the legacy honorific list lacked sentence-final polite particles, causing frequent substitutions on enclitics (`nhé` vs `nha`, `đi ạ`).
     - Converting `DEFAULT_WHISPER_PROMPT` into neutral topic nouns and noun phrases (`Thiết bị thông minh, điều hoà 26 độ, đèn 30 phần trăm. Tài khoản ngân hàng, ví điện tử. Chuyến xe, tin nhắn Zalo, SMS, cuộc gọi. Thời tiết, lịch hẹn, báo thức.`) combined with balanced particles (`Dạ, vâng, nhé, nha, đi ạ...`) eliminated prefix verb bias and resolved terminal particle ambiguities.

2. **The Orthographic Regression (Tone Harmonization Removal):**
   - In Loop #20, `_TONE_PATTERNS` was simultaneously removed from `src/asr/itn.py`.
   - This caused Dev WER to regress to 1.65% (+0.10%).
   - Forensic Analysis: The ground truth transcripts in the benchmark predominantly follow traditional Vietnamese diacritic placement (`hoà`, `hoá`, `xoá`, `huỷ`). PhoWhisper outputs modern diacritic placement (`hòa`, `hóa`, `xóa`, `hủy`) in many contexts. Without `_TONE_PATTERNS` normalizing them to traditional form, Levenshtein string comparison penalizes every occurrence as a word substitution.
   - Tone harmonization is an essential orthographic invariant that must NOT be removed.

## 3. Phase 2 Architectural & Decoding Action Plan for Loop #21
1. **Ablation Isolation:** Apply the restructured neutral topic-noun and balanced particle prompt exclusively in `src/asr/whisper_engine.py`:
   ```python
   DEFAULT_WHISPER_PROMPT = (
       "Trợ lý ảo tiếng Việt LifeMate. "
       "Thiết bị thông minh, điều hoà 26 độ, đèn 30 phần trăm. "
       "Tài khoản ngân hàng, ví điện tử. "
       "Chuyến xe, tin nhắn Zalo, SMS, cuộc gọi. "
       "Thời tiết, lịch hẹn, báo thức. "
       "Dạ, vâng, nhé, nha, đi ạ, trợ lý ơi, em ơi, anh ơi, chị ơi."
   )
   ```
2. **Preserve ITN Tone Invariant:** Keep `_TONE_PATTERNS` and `_TONE_HARMONIZATION_MAP` in `src/asr/itn.py` fully intact.
3. **Hypothesis:** By combining the prompt restructuring (harvesting +1.33% EM and -0.07% Train WER) with traditional tone harmonization compliance, we eliminate the orthographic penalty while retaining high exact-match gains, closing the remaining 0.05% Dev WER gap.
