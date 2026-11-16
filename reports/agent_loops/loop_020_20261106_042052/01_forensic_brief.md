# Loop #20 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Executive Summary & Baseline State
- **Prior Best Accepted Loop:** Loop #17 (Dev WER: 1.55%, Dev CER: 1.09%, Dev EM: 86.96%, Train WER: 0.99%, Train EM: 92.00%, Latency: 0.348s).
- **Target Definition of Done (DoD):** Dev WER <= 1.50%, Dev CER <= 1.00%, Dev EM >= 95.00%, Latency <= 0.50s.
- **Current Gap to Primary Milestone:** Exactly 0.05% Dev WER (~1 word correction across the 299-sample Sealed Dev set).
- **Recent Loop Post-Mortem:**
  - Loop #18 (asymmetric collar padding) and Loop #19 (temperature/comma suppression) yielded neutral deltas (+0.00% Dev WER) and were rejected by the Pareto Gate to maintain lean architectural hygiene.
  - The baseline remains cleanly at Loop #17.

---

## 2. Train Glass-Box Forensic Diagnosis (12 Remaining Errors in Loop #17)

Detailed forensic investigation of the 12 error utterances in `reports/agent_loops/loop_017_20261106_030003/03_train_error_diff.json` reveals two major, high-impact structural root causes:

### Root Cause 1: Cross-Attention Verb Priming and Leading Verb Hallucination
- **Acoustic & Decoder Mechanism:**
  - In `train_0006`, Ground Truth is:
    `"Nếu giá mua vào vàng SJC hôm nay dưới 123 triệu thì 9 giờ sáng mai nhắc tôi đi mua nhé."`
    Whereas Prediction was:
    `"Điều giá thu mua vào vàng SJC hôm nay dưới 123 triệu thì 9 giờ sáng mai nhắc tôi đi mua nhé."`
  - In Whisper's causal decoder, prompt tokens are prepended as initial KV context. When an utterance begins with a low-energy, unstressed conjunction like `"Nếu"` (/n/ nasal onset), the decoder's cross-attention and causal self-attention weights attend strongly to the salient initial tokens of `DEFAULT_WHISPER_PROMPT`.
  - The prior prompt began with:
    `"Trợ lý ảo tiếng Việt LifeMate điều khiển thiết bị thông minh, điều hoà 26 độ..."`
  - The prominence of the action verb `"điều khiển"` and appliance noun `"điều hoà"` right at the head of the prompt injects strong prior probability on the token `"Điều"`, driving the autoregressive decoder to hallucinate `"Điều giá thu mua vào..."` instead of `"Nếu giá mua vào..."`.
  - Similarly, action verbs such as `"chuyển khoản"`, `"đặt xe"`, `"gọi điện"`, `"tra cứu"`, and `"nhắc nhở"` skew prefix distributions when conversational commands start with conditional clauses or contextual framing.
  - Furthermore, prompt honorifics (`"Dạ, vâng, trợ lý ơi, em ơi, anh ơi, chị ơi."`) completely lacked sentence-final polite particles (`"nhé"`, `"nha"`, `"đi ạ"`), inducing ambiguity on enclitics (`train_0011`: `"nhé"` -> `"nha"`, `train_0147`: `"Dạ Zalo đi ạ."` -> `"Nhả Zalo đi ạ."`, `train_0166`: `"Gọi cho Lan đi ạ..."` -> `"Gọi cho Lan đi..."`).
- **Remediation:**
  - Restructure `DEFAULT_WHISPER_PROMPT` into neutral topic nouns and noun phrases (`"Thiết bị thông minh, điều hoà 26 độ, đèn 30 phần trăm. Tài khoản ngân hàng, ví điện tử. Chuyến xe, tin nhắn Zalo, SMS, cuộc gọi. Thời tiết, lịch hẹn, báo thức."`).
  - Completely eliminate leading action verbs (`"điều khiển"`, `"chuyển khoản"`, `"đặt"`, `"gọi"`, `"tra cứu"`, `"nhắc nhở"`).
  - Add balanced polite enclitics and endings: `"Dạ, vâng, nhé, nha, đi ạ, trợ lý ơi, em ơi, anh ơi, chị ơi."`.

### Root Cause 2: Destructive Forced Tone Harmonization Overrides in ITN
- **Acoustic & Orthographic Mechanism:**
  - In `train_0045`, Ground Truth is:
    `"Trợ lý ơi, chuyển giúp tôi cho anh Trần Văn Tuấn số tiền bằng hóa đơn nước tháng này nhé."`
    Whereas Prediction was:
    `"Trợ lý ơi, chuyển giúp tôi cho anh Trần Văn Tuấn số tiền bằng hoá đơn nước tháng này nhé."`
  - Analysis of Loop 6 raw output confirms that PhoWhisper naturally transcribed `"hóa đơn nước"` (exact match with GT).
  - However, in Loop 7, a deterministic tone harmonization table (`_TONE_PATTERNS`) was introduced into `src/asr/itn.py` mapping `hóa -> hoá`, `hòa -> hoà`, `xóa -> xoá`, `hủy -> huỷ`.
  - An empirical audit of the 1,246 Train split transcripts demonstrates that human annotators use both modern and traditional diacritic placement (`hóa`: 3 vs `hoá`: 25, `hòa`: 2 vs `hoà`: 40, `xóa`: 1 vs `xoá`: 22, and especially `hủy`: 34 vs `huỷ`: 39 - an almost 50/50 split!).
  - Because `compute_wer` and `compute_cer` compare exact Unicode strings without tone normalization, unconditionally overwriting modern tone placement destroys exact matches and artificially inflicts substitution penalties on valid transcriptions.
- **Remediation:**
  - Remove `_TONE_HARMONIZATION_MAP` and `_TONE_PATTERNS` from `src/asr/itn.py`.
  - Allow PhoWhisper's rich acoustic and language modeling priors to preserve the native orthography of the spoken utterance non-destructively.

---

## 3. Phase 2 Architectural & Decoding Implementation Plan

1. **`src/asr/whisper_engine.py`:**
   - Update `DEFAULT_WHISPER_PROMPT` to:
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
2. **`src/asr/itn.py`:**
   - Remove `_TONE_HARMONIZATION_MAP` and `_TONE_PATTERNS`.
   - Remove step 5 (tone harmonization) from `inverse_text_normalize`, making ITN non-destructive for tone placements.
3. **Verification:**
   - Execute mock unit tests via `pytest tests/ -q` to guarantee 100% green test suite.
