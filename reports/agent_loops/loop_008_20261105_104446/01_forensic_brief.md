# Loop 008 Forensic Diagnosis & Architectural Refactoring Brief

**Loop ID:** 8  
**Hypothesis:** Enhanced LifeMate domain entity prompt conditioning with selective tone conflict de-biasing  
**Author:** Lead ASR & Deep Learning Researcher  
**Target:** Train Glass-Box Forensic Analysis & Precision Recovery (DoD Target: Dev WER <= 1.50%, EM >= 95.00%, Latency <= 2.0s)

---

## 1. Train Glass-Box Forensic Diagnosis (Analysis of Residual Train Errors)

Detailed acoustic and orthographic investigation of the remaining Train errors:

### A. Conversational Opening Particles & Onset Phonetic Confusions (`train_0147`, `train_0019`)
- **`train_0147`:** Acoustic onset confusion between dental voiced fricative /z/ (or glide /j/ in Southern/Central dialects) in casual discourse marker `"Dạ"` vs coronal nasal /ɲ/ in `"Nhạc"` (`"Nhạc Zalo"` vs ground truth `"Dạ Zalo"`).
- **`train_0019`:** Low-energy casual conversational vocative `"Chị ơi"` was acoustically misheard by the decoder as `"Chị Huyết"` due to lack of conversational turn-taking prefix context.
- **Root Cause & Acoustic Origin:** Whisper's autoregressive decoder relies heavily on initial prompt conditioning to calibrate language model priors for initial tokens. Without conversational discourse markers in `DEFAULT_WHISPER_PROMPT`, the decoder assigns lower prior probabilities to colloquial openers (`"Dạ"`, `"vâng"`, `"em nghe"`, `"Chị ơi"`, `"anh ơi"`, `"em ơi"`), defaulting to aberrant nouns.
- **Principled Solution:** Prepend high-frequency conversational openings and vocatives (`"Trợ lý ảo giọng nói tiếng Việt LifeMate. Dạ, vâng, em nghe. Chị ơi, anh ơi, em ơi, trợ lý ơi:..."`) to the declarative prompt to provide appropriate conversational conditioning.

### B. Domain Entity Collocations & Contextual Disambiguation (`train_0083`, `train_0087`)
- **`train_0083`:** Ground truth contained `"26 độ"`, but prediction truncated or failed to recognize unit continuation (`"26."`).
- **`train_0087`:** Phonetic confusion on alveolar liquid /r/ vs velar plosive /g/ (`"gút tiết kiệm"` instead of `"rút tiết kiệm"`).
- **Root Cause & Decoder Dynamics:** Specialized Vietnamese domain collocations (banking, smart home, mobility) have high semantic mutual information. When `DEFAULT_WHISPER_PROMPT` includes key collocations (`"điều hoà 26 độ"`, `"rút tiết kiệm"`, `"tiền mạng, tiền điện, tiền nước, ví điện tử"`, `"đặt xe ô tô bốn chỗ 4 chỗ"`), cross-attention layers attend to these entity co-occurrences, resolving acoustic ambiguity in noisy/dialectal clips.

### C. Tone Conflict in Inverse Text Normalization (`train_0045`)
- **Error in `train_0045`:** Ground truth transcript is `"hóa đơn nước"`, but ITN post-processing forcibly rewrote it to `"hoá đơn nước"`.
- **Corpus Linguistic Analysis:** While `hòa/hoà`, `xóa/xoá`, and `hủy/huỷ` show near-unanimous skew towards traditional diacritics in the training set (>95%), the root `"hóa"` vs `"hoá"` exhibits balanced bimodal distribution (approx. 250 instances of `"hóa"` vs 360 instances of `"hoá"`).
- **Principled Solution:** Remove `"hóa": "hoá"` from `_TONE_HARMONIZATION_MAP` in `src/asr/itn.py`. Selective de-biasing prevents deterministic artificial regressions on samples with modern diacritic conventions while maintaining consensus harmonization for uncontested pairs.

---

## 2. Architectural Refactoring & Optimization Details

1. **`src/asr/whisper_engine.py`:**
   - Update `DEFAULT_WHISPER_PROMPT`:
     ```python
     DEFAULT_WHISPER_PROMPT = (
         "Trợ lý ảo giọng nói tiếng Việt LifeMate. Dạ, vâng, em nghe. "
         "Chị ơi, anh ơi, em ơi, trợ lý ơi: điều khiển nhà thông minh (bật đèn, tắt đèn phòng khách, phòng ngủ, bếp, điều hoà 26 độ, quạt), "
         "tài chính ngân hàng (chuyển khoản, tiền mạng, tiền điện, tiền nước, ví điện tử, nạp tiền, rút tiết kiệm, rút tiền, hoá đơn, vàng SJC), "
         "nhắc nhở lịch hẹn khám răng, đặt xe ô tô bốn chỗ 4 chỗ, liên lạc Zalo SMS gọi điện cho Minh, Lan, Tuấn, Hương, Bố, Mẹ, Phạm Hùng."
     )
     ```
2. **`src/asr/itn.py`:**
   - Remove `"hóa": "hoá"` from `_TONE_HARMONIZATION_MAP` to prevent forced tone conflicts on bimodal corpus distributions.
3. **`tests/test_itn.py`:**
   - Update test expectations to reflect selective tone harmonization (`hòa` -> `hoà`, `xóa` -> `xoá`, `hủy` -> `huỷ`, leaving `hóa` unforced).
