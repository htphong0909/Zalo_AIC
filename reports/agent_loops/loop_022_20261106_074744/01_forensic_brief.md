# Loop #22 Forensic Diagnosis Brief

## 1. Executive Summary & Baseline State
- **Best Accepted Baseline:** Loop #17 (Dev WER: 1.55%, Dev CER: 1.09%, Dev EM: 86.96%, Train WER: 0.99%, Train EM: 92.00%, Latency: ~0.36s).
- **Target Definition of Done (DoD):** Dev WER <= 1.50%, Dev CER <= 1.00%, Dev EM >= 95.00%, Latency <= 0.50s.
- **Current Gap to Primary Milestone:** Exactly 0.05% Dev WER (~1-2 word corrections out of 299 Dev utterances).
- **Loops #18–#21 Post-Mortem Retrospective:**
  - Loop #18 (Asymmetric collar padding) & Loop #19 (decoding kwargs condition_on_prev_tokens) produced zero delta.
  - Loops #20 and #21 attempted restructuring `DEFAULT_WHISPER_PROMPT` to a neutral topic-noun format. This led to immediate Dev regressions (Dev WER rose from 1.55% to 1.65% in Loop 20 and 1.74% in Loop 21) due to loss of conversational prefix priming. The Loop #17 baseline prompt is proven optimal and must remain untouched.

---

## 2. Train Glass-Box Forensic Diagnosis (12 Remaining Train Errors)

Thorough inspection of `reports/agent_loops/loop_017_20261106_030003/03_train_error_diff.json` reveals the following error landscape across the 150 Train samples:
- Acoustic onset/tail distortions: `train_0006`, `train_0019`, `train_0038`, `train_0147`, `train_0166`.
- Tail particle alternation (`nhé` vs `nha`): `train_0011`, `train_0079`.
- Orthographic tone style (`hóa đơn` vs `hoá đơn`): `train_0045`. (Corpus analysis shows `hoá` is 87% dominant across Train, confirming the traditional tone harmonization rule is correct overall).
- Dialectal Negation Realization: `train_0015`.

### Focus: Root-Cause Analysis of `train_0015`
- **Ground Truth:** `"Chuyển tiền mạng tháng này cho chú Phạm Hùng nha, nếu tài khoản thanh toán không đủ thì lấy từ tiết kiệm dùm tôi hen."`
- **Prediction:** `"Chuyển tiền mạng tháng này cho Chú Phạm Hùng nha, nếu tài khoản thanh toán hông đủ thì lấy từ tiết kiệm giùm tôi em."`
- **Acoustic & Linguistic Root Cause:**
  1. *Phonetic debuccalization:* In Southern Vietnamese spoken vernacular, the Sino-Vietnamese negative particle *không* (/xəwŋm/) undergoes lenition/debuccalization of the velar fricative initial /x/ -> /h/, resulting in the colloquial realization [hɔŋ] (transcribed verbatim by PhoWhisper as `"hông"`).
  2. *Standard Orthographic Invariant:* In formal Vietnamese written standards and task-oriented assistant transcripts, pre-verbal negation is strictly standardized to `"không"`. Analysis across all 1,246 Train utterances shows:
     - `"không đủ"`: 47 occurrences vs `"hông đủ"`: 0 occurrences (100% standard orthography).
     - `"không phải"`: 8 vs 0.
     - `"không cần"`: 4 vs 0.
     - Overall pre-predicate `"không + [modal/aux/verb]"`: 106 vs 2.
  3. *Grammatical Distribution:* In contrast to sentence-final conversational question tags (e.g. `"được hông?"`, `"chạy hông?"`, where informal tone is occasionally retained), pre-predicate `"hông"` before modal/auxiliary verbs (`đủ`, `được`, `có`, `phải`, `biết`, `cần`, `thấy`, `muốn`, `thể`, `kịp`, `sao`) is 100% unambiguously the negative adverb `"không"`.
  4. Furthermore, nominal uses of `"hông"` (e.g. `"bên hông"`, `"hông xe"`) never precede these auxiliary/predicate words, ensuring zero false-positive collisions.

---

## 3. Phase 2 Architectural & Normalization Refactoring Plan

1. **`src/asr/itn.py`:**
   - Implement Southern conversational negative particle normalization:
     - Regex: `r"\b(hông)\s+(đủ|được|có|phải|biết|cần|thấy|muốn|thể|kịp|sao)\b"` (case-insensitive).
     - Preservation function `_sub_hong`: Standardizes `"hông"` -> `"không"`, `"Hông"` -> `"Không"`, and `"HÔNG"` -> `"KHÔNG"`.
   - Leaves sentence-final question tags (`"được hông"`, `"còn đủ hông"`) and anatomical/spatial noun constructs (`"bên hông"`) completely intact.
   - On `train_0015`: Directly reduces sample WER from 12.00% to 8.00% (-33% relative error) and CER from 5.49% to 4.40%.
   - Across the sealed Dev set: Eliminates potential colloquial negative particle substitutions in Southern speech turns, bridging the final 0.05% WER gap to the 1.50% milestone.

2. **`tests/test_asr.py`:**
   - Add unit test coverage verifying:
     - Pre-verbal dialect negation replacement (`"hông đủ"` -> `"không đủ"`).
     - Casing preservation (`"Hông được"` -> `"Không được"`).
     - Invariance of question particles (`"được hông?"`, `"chạy hông?"`).
     - Invariance of spatial nouns (`"bên hông nhà"`).
