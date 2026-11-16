# Loop #37 Forensic Brief: Prompt Lexical Priming Disambiguation

## 1. Train Glass-Box Forensic Diagnosis
Analysis of the 12 failing samples from the Train glass-box evaluation (`reports/agent_loops/loop_036_20261107_011753/03_train_error_diff.json`) revealed that multiple onset errors stem directly from prompt-induced prior distortion rather than acoustic representation deficiency:
1. **`train_0006`:** GT is *"Nếu giá mua vào..."* but model predicted *"Điều giá thu mua vào..."* (WER 9.52%).
   - Root Cause: `DEFAULT_WHISPER_PROMPT` opened with `"Trợ lý ảo tiếng Việt LifeMate điều khiển thiết bị thông minh, điều hoà..."`. The double appearance of `điều` strongly skewed the first-token prior when encountering the low-energy voiceless /n/ onset of *"Nếu"*.
   - Diagnostic verification: Ablation replacing `"điều khiển"` with `"quản lý"` and adding `"nếu có"` immediately restored *"Nếu giá"* and cut WER from 9.5% to 4.8%.
2. **`train_0147`:** GT is *"Dạ Zalo đi ạ."* but model predicted *"Nhả Zalo đi ạ."* (WER 25.0%).
   - Root Cause: Colloquial affirmative and routing onsets (*"Dạ"*, *"Zalo"*) require contextual co-occurrence priming to prevent Whisper from substituting rhyming nouns or unvoiced plosives.

## 2. Principled Hypothesis
By harmonizing `DEFAULT_WHISPER_PROMPT` to disambiguate conditional onset priors (`"Nếu"` vs `"Điều"`) while reinforcing colloquial conversational anchors (`"nếu có, nếu không đủ"`, `"Dạ, vâng"`), we eliminate prompt-induced hallucinations without compromising high-frequency domain recall.

## 3. Targeted Code Changes
Update `DEFAULT_WHISPER_PROMPT` in `src/asr/whisper_engine.py`:
- Replace `"điều khiển thiết bị"` with `"quản lý thiết bị"`.
- Introduce balanced conditional anchors `"nếu có, nếu không đủ"`.
