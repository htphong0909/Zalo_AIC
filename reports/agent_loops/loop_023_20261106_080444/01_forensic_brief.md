# Loop #23 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Baseline State & Milestone Gap
- **Prior Best Accepted Loop:** Loop #17 (Dev WER: 1.55%, Dev CER: 1.09%, Dev EM: 86.96%, Train WER: 0.99%, Train EM: 92.00%, Latency: 0.346s).
- **Target Definition of Done (DoD):** Dev WER <= 1.50%, Dev CER <= 1.00%, Dev EM >= 95.00%, Latency <= 0.50s.
- **Current Gap to Milestone 1:** Exactly 0.05% Dev WER (~1-2 word corrections across the entire sealed Dev split).
- **Post-Mortem Analysis of Loops #18-#22:**
  - Loop #18 (Asymmetric collar 60ms/90ms) & Loop #19 (temperature/compression ratio fallback): Both produced 0.00% delta on Dev. Changing padding margins when VAD margin is already 150ms has zero effect on token emission, and deterministic greedy decoding already avoided fallback traps.
  - Loops #20-#21 (Prompt restructuring): Modifying prompt format caused +0.10% to +0.19% WER regressions, confirming the current `DEFAULT_WHISPER_PROMPT` is near optimal.
  - Loop #22 (Southern negative particle normalization): Caused +0.01% Dev WER regression due to dialect annotation inconsistencies in ground truth.

---

## 2. Train Glass-Box Forensic Diagnosis (12 Remaining Errors in Loop #17)

Detailed analysis of the 12 error utterances in `reports/agent_loops/loop_017_20261106_030003/03_train_error_diff.json` reveals that the remaining errors are primarily driven by **myopic greedy decoding local optima traps**:

### Root Cause: Greedy Decoding ($B=1$) Myopia on Near-Tie Token Probabilities
1. **Semantic/Syntactic Path Divergence:**
   - `train_0006`: Ground truth `"Nếu giá mua vào vàng SJC..."` was decoded as `"Điều giá thu mua vào vàng SJC..."`. At step $t=1$, greedy search selected `"Điều"` over `"Nếu"` due to a microscopic logit difference, permanently committing to a divergent syntactic path.
   - `train_0019`: Ground truth `"Chị ơi, giúp tôi đặt..."` was decoded as `"Chị ơi, tôi cần đặt..."`. High-frequency assistant prior `"tôi cần"` out-prioritized `"giúp tôi"` at step $t=3$ in greedy selection.
2. **Contextual Recency & Lexical Confusion:**
   - `train_0079`: Ground truth `"...trong năm ngày tới... cái nào trên 500 nghìn..."` was decoded as `"...500 ngày..."`. Recency bias from the earlier `"năm ngày tới"` primed `"ngày"`, causing greedy decoding to ignore acoustic evidence for `"nghìn"`.
   - `train_0147`: Ground truth `"Dạ Zalo đi ạ."` was decoded as `"Nhả Zalo đi ạ."` (onset consonant ambiguity committed to nonsensical verb `"Nhả"`).
   - `train_0208`: Ground truth `"...dùng tiền tiết kiệm chuyển thay."` was decoded as `"...dùng tiết kiệm chuyển thôi."` (omission of `"tiền"` and confusion of `"thay"` -> `"thôi"`).

### Why Constrained 2-Beam Search ($B=2$) Resolves the Bottleneck
- In greedy search ($B=1$), $\hat{y}_t = \arg\max_{y} P(y \mid y_{<t}, X)$ irrevocably discards all competing hypotheses at every step.
- In 2-beam search ($B=2$), the top-2 hypothesis branches are tracked in parallel. When downstream acoustic evidence (e.g., conditional `"thì"` confirming `"Nếu"`, or clear spectral energy for `"nghìn"`) enters the decoder, the joint acoustic-language sequence score $\sum \log P(y_t \mid y_{<t}, X)$ enables the true hypothesis to overtake the premature local prior.
- **Why Loop #05 Failed vs Why Loop #23 Will Succeed:**
  - In Loop #05, beam search failed because:
    1. `repetition_penalty` was 1.10 (which Loop #15 proved severely harms Dev EM).
    2. Over-engineered keyword prompting induced hallucination.
    3. Missing acoustic pre-roll collar caused Conv1D boundary distortions.
    4. Inference was unbatched (sequential), causing 1.73s latency.
  - In Loop #23:
    1. Acoustic foundation is pristine: 60ms bilateral collar, -35 dB VAD trimming, 0.95 peak norm.
    2. Calibrated `repetition_penalty=1.03` prevents looping while protecting valid repetitions.
    3. Batched inference (`batch_size=8`) with SDPA flash-attention runs 16 decoder sequences ($8 \times 2$) in parallel on GPU, maintaining latency well under the 0.50s DoD budget (~0.40s estimated).
    4. `early_stopping=True` terminates beam search immediately once 2 full sentences complete.

---

## 3. Phase 2 Architectural & Decoding Implementation Plan

1. **`src/asr/whisper_engine.py`:**
   - In `WhisperASREngine.__init__`:
     - Add `num_beams: int = 2` parameter.
   - In `WhisperASREngine._build_gen_kwargs`:
     - When `num_beams > 1`, set `gen_kwargs["num_beams"] = int(self.num_beams)` and `gen_kwargs["early_stopping"] = True`.
2. **`tests/test_asr.py`:**
   - Update `DummyEngine` and unit test assertions to verify `num_beams=2` and `early_stopping=True`.
