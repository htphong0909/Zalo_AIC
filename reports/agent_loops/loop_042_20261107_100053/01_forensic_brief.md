# Forensic Diagnosis & Hypothesis Brief: Loop #042

## 1. Problem Diagnosis (From Loops #38, #40, #41)
- **Baseline (Loop #38):** Dev WER: 1.50% (achieved <= 1.50% DoD), Dev CER: 1.03% (target <= 1.00%, only 0.03% away), Dev EM: 86.96%.
- **Findings from Loop #40 & #41:**
  - Loop #40 (LoRA r=8, alpha=16, 2 epochs) reduced Train WER from 0.82% -> 0.77%, fixing acoustic confusions ('500 ngày' -> '500 nghìn', 'giùm tôi' -> 'dùm tôi'), while maintaining Dev WER at 1.50% without any regression.
  - Loop #41 (LoRA r=8, 6 epochs) dropped Train WER to 0.46% (Train EM 93.33%), but 6 epochs overfitted the 150 training samples, causing Dev WER to regress to 1.74%.
  - Zero-shot external CTC models (`khanhld/chunkformer`, `wav2vec2-vlsp2020`) cannot be used as standalone drop-in decoders due to missing HF processor classes or missing KenLM decoders.
  - Zero-shot `whisper-large-v3-turbo` exhibits severe dialect repetition loops and hallucination on Vietnamese commands.

## 2. Hypothesis (Loop #042)
- **Hypothesis:** Hyper-compact Dual-Tower Acoustic LoRA (r=4, alpha=8, dropout=0.1, lr=8e-5, 3 epochs, weight_decay=0.01) regularizes phonetic cross-attention to prevent speaker memorization while grounding regional dialect shifts.
- **Architectural & Regularization Details:**
  - Rank r=4, alpha=8: reduces trainable parameters by 50% compared to Loop #40 (from 2.62M down to ~1.31M parameters, strictly 0.08% of backbone).
  - Target modules: 128 acoustic modules (`encoder.layers.*.self_attn` and `decoder.layers.*.encoder_attn`). Autoregressive decoder self-attention and FFN remain 100% frozen.
  - Regularization: `lora_dropout=0.1`, `weight_decay=0.01`, conservative `learning_rate=8e-5`.
  - Epochs: 3 epochs (~57 optimization steps), striking the exact sweet spot between underfitting (2 epochs) and overfitting (6 epochs).
