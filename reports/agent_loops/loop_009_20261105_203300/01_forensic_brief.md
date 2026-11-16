# Loop 009 Forensic Diagnosis & Architectural Refactoring Brief

**Loop ID:** 9  
**Hypothesis:** Parameter-Efficient Acoustic Fine-Tuning (PEFT LoRA r=16, alpha=32) on PhoWhisper Large to eliminate dialect phoneme confusions and overcome the prompt-biasing acoustic ceiling  
**Author:** Lead ASR & Deep Learning Researcher  
**Target:** Train Glass-Box Forensic Analysis & Precision Recovery (DoD Target: Dev WER <= 1.50%, EM >= 95.00%, Latency <= 2.0s)

---

## 1. Train Glass-Box Forensic Diagnosis

### A. Failure Mode Analysis of Zero-Shot Prompt Biasing (Loop #08 Forensic Retrospective)
In Loop #08, expanding `DEFAULT_WHISPER_PROMPT` with domain entities (`"điều hoà 26 độ"`, `"hoá đơn"`, `"đặt xe ô tô bốn chỗ 4 chỗ"`, `"rút tiết kiệm"`) resulted in severe regression:
- **Dev EM collapsed** from **87.29% (Loop #07)** to **83.95% (Loop #08)** (delta Dev EM = -3.34%, delta Dev WER = +0.17%).
- **Attention Bleeding Phenomenon:** Autoregressive Whisper decoders attend to initial prompt tokens across all generation steps. When domain phrases were injected, decoder cross-attention over-relied on prompt priors over genuine acoustic encoder representations:
  - `train_0166`: GT `"Gọi cho Lan đi ạ, với lại đặt báo thức..."` was corrupted into `"Mở cho Lan điều hoá đơn quá thức..."`, directly hallucinating `"điều hoà"` and `"hoá đơn"` from the prompt!
  - `train_0070`, `train_0076`, `train_0084`, `train_0111`, `train_0127`, `train_0130`: Spoken Vietnamese numbers `"bốn chỗ"` were artificially converted to `"4 chỗ"` due to the prompt collocation `"bốn chỗ 4 chỗ"`, directly causing WER/EM degradation.
- **The Acoustic Ceiling:** Text prompt biasing cannot correct true acoustic/phonetic confusions. An autoregressive language prior cannot force the decoder to correct an acoustic misrecognition without risking false-positive hallucinations on every utterance where the acoustic evidence is slightly ambiguous.

### B. Remaining Train Error Taxonomy on Loop #07 Baseline
Examination of the remaining 14 Train errors on the best baseline (Loop #07) demonstrates four distinct acoustic/phonetic failure modes:
1. **Dialect Consonant Mergers & Allophonic Variations:**
   - `train_0087`: GT `"rút tiết kiệm"` transcribed as `"gút tiết kiệm"`. In Southern and Central Vietnamese dialects, the alveolar liquid /r/ is frequently realized as a voiced velar fricative [ɣ] or plosive [g], which the zero-shot model confuses with /g/ ("gút").
   - `train_0147`: GT `"Dạ Zalo"` transcribed as `"Nhạc Zalo"`. Low-energy colloquial discourse opener /z/ (Northern) or /j/ (Southern) in `"Dạ"` is confused with coronal nasal /ɲ/ in `"Nhạc"`.
2. **Colloquial Particle & Discourse Marker Mismatch:**
   - `train_0011`, `train_0079`: GT `"nhé"` vs predicted `"nha"`. Spoken dialect variation between Northern standard `"nhé"` and Southern colloquial `"nha"`.
3. **Acoustic Unit Drop & Suffix Truncation:**
   - `train_0083`: GT `"26 độ"` vs predicted `"26"`. Fast spoken speech drops acoustic energy on the monosyllable `"độ"`.
4. **Loanword & Proper Name Contextual Fusion:**
   - `train_0015`: GT `"dùm tôi hen"` vs predicted `"giùm tôi em"`. Dialectal closing particle `"hen"` misidentified due to lack of conversational acoustic tuning.

---

## 2. Theoretical & Mathematical Basis for PEFT LoRA Adaptation

To overcome the prompt-biasing ceiling without causing attention bleeding or hallucinations, we must adapt the **acoustic representations** themselves:

1. **Parameter-Efficient LoRA on Attention Projections:**
   For a frozen pre-trained projection matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA decomposes the weight update into two low-rank matrices:
   $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B \cdot A$$
   where $A \in \mathbb{R}^{r \times k}$ (initialized with Gaussian noise) and $B \in \mathbb{R}^{d \times r}$ (initialized to zero), with rank $r=16$ and scaling factor $\alpha=32$.
   
2. **Target Modules (`q_proj`, `v_proj`):**
   By injecting LoRA into the query and value projection layers of both the acoustic encoder and decoder self/cross-attention:
   - The **acoustic encoder** learns the phonetic space of LifeMate Vietnamese speech (mapping Southern [ɣ] to /r/, unstressed discourse markers [za] to "Dạ").
   - The **cross-attention layers** learn LifeMate domain semantic associations natively in the latent space rather than relying on brittle text prompts.

3. **Strict Constraints Met:**
   - Trainable parameters are ~15M (<0.5% of the 1.54B PhoWhisper Large model), completely preserving general linguistic robustness and preventing catastrophic forgetting.
   - Total model size remains strictly $\le 1.55\text{B}$ parameters (well below the 2.0B ASR ceiling and 6.0B total system ceiling).
   - Once trained, the LoRA adapter can be dynamically loaded or permanently fused into the base weights via `merge_and_unload()`, yielding **zero inference latency overhead** ($\le 0.5\text{s}$ per sample).

---

## 3. Engineering & Refactoring Plan for Loop #09

1. **ASR Engine Update (`src/asr/whisper_engine.py`):**
   - Add parameter `adapter_id: str | Path | None = None` to `WhisperASREngine.__init__`.
   - Support loading LoRA weights via `peft.PeftModel.from_pretrained(self.model, adapter_id)` with optional `merge_and_unload()`.
   - Maintain full backward compatibility for zero-shot or existing baseline configurations.

2. **Training Pipeline (`scripts/train_phowhisper_lora.py`):**
   - Load `kzap201/phowhisper-large-zaloai`.
   - Setup `LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")`.
   - Prepare dataset strictly from `train_ids` in `configs/splits/dev_v1.json` and `data/train/train.jsonl` (Sealed Blind Box Invariant: zero dev leakage).
   - Use HuggingFace `Seq2SeqTrainer` with AdamW ($1\times 10^{-4}$ learning rate), cosine learning rate schedule, FP16 mixed precision, gradient checkpointing.
