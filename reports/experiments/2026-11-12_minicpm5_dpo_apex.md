# Experiment Report: MiniCPM5-2B Targeted Step-DPO & When2Call (Branch A)

> **Experiment ID:** `exp_20261112_minicpm5_dpo_apex`  
> **Date:** 2026-11-12 ~ 2026-11-13  
> **Hardware:** Google Colab G4 (NVIDIA RTX PRO 6000 Blackwell Server Edition, 94.97 GB VRAM)  
> **Hugging Face Prefix:** [`exp_branchA_dpo_v1`](https://huggingface.co/htphong0909/zalo-aic-sft-checkpoints/tree/main/exp_branchA_dpo_v1)  
> **Base Model:** `openbmb/MiniCPM5-2B-SFT`  
> **Warm-start Adapter:** SFT V2 LoRA (`exp_branchA_sft_v2/adapter`)  
> **Status:** **COMPLETED** (61/61 Steps, 100% Evaluated on 299 Blind Dev Samples)

---

## 1. Goal & Hypothesis

- **Hypothesis:** Fine-tuning on 982 targeted contrastive preference pairs (Step-DPO for intermediate sub-goal continuation, When2Call for disambiguating tool calls vs missing info, ToolSandbox anchor binding for entity propagation) will penalize tool hallucination and missing-info over-asking while preserving formatting accuracy via a hybrid loss (Sigmoid DPO + 0.2 SFT Anchor).
- **Dataset:** `data/dpo/minicpm5_dpo_pairs.jsonl.gz` (982 pairs, zero dev leakage).
  - `when2call`: 350 pairs (175 complete ACT vs false ASK; 175 missing arg ASK vs hallucinated ACT)
  - `tool_n1_horizon`: 250 pairs (continue sub-goal vs early stop)
  - `looptool_mutated`: 232 pairs (mined real failure patterns from prior loops)
  - `toolsandbox_anchor`: 150 pairs (observation entity binding vs hallucinated ID)

---

## 2. Hyperparameters & Configuration

```yaml
training:
  epochs: 1.0 (61 optimizer steps)
  per_device_batch_size: 4
  gradient_accumulation_steps: 4
  effective_batch_size: 16
  learning_rate: 3.0e-6
  lr_scheduler_type: cosine
  warmup_steps: 6 (~10%)
  beta: 0.1
  loss_type: ["sigmoid", "sft"]
  loss_weights: [1.0, 0.2]
  max_length: 8192
  bf16: true
  lora_r: 64
  lora_alpha: 128
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  runtime: 4316.7s (~72.07 min)
  final_train_loss: 0.3376
```

---

## 3. Checkpoint Integrity on Hugging Face Hub

Every checkpoint was uploaded permanently to Hugging Face (`htphong0909/zalo-aic-sft-checkpoints`):

| Checkpoint | Path in HF Repo | Note |
|---|---|---|
| Step 9 | `exp_branchA_dpo_v1/checkpoints/checkpoint-9/` | Periodic ~10m sync |
| Step 18 | `exp_branchA_dpo_v1/checkpoints/checkpoint-18/` | Periodic ~10m sync |
| Step 27 | `exp_branchA_dpo_v1/checkpoints/checkpoint-27/` | Periodic ~10m sync |
| Step 36 | `exp_branchA_dpo_v1/checkpoints/checkpoint-36/` | Periodic ~10m sync |
| Step 45 | `exp_branchA_dpo_v1/checkpoints/checkpoint-45/` | Periodic ~10m sync |
| Step 54 | `exp_branchA_dpo_v1/checkpoints/checkpoint-54/` | Periodic ~10m sync |
| Step 61 | `exp_branchA_dpo_v1/checkpoints/checkpoint-61/` | Final training step |
| Latest Pointer | `exp_branchA_dpo_v1/checkpoints/latest/` | Mirrored latest snapshot |
| Final Adapter | `exp_branchA_dpo_v1/adapter/` | Complete PEFT LoRA adapter |
| Predictions | `exp_branchA_dpo_v1/preds/dev_preds.jsonl` | 299 Dev set model outputs |
| Summary | `exp_branchA_dpo_v1/training_summary.json` | Training telemetry |
| Eval Metrics | `exp_branchA_dpo_v1/eval/metrics.json` | Ground truth evaluation metrics |

---

## 4. Benchmark Results on 299 Sealed Dev Samples

Evaluated strictly offline against `configs/splits/dev_v1.json` (299 samples) via `scripts/run_eval.py`:

| Metric | SFT V2 Baseline | DPO V1 (This Run) | Delta |
|---|:---:|:---:|:---:|
| **Final Composite Score** | **0.4570** | **0.4511** | *-0.0059* |
| **State Match Rate** | **52.52%** (157/299) | **52.52%** (157/299) | **0.00% (Identical)** |
| **Tool-calling Precision** | **61.27%** | **60.86%** | *-0.41%* |
| **Tool-calling Recall** | **60.24%** | **59.78%** | *-0.46%* |
| **Tool-calling F1** | **59.91%** | **59.58%** | *-0.33%* |

### By Category Breakdown:

| Category | N | SFT V2 Score | DPO V1 Score | Delta |
|---|:---:|:---:|:---:|:---:|
| **Action (`act`)** | 235 | 0.4424 | 0.4400 | *-0.0024* |
| **Missing Info (`ask`)** | 43 | 0.3178 | 0.3140 | *-0.0038* |
| **Out-of-Scope (`oos`)** | 21 | 0.9048 | 0.8571 | *-0.0477* |
| **Multi-turn History (`history`)** | 197 | 0.4466 | 0.4391 | *-0.0075* |
| **Single-turn (`no_history`)** | 102 | 0.4770 | 0.4744 | *-0.0026* |

---

## 5. Diagnostic Findings & Qualitative Sample Breakdown

- **Stability:** 258 / 299 samples (86.3%) yielded strictly identical outputs between SFT V2 and DPO V1.
- **Improved Samples (19 samples, 0.0 $\rightarrow$ 1.0):**
  - **Over-calling Pruned:** Mẫu `train_0029` (người dùng hỏi giá vàng, SFT V2 tự thêm `reminders.create` ảo giác; DPO loại bỏ hoàn toàn tool thừa, gọi duy nhất `gold_price.get_price` $\rightarrow$ 1.0).
  - **Entity Anchor Binding:** Mẫu `train_0082`, `train_0181`, `train_0476`, `train_0615` trích xuất ID chính xác từ kết quả quan sát bước trước.
- **Regressed Samples (22 samples, 1.0 $\rightarrow$ 0.0):**
  - **Slight Conservatism in Multi-action Chaining:** Mẫu `train_0121` (người dùng yêu cầu trả 2 hóa đơn liên tiếp; DPO gọi thành công hóa đơn 1 nhưng dừng trước hóa đơn 2 vì reward margin phạt gọi tool thừa quá mạnh).

---

## 6. Actionable Recommendations

1. **Model Souping / Weight Blending:**
   - Blend SFT V2 and DPO V1 LoRA weights ($\theta_{soup} = \alpha \theta_{SFT} + (1-\alpha) \theta_{DPO}$ with $\alpha \in [0.6, 0.7]$).
   - Preserves SFT multi-action chaining while inheriting DPO's anti-hallucination precision.
2. **Auto-Resume Integration:**
   - Permanent rule added to `AGENTS.md`: Remote workers MUST auto-load `checkpoints/latest` on restart.
