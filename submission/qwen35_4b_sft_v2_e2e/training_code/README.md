# Training & Reproducibility Guide (`Qwen3.5-4B SFT V2 + PhoWhisper-Large`)

Thư mục này chứa toàn bộ mã nguồn xử lý dữ liệu, tổng hợp mẫu, huấn luyện SFT V2, cắt tỉa trọng số text-only cho suy luận <6B tham số, và mã đánh giá. **Base model (`Qwen/Qwen3.5-4B`, `vinai/PhoWhisper-large`) không được đóng gói trong Docker submission**; quá trình tái lập sẽ tải trực tiếp từ Hugging Face Hub với `seed=42`.

## 1. Đường dẫn Hugging Face (Dữ liệu & Trọng số)

| Thành phần | Vai trò | Hugging Face URL | Ghi chú |
|---|---|---|---|
| **LLM Base Model** | Tiền huấn luyện gốc | https://huggingface.co/Qwen/Qwen3.5-4B | Không đóng gói trong container |
| **LLM SFT V2 Dataset** | Tập train SFT (`4,954` mẫu sạch, loại trừ `dev_v1.json`) | https://huggingface.co/htphong0909/zalo-aic-sft-checkpoints/blob/main/exp_qwen35_sft_v2/assets/qwen35_train_sft_v2.jsonl.gz | Tổng hợp từ `data/train` + `src/data_gen/` *(chuyển repo sang public trước khi BTC reproduce)* |
| **LLM SFT V2 LoRA & Merged** | Adapter LoRA & bản merged pure-text (`4,205,751,296` params) | https://huggingface.co/htphong0909/zalo-aic-sft-checkpoints/tree/main/exp_qwen35_sft_v2 | Thư mục `merged/` dùng cho inference offline *(chỉ đọc với `HF_TOKEN` nếu repo đang private)* |
| **ASR Base Model** | Tiền huấn luyện gốc | https://huggingface.co/vinai/PhoWhisper-large | Revision `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb` |
| **ASR Fine-tuned Model** | Trọng số nhận dạng tiếng Việt (`1,543,304,960` params) | https://huggingface.co/kzap201/phowhisper-large-zaloai | Dùng trực tiếp cho inference offline (`v2`, đã public) |

> Lưu ý dữ liệu gốc cuộc thi (`train/`, `public_test/`) được cung cấp bởi BTC LifeMate ZAC 2026 và đặt tại `--root data/`.

## 2. Cấu trúc `training_code/`

```text
training_code/
├── README.md
├── configs/
│   ├── qwen35_sft_v2.yaml              # Siêu tham số SFT V2 (LoRA r=32, alpha=64, epochs=1.5, lr=5e-5, seed=42)
│   └── splits/dev_v1.json              # 299 mẫu dev bị loại khỏi tập train (zero leakage)
├── scripts/
│   ├── build_qwen38_sft_data.py        # Tạo tập qwen35_train_sft_v2.jsonl.gz với 7 kỹ thuật tổng hợp & kiểm tra rò rỉ
│   ├── train_qwen38_sft.py             # Huấn luyện LoRA SFT + auto merge + export pure-text Qwen3_5ForCausalLM
│   ├── prune_qwen35_text_only.py       # Loại bỏ vision tower & MTP head giữ nguyên chất lượng text
│   ├── train_phowhisper_lora.py        # Script adaptation âm học LoRA encoder cho PhoWhisper
│   ├── run_asr.py                      # Chạy & chấm ASR
│   ├── run_agent.py                    # Chạy agent oracle / e2e
│   ├── run_eval.py                     # Chấm điểm Tool F1 + State Hash Match
│   └── count_params.py                 # Kiểm tra ngân sách tham số từ safetensors headers
└── src/                                # Toàn bộ source (`core/`, `asr/`, `agent/`, `data_gen/`, `env/`, `eval/`)
```

## 3. Các bước tái lập từ đầu (`seed=42`)

### Bước 1: Tạo tập dữ liệu SFT V2 (4,954 mẫu, zero dev leakage)

```bash
python3 scripts/build_qwen38_sft_data.py \
  --split train \
  --root data \
  --exclude-ids configs/splits/dev_v1.json \
  --out-dir data/sft \
  --out-stem qwen35_train_sft_v2 \
  --seed 42
```

### Bước 2: Huấn luyện LoRA SFT V2 trên `Qwen/Qwen3.5-4B` & xuất trọng số pure-text

- Siêu tham số cố định trong `configs/qwen35_sft_v2.yaml`:
  - `seed: 42`, `max_seq_length: 16384`, `learning_rate: 5e-5`, `num_train_epochs: 1.5`, `per_device_train_batch_size: 2`, `gradient_accumulation_steps: 8` (effective batch = 16).
  - LoRA `r=32`, `lora_alpha=64`, `lora_dropout=0.05` trên cả attention (`q_proj, k_proj, v_proj, o_proj`), MLP (`gate_proj, up_proj, down_proj`) và GatedDeltaNet (`in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj`).
  - Huấn luyện hoàn tất sinh ra `merged/` thuần `Qwen3_5ForCausalLM` (`4.21B` tham số).

```bash
python3 scripts/train_qwen38_sft.py \
  --config configs/qwen35_sft_v2.yaml \
  --dataset data/sft/qwen35_train_sft_v2.jsonl.gz \
  --out-dir experiments/exp_qwen35_sft_v2 \
  --seed 42
```

### Bước 3: ASR `kzap201/phowhisper-large-zaloai`

- Được fine-tune toàn phần (`FP32` master weights + `BF16` autocast) từ `vinai/PhoWhisper-large` (revision `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb`) trên tập audio train LifeMate (`4,593` clips, loại bỏ hoàn toàn clips dev), `per_device_train_batch_size=32`, `lr=1e-5`, `reduce_lr_on_plateau`, `seed=42`, chọn checkpoint tại step `2592` (epoch `18`). Xem metadata đầy đủ trong `release.json` tại Hugging Face repo `kzap201/phowhisper-large-zaloai`.
- Ngoài ra `scripts/train_phowhisper_lora.py` đi kèm phục vụ tinh chỉnh encoder LoRA bổ sung nếu cần.

### Bước 4: Đánh giá suy luận E2E trên Dev 299

```bash
python3 scripts/run_asr.py \
  --split train --root data --ids configs/splits/dev_v1.json \
  --engine whisper --model-id kzap201/phowhisper-large-zaloai --dtype float16 \
  --no-context-bias --nbest-beams 3 --nbest-max-delta 0.65 \
  --out asr_dev299/dev299_transcripts.jsonl

python3 scripts/run_agent.py \
  --config ../configs/e2e_qwen35_4b_sft_v2.yaml \
  --split train --root data --ids configs/splits/dev_v1.json \
  --transcripts asr_dev299/dev299_transcripts.jsonl \
  --out preds_dev299.jsonl

python3 scripts/run_eval.py \
  --predictions preds_dev299.jsonl --root data --ids configs/splits/dev_v1.json \
  --out eval_dev299
```
