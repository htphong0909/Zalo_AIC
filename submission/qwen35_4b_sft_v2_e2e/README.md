# 67Cat — ZAC 2026 LifeMate Voice Assistant Submission (`Qwen3.5-4B SFT V2 + PhoWhisper-Large`)

Giải pháp End-to-End chuyển đổi trực tiếp yêu cầu giọng nói tiếng Việt (kèm hội thoại nhiều lượt) thành chuỗi tool calls thực thi trên `zaic_tools.ToolEnv`. Toàn bộ quá trình suy luận chạy **100% offline**, cố định seed (`42`), giải mã tất định (`temperature=0.0`, greedy Whisper) và tuân thủ chặt chẽ mọi ràng buộc dung lượng, độ trễ và bảo mật môi trường của BTC.

## 1. Kiến trúc hệ thống & Ngân sách tham số

| Thành phần | Mô hình / Kiến trúc | Trọng số offline trong `/code/saved_models/` | Số tham số | Giới hạn BTC |
|---|---|---|---|---|
| **ASR** | Fine-tuned `WhisperForConditionalGeneration` (`kzap201/phowhisper-large-zaloai`, v2) | `saved_models/phowhisper-large-zaloai` | `1,543,304,960` (`1.54B`) | — |
| **LLM** | Pure-text fine-tuned `Qwen3_5ForCausalLM` (`Qwen/Qwen3.5-4B` LoRA SFT V2 merged, vision tower & MTP pruned) | `saved_models/qwen35_sft_v2_merged` | `4,205,751,296` (`4.21B`) | `< 5.0B`, context `<= 16,384` |
| **Tổng hệ thống** | ASR + LLM (không dùng model phụ trợ ngoài) | 2 thư mục model trên | **`5,749,056,256` (`5.75B`)** | **`< 6.0B`** |

> Lưu ý: Không đóng gói base model gốc (`Qwen/Qwen3.5-4B`, `vinai/PhoWhisper-large`) vào Docker; container chỉ chứa trọng số đã tinh chỉnh phục vụ suy luận.

## 2. Pipeline suy luận End-to-End

1. **Tiền xử lý âm thanh & ASR (`src/asr/whisper_engine.py`, `src/asr/itn.py`, `src/asr/aura_nbest.py`):**
   - Cắt khoảng lặng bảo toàn biên (`-35 dB`, margin `170 ms`), chuẩn hóa biên độ đỉnh `0.95`, đệm biên `20 ms / 60 ms`.
   - Suy luận **greedy deterministic** (`language="vi"`, `task="transcribe"`, `repetition_penalty=1.02`, không bias ngữ cảnh từ ngoài `--no-context-bias`) kèm Inverse Text Normalization tiếng Việt (`src/asr/itn.py`).
   - Với lượt yêu cầu hiện tại (`sample.audio`), giải mã thêm 3-beam diverse N-best (`nbest_beams=3`, `nbest_max_delta=0.65`) để tạo gợi ý âm vị gọn (`asr_phonetic_diff`) khi có sai lệch nhỏ ở thực thể/từ khó.
   - Các lượt `user` trong `sample.history` được nhận dạng bằng Whisper greedy (`num_beams=1`).
2. **Agent ReAct Loop & Track A Runtime (`src/agent/react_loop.py`, `src/agent/llm_client.py`):**
   - Backbone **vLLM (`0.33.0`) + XGrammar (`0.2.8`)** với `dtype=bfloat16`, `max_model_len=16384`, `max_new_tokens=384`, `temperature=0.0`, `enable_thinking=false`, `repetition_penalty=1.05`, `guided_decoding=true` (ép cấu trúc XML `<tool_call><function=...>`, tên tool và tên tham số hợp lệ theo `installed_apps` của từng mẫu).
   - Giới hạn an toàn: `max_steps=10`, `max_repairs=3`, `timeout_s=25.0`, `max_prompt_tokens=15000`, `max_result_chars=1500`.
   - 4 lớp kiểm soát Track A (`configs/e2e_qwen35_4b_sft_v2.yaml`):
     - `enable_xref_guard`: rà soát chéo `x-ref` trước khi gọi tool ghi/xóa để chống hallucinate ID.
     - `enable_arg_sanitizer`: chuẩn hóa và lọc tham số rỗng/sai kiểu nhẹ trước khi gửi vào `ToolEnv`.
     - `enable_ask_guard`: bảo đảm `sys.ask_missing_info` dùng đúng tên tool & tên tham số thiếu trong schema.
     - `enable_boundary_verifier`: kiểm tra trạng thái cuối & xác minh lỗi schema trước khi kết thúc.
3. **Cách ly môi trường (`src/env/safe_env.py` & `agent.py`):**
   - Khám phá động `<split>` trong `/data` bằng `<split>_tools_catalog.json` (không đọc `<split>_environment/`).
   - Nạp `zaic_tools` trực tiếp từ `/data/<split>/zaic_tools` qua `sys.path.insert(0, ...)`; **không** đóng gói bản sao `zaic_tools` trong `/code/`.
   - Mỗi mẫu khởi tạo đúng một phiên `SafeEnv` (`ToolEnv`), tuyệt đối không gọi `env.get_state()` hay đọc `state.json`.

## 3. Huấn luyện & Dữ liệu tổng hợp (`training_code/`)

- Chi tiết đầy đủ pipeline dữ liệu (When2Call Step-0/Post-Lookup, APIGen-MT multi-intent chaining, Recovery schema repair, Hammer/Tool-N1 Irrelevance OOS, AURA ASR robustness) và script tái lập SFT V2 nằm trong [`training_code/README.md`](training_code/README.md).
- Tập dữ liệu SFT V2 (`4,954` mẫu sạch, loại bỏ hoàn toàn `299` mẫu Dev `configs/splits/dev_v1.json`) và trọng số `exp_qwen35_sft_v2` được lưu trữ tại:
  `https://huggingface.co/htphong0909/zalo-aic-sft-checkpoints` *(public repository; repo ASR `https://huggingface.co/kzap201/phowhisper-large-zaloai` cũng public)*.

## 4. Kết quả kiểm thử đã xác nhận

| Tập đánh giá | Cấu hình | Final Score | State Match Rate | Tool Precision / Recall / F1 |
|---|---|---|---|---|
| **Dev 299 (unseen, `N=299`)** | Track A E2E Voice (`PhoWhisper 3-beam` + `Qwen3.5-4B SFT V2`) | **`0.7503`** | `81.29%` | `0.8029` / `0.8143` / `0.8039` |
| **Dev 299 (unseen, `N=299`)** | Track A Oracle Text (`vLLM 0.33 + XGrammar`) | `0.7788` | `83.45%` | `0.8234` |
| **Fold 0 (`N=250`)** | Track A E2E Voice | `0.8596` | `89.61%` | `0.8940` / `0.8972` / `0.8942` |

## 5. Hướng dẫn chạy & kiểm tra Docker offline

```bash
# 1. Tải trọng số inference vào /code/saved_models (không tải base model; xuất HF_TOKEN nếu repo LLM vẫn đặt private)
python3 prepare_models.py --saved-dir saved_models

# 2. Build image CUDA 13.2.0
docker build --network=host -t zac2026:lifemate .

# 3. Chạy suy luận offline (--network none, /data chỉ đọc) -> /result/submission.jsonl
docker run --rm --network none --gpus '"device=0"' \
  -v /path/to/data:/data:ro \
  -v /path/to/result:/result \
  zac2026:lifemate /bin/bash /code/predict.sh

# 4. Chạy JupyterLab (cổng 9777, mật khẩu/token: zac2026) để kiểm tra predict_notebook.ipynb -> /result/jupyter_submission.jsonl
docker run --rm --network none --gpus '"device=0"' -p 9777:9777 \
  -v /path/to/data:/data:ro \
  -v /path/to/result:/result \
  zac2026:lifemate /bin/bash /code/start_jupyter.sh
```
