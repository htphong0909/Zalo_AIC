# Đánh Giá Pipeline Hiện Tại & Thiết Kế Phương Án Post-Training Chuyên Biệt Cho `LiquidAI/LFM3-1.7B` (SOTA 2024–2026)

> **Dự án:** Zalo AI Challenge 2026 — Track **LifeMate (Voice Assistant Tool-calling Agent)**  
> **Mô hình Trọng tâm (Chốt chiến lược):** **`LiquidAI/LFM3-1.7B`** (`1.73B` tham số, kiến trúc Tri-Hybrid) kết hợp ASR Champion **`kzap201/phowhisper-large-zaloai`** (`1.54B`) $\Longrightarrow$ **Tổng tham số toàn hệ thống = `3.27B << 6.0B`** (siêu nhẹ, tốc độ suy luận cực nhanh, kế thừa trực tiếp bộ 3 notebook chính chủ từ tác giả LiquidAI tại [`src/lfm3-post-training/`](../../src/lfm3-post-training/)).  
> **Tài nguyên nghiên cứu đối ứng đã lưu trữ:**  
> - **14 Bài báo PDF (2024–2026):** [`references/tool_calling_papers/`](../../references/tool_calling_papers/README.md)  
> - **12 Kho Mã Nguồn Công Khai:** [`references/tool_calling_repos/`](../../references/tool_calling_repos/README.md)

---

## PHẦN I. VÌ SAO CHỌN `LiquidAI/LFM3-1.7B` & CHẨN ĐOÁN 7 NGUYÊN NHÂN KHIẾN BẢN CŨ CHỈ ĐẠT ~23%

### 1.1. Lợi Thế Chiến Lược Của `LiquidAI/LFM3-1.7B` Trong Bài Toán LifeMate

1. **Dư địa ngân sách tham số & VRAM cực lớn (`3.27B / 6.00B`):**
   - Ghép `PhoWhisper Champion` (`1.54B`, `1.50%` Dev WER) với `LiquidAI/LFM3-1.7B` (`1.73B`) chỉ tốn **`3.27B` tham số** (dư tới `2.73B` so với trần 6B, không bao giờ lo rủi ro vi phạm quy chế đếm tham số safetensors).
   - Ở độ chính xác `bfloat16`, trọng số `LFM3-1.7B` chỉ chiếm **~3.46 GB VRAM**. Kiến trúc **Tri-Hybrid** (kết hợp GQA Attention, Gated Short Convolutions `in_proj`/`out_proj` và SwiGLU MLP `w1`/`w2`/`w3`) tiêu thụ KV-cache nhỏ hơn nhiều so với Transformer thuần ở ngữ cảnh 8,000–12,000 tokens.
2. **Đòn bẩy huấn luyện RL & Full/High-Rank LoRA vượt trội trên 1 GPU 24GB:**
   - Trong khi mô hình 4B–5B bị bó hẹp ở `G=4..6` rollouts và dễ tràn bộ nhớ ở ngữ cảnh 12K, `LFM3-1.7B` cho phép nâng LoRA rank lên **`r = 32..64`** (hoặc thậm chí **Full Fine-Tuning** ở pha SFT!) và nâng kích thước nhóm lấy mẫu GRPO/GiGPO lên **`num_generations = 12..16`** ngay trên 1 GPU Colab L4/A100. Theo **ToolRL (NeurIPS 2025)** và **GiGPO (NeurIPS 2025)**, nhóm rollout lớn giúp mô hình 1.5B–1.7B khám phá trúng chuỗi gọi hàm hiếm và ổn định phương sai gradient vượt bậc.
3. **Hỗ trợ Native Tool Calling & Cookbook chính chủ từ tác giả LiquidAI:**
   - Tokenizer của LFM3 tích hợp sẵn template XML tool-calling và hàm giải mã gốc `tokenizer.parse_response(completion_ids, prefix=prompt, tools=tools)`.
   - Kho mã nguồn đã có sẵn 3 tài liệu gốc của tác giả: [`lfm3_sft_with_trl.ipynb`](../../src/lfm3-post-training/lfm3_sft_with_trl.ipynb), [`lfm3_dpo_with_trl.ipynb`](../../src/lfm3-post-training/lfm3_dpo_with_trl.ipynb) và [`lfm3_grpo_with_trl.ipynb`](../../src/lfm3-post-training/lfm3_grpo_with_trl.ipynb).

---

### 1.2. Bảy Thủ Phạm Gốc Rễ Khiến Nỗ Lực Post-Training `LFM3-1.7B` Đầu Tiên Kẹt Ở `23.42%` (SFT) $\rightarrow$ `23.13%` (GRPO)

Đối chiếu trực tiếp code hiện tại (`src/agent/lfm3_*.py`, `scripts/build_lfm3_data.py`, `configs/lfm3_*.yaml`) với 3 notebook gốc của LiquidAI và bộ chấm điểm chính thức `src/eval/metrics.py`, ta thấy rõ **mô hình LFM3 không hề yếu, mà đang bị 7 lỗi dữ liệu/hàm thưởng kìm hãm**:

| # | Thủ phạm trong Code hiện tại | Phân tích hậu quả trực tiếp trên `dev_v1` (`n=299`) |
| :---: | :--- | :--- |
| **1** | **Bị trói vào ASR cũ `ChunkFormer-CTC` (`21.89%` WER)** (`configs/e2e_chunkformer_ctc_lfm3_17b.yaml` & `simulate_ctc_transcript` tại `build_lfm3_data.py:66-106`) | `LFM3-1.7B` chưa từng được chạy hay huấn luyện với **ASR Champion `PhoWhisper` (`1.50%` WER, `86.96%` EM)**! Việc bắt LFM3 đọc văn bản không dấu câu, viết thường toàn bộ, biến mọi chữ số thành chữ viết (`"2,8 triệu" -> "hai triệu tám"`, `"sjc" -> "ét giê xê"`) gây ra hàng loạt lỗi `arg_value_mismatch` (`104/299` mẫu). |
| **2** | **SFT chỉ chạy 50 steps (~0.25 epoch), cắt cụt `4096` tokens & bỏ sót `gate_proj`** (`configs/lfm3_sft.yaml:23,30`, `lfm3_sft.py:11-21`, `llm_client.py:132`) | 50 bước x 16 mẫu = 800 bước mẫu (chưa nổi 1/3 epoch!). `max_length: 4096` cắt cụt hội thoại đa lượt + 32 tool schemas (dài 4.5K–9K tokens); `llm_client.py:132` lại cắt tràn từ **bên trái**, xóa mất System Prompt & Tools! Ngoài ra `LFM3_TARGET_MODULES` bỏ quên **`"gate_proj"`** (cổng attention đầu ra theo từng head mà tác giả LiquidAI gợi ý tại `lfm3_dpo_with_trl.ipynb` dòng 291). |
| **3** | **Bỏ qua hoàn toàn công thức DPO vàng của tác giả LFM3** (`src/lfm3-post-training/lfm3_dpo_with_trl.ipynb` dòng 308 & 330) | Trong `lfm3_dpo_with_trl.ipynb`, tác giả LiquidAI chứng minh `DPOConfig(loss_type=["sigmoid", "sft"], loss_weights=[1.0, 1.0])` trên chính lỗi tự sinh của `LFM3-1.7B` nâng độ chính xác từ chối câu hỏi bất khả thi từ **20% lên 90%**. Do bỏ qua pha này và thay bằng oversampling `2x` thô + xóa 88% `final_stop` (`build_lfm3_data.py:354-366`), lỗi hỏi nhầm `false_ask` nổ tung **gấp 9 lần** (`từ 1 lên 29 mẫu`!) còn `missed_oos` vẫn hỏng `13/21`. |
| **4** | **Bẫy Copy-Paste `lfm3_scorer.py` từ dataset mẫu sang bài toán Zalo** (`src/agent/lfm3_scorer.py:39-54, 116-180` vs `src/eval/metrics.py:36-49`) | `lfm3_scorer.py` copy hàm `canonical()` và `same_value()` mà LiquidAI viết riêng cho dataset `phone-assistant-tools`: tự động điền giá trị `default` từ schema, bỏ dấu tiếng Việt, `casefold()` cho `enum`, và `sorted()` mảng. Nhưng `src/eval/metrics.py` của BTC Zalo **không điền `default`** (truyền dư khóa mặc định làm trượt `set(p) == set(g)` $\rightarrow$ **0 điểm**!) và dùng `==` tuyệt đối cho mọi trường không có `"x-match": "fuzzy"` (kể cả thứ tự mảng `params` trong `sys.ask_missing_info`!). GRPO đã thưởng `1.0` cho các lời gọi hàm bị BTC chấm `0.0`! |
| **5** | **`lfm3_reward.py` Teacher-Forced đơn bước & mù `state_hash`** (`src/agent/lfm3_reward.py:52-97`) | `validate_step_in_env` chỉ nạp `prefix_calls` mẫu chuẩn rồi gọi `env.validate()`, bắt mọi lỗi trả về `True` (`except Exception: return True` dòng 71!), **không gọi `env.call()` cho dự đoán** và **không kiểm tra `env.get_state_hash()`**. Vì thế `LFM3-GRPO` tăng nhẹ Tool F1 (`42.42% -> 42.90%`) nhưng làm **tụt `state_match_rate` (`30.94% -> 30.58%`)** và tụt điểm chung cuộc (`23.42% -> 23.13%`)! |
| **6** | **Lệch số lượng công cụ lúc Học vs Thi & xóa mất gợi ý `x-ref`** (`build_lfm3_data.py:210-243`, `tool_selector.py:42-68`) | 80% mẫu train bị cắt còn `req_apps + 3 app nhiễu` (với mẫu `OOS` chỉ thấy đúng **3 app**!), nhưng khi thi gặp tới 18 app / 62 tools. Đồng thời `compact_tool_spec` xóa sạch trường `x-ref` và `x-match`, bắt mô hình nhỏ 1.73B phải tự đoán tham số nào là khóa ngoại cần gọi hàm `read` trước. |
| **7** | **Chia bước ReAct cứng nhắc & bẫy CLI mất `tools`** (`sft_dataset.py:39-44`, `build_lfm3_data.py:132-143, 295`) | `group_calls_into_steps` ép tách 2 lệnh `read` độc lập có tham số sang 2 lượt (phạt oan khi LFM3 gọi song song); còn nếu chạy `build_lfm3_data.py` quên truyền `--model-id`, `_fallback_render_prompt` đánh rơi toàn bộ danh sách `tools` khỏi prompt! |

---

## PHẦN II. TINH HOA 14 BÀI BÁO KHOA HỌC (2024–2026) ÁP DỤNG CHO MÔ HÌNH `1.7B`

Các công trình SOTA giai đoạn 2024–2026 (lưu tại [`references/tool_calling_papers/`](../../references/tool_calling_papers/README.md) và [`references/tool_calling_repos/`](../../references/tool_calling_repos/README.md)) đặc biệt tập trung vào phân khúc **On-Device Small Language Models (`0.5B – 3B`)** — trùng khớp hoàn hảo với quy mô **`1.73B`** của `LFM3`:

1. **Hammer (ICLR 2025 — `0.5B / 1.5B / 3B`, `03_Hammer_ICLR2025`):** Phát hiện bệnh lý đặc trưng nhất của mô hình `< 3B` là *học vẹt tên hàm/tham số* thay vì đọc mô tả JSON Schema. Kỹ thuật **Function & Parameter Masking** (`train/data_processing.py:66-135`, thay tên hàm & khóa tham số bằng chuỗi ngẫu nhiên trên một tỷ lệ mẫu SFT) cộng với **Irrelevance Augmentation** giúp `Hammer-1.5B` vượt nhiều mô hình 7B–8B.
2. **When2Call (NAACL 2025 — `1.5B – 8B`, `08_When2Call_NAACL2025`):** Giải quyết trực diện 3 kịch bản của LifeMate (*Call Tool* vs *Ask Clarification `sys.ask_missing_info`* vs *Abstain `[]`*) bằng **Preference Optimization (DPO)** đối chiếu đa lựa chọn — ghép nối tự nhiên 100% với notebook `lfm3_dpo_with_trl.ipynb` (`loss_type=["sigmoid", "sft"]`) của chính LiquidAI!
3. **ToolRL (NeurIPS 2025 — `1.5B / 3B`, `04_ToolRL_NeurIPS2025`):** Chứng minh với mô hình 1.5B, phần thưởng nhị phân thô làm hỏng quá trình học khi có nhiều tham số; cần **hàm thưởng phân rã có cấu trúc** (Format + Khớp tập tên công cụ + Khớp tập khóa tham số + Khớp giá trị tham số), giúp mô hình 1.5B/3B vượt SFT +15%.
4. **GiGPO (NeurIPS 2025 — `1.5B / 3B`, `05_GiGPO_verl-agent_NeurIPS2025`):** Giải quyết bài toán gán tín dụng (credit assignment) cho chuỗi ReAct đa bước trên mô hình 1.5B/3B **không cần mạng Critic** bằng cách gom nhóm các bước có cùng trạng thái môi trường mỏ neo (`Anchor State` $\equiv$ `SafeEnv.get_state_hash()`) để tính lợi thế vi mô bậc bước + lợi thế vĩ mô toàn quỹ đạo.
5. **APIGen-MT & xLAM (NeurIPS 2025 & NAACL 2025 — `xLAM-2-1b-fc-r` & `3b`, `06_xLAM_APIGen-MT_NeurIPS2025`):** Chứng minh một mô hình **1B–3B** (`xLAM-2-1B/3B`) hoàn toàn có thể đánh bại GPT-4o trên benchmark hội thoại đa lượt có trạng thái ($\tau$-bench, BFCL) nếu dữ liệu huấn luyện được tổng hợp từ *Reverse Task Blueprint* có kiểm chứng trạng thái thực thi thật.
6. **LoopTool (ACL 2026 — `07_LoopTool_ACL2026`) & Tool-N1 (ICLR 2026 — `01_Tool-N1_ICLR2026`):** Đóng vòng lặp Dữ liệu – Huấn luyện bằng cách lọc động các mẫu có phương sai nhóm khác không (`0 < pass@G < 1` trên tập Train) cho GRPO, đồng thời mở rộng dữ liệu tập trung vào các cụm lỗi (`pass@G == 0`) của checkpoint hiện tại.
7. **AURA (IEEE ASRU 2025 — `12_AURA_ASRU2025`), ToolSandbox (NAACL 2025 — `09_ToolSandbox_NAACL2025`), HammerBench (ACL 2025 — `10_HammerBench_ACL2025`), TinyAgent (EMNLP 2024 — `1.1B`, `11_TinyAgent_EMNLP2024`) & SimpleTool (ICML 2026 — `0.5B–4B`, `02_SimpleTool_RealtimeTool_ICML2026`):** Huấn luyện kháng nhiễu đa giả thuyết ASR, gắn cột mốc trạng thái trung gian (*State Milestones*) và thu gọn/chú thích schema tối ưu cho mô hình 1B–2B.

---

## PHẦN III. KIẾN TRÚC POST-TRAINING 4 GIAI ĐOẠN TỐI ƯU HÓA RIÊNG CHO `LiquidAI/LFM3-1.7B`

### 3.1. Sơ Đồ Kiến Trúc Tổng Thể (`PhoWhisper 1.54B + LFM3-1.7B = 3.27B << 6.0B`)

```text
[Audio .wav (Current + History User Turns)]
        │
        ▼  (0.36s/sample, FP16, VAD + Collar + Padding + Vietnamese ITN)
┌────────────────────────────────────────────────────────────────┐
│ ASR Champion: kzap201/phowhisper-large-zaloai (1.540B)         │
└────────────────────────────────────────────────────────────────┘
        │  Clean Punctuated Text + Arabic Numerals (1.50% WER)
        ▼
┌────────────────────────────────────────────────────────────────┐
│ LFM3 Schema & Prompt Formatter (Safe History Compaction)       │
│ • Inject explicit [x-ref: app.field] & [fuzzy] into schemas    │
│ • Preserve 100% System + Tools (never left-truncate prompt!)   │
└────────────────────────────────────────────────────────────────┘
        │
        ▼  (BF16, repetition_penalty=1.1, ~0.8s - 2.0s / sample)
┌────────────────────────────────────────────────────────────────┐
│ LiquidAI/LFM3-1.7B (1.730B Tri-Hybrid, Merged 10-Module LoRA)  │
│   ├─ Stage 0: Real PhoWhisper 4-Beam + 20% Hammer Masking      │
│   │           + DAG Step Unrolling + SafeEnv Blueprints        │
│   ├─ Stage 1: LFM3 Native SFT (3 epochs, 10K ctx, +gate_proj)  │
│   ├─ Stage 2: LFM3 Sigmoid+SFT DPO (When2Call 3-Way Boundary)  │
│   └─ Stage 3: Multi-Turn SafeEnv ToolRL + GiGPO + LoopTool     │
└────────────────────────────────────────────────────────────────┘
        │  Native XML <tool_call><function=...> + parse_response()
        ▼
┌────────────────────────────────────────────────────────────────┐
│ SafeEnv ReAct Loop (max_steps=8, env.validate + xref_guard)    │
│ • Output: Ordered tool_calls + Verified env.get_state_hash()   │
└────────────────────────────────────────────────────────────────┘
```

---

### 3.2. Thiết Kế Kỹ Thuật Chi Tiết Từng Giai Đoạn Cho `LFM3-1.7B`

#### 🔹 GIAI ĐOẠN 0: Chuẩn Hóa Hạ Tầng LFM3 & Tái Thiết Dữ Liệu `lfm3_data_v2` (Chỉ trên `train_only` = 1,246 mẫu)

1. **Sửa ngay 4 điểm nghẽn hạ tầng cho LFM3:**
   - Tạo cấu hình chuẩn **`configs/e2e_phowhisper_lfm3_17b.yaml`** (ghép `kzap201/phowhisper-large-zaloai` + `LiquidAI/LFM3-1.7B` với `repetition_penalty: 1.1`, cập nhật `ASRConfig` trong `src/core/config.py`) để đo ngay mốc **Zero-Shot LFM3 trên PhoWhisper thật** (thay cho mốc 14.92% trên ChunkFormer lỗi).
   - Bổ sung **`"gate_proj"`** vào `LFM3_TARGET_MODULES` trong `src/agent/lfm3_sft.py`:
     ```python
     LFM3_TARGET_MODULES = [
         "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
         "in_proj", "out_proj",
         "w1", "w2", "w3",
     ]
     ```
   - Sửa lỗi cắt cụt bên trái trong `src/agent/llm_client.py:132-133`: Nếu prompt vượt ngưỡng `max_input_len`, thu gọn `tool_result` của các lượt cũ trong `history.py` chứ tuyệt đối không bao giờ cắt mất phần đầu chứa System Prompt và danh sách Tool Schemas.
   - Nâng cấp `compact_tool_spec` (`src/agent/tool_selector.py:42-68`): Giữ lại gợi ý khóa ngoại `x-ref` và `x-match: fuzzy` ngay trong mô tả tham số (ví dụ: `"[Bắt buộc lấy ID thực tế từ kết quả tra cứu x-ref: contacts]"`). Với mô hình nhẹ `1.73B`, việc tường minh hóa ràng buộc ngay trên bề mặt schema giúp tăng vọt độ chính xác gọi chuỗi `read -> write`.

2. **Nâng cấp `scripts/build_lfm3_data.py` thành Pipeline Dữ Liệu 4 Tầng:**
   - **Tầng 1 — Real `PhoWhisper` 4-Beam Transcripts (*AURA, ASRU 2025*):** Xóa bỏ `simulate_ctc_transcript`. Chạy `PhoWhisper` trên toàn bộ 1,246 mẫu `train_only` (cả lượt nói hiện tại và các lượt `.wav` trong `history`) với `num_beams=4, num_return_sequences=4`. Kết hợp bản `audio_text` gốc với các giả thuyết `PhoWhisper` có khác biệt nhỏ (sai dấu phương ngữ, tên riêng danh bạ, từ mượn) để `LFM3-1.7B` miễn nhiễm trước nhiễu thực tế của chính `PhoWhisper`.
   - **Tầng 2 — DAG Step Unrolling, 100% `final_stop` & Full Installed Apps:**
     - Bắt buộc truyền `tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM3-1.7B", trust_remote_code=True)` (cấm chạy fallback đánh rơi `tools`).
     - Sửa `group_calls_into_steps` (`src/agent/sft_dataset.py`): Gom các lệnh `read` độc lập vào cùng 1 lượt, chỉ tách sang lượt sau khi có phụ thuộc dữ liệu `read -> write` (`x-ref`).
     - **Giữ 100% bước dừng `final_stop`** (xóa bỏ dòng lọc `rng.random() > 0.12`!) để LFM3 học dứt khoát điểm dừng sau khi nhận đủ `tool_result`.
     - **Giữ nguyên 100% `installed_apps` gốc** (không cắt còn 3 distractors!). Đặc biệt với mẫu `OOS`, luôn đảm bảo có **10–18 distractor apps** gần nghĩa để LFM3 học từ chối vững chắc trong môi trường nhiễu cao.
     - Chuẩn hóa thứ tự mảng `params` trong `sys.ask_missing_info` theo đúng thứ tự khai báo của `<split>_tools_catalog.json` và kiểm tra tự động mọi mẫu qua `src/eval/metrics.py` đạt **`score == 1.0`**.
   - **Tầng 3 — `Hammer` Function & Parameter Masking (`+25%` dữ liệu SFT, từ `03_Hammer_ICLR2025/train/data_processing.py`):**
     - Với mô hình nhỏ `1.73B`, áp dụng hàm `replace_function_names_new` & `replace_param_names_new` trên 25% bản sao quỹ đạo `act` (đổi tên hàm như `reminders.create_reminder` $\rightarrow$ `app_k4.fn_m92` và khóa tham số thành mã ngẫu nhiên đồng bộ trong cả schema `tools`, lịch sử và lời gọi đích, giữ nguyên `description`). Điều này ép các tầng GQA + LIV Conv của `LFM3-1.7B` phải đối chiếu ngữ nghĩa câu nói với `description` của từng tham số, chống học vẹt triệt để và chống bịa đặt khóa optional không tồn tại.
   - **Tầng 4 — `APIGen-MT` Executable Blueprints (`+1,500` mẫu kiểm chứng qua `SafeEnv`):**
     - Từ các sandbox `SafeEnv("train", sid)` của `train_only` (chỉ truy vấn bằng các tool `read` hợp lệ!), sinh thêm các kịch bản đa bước có `gold_state_hash` xác thực, các kịch bản cố tình ẩn 1 tham số bắt buộc (`ask_missing_info`), và các kịch bản yêu cầu tính năng của app **chưa cài đặt** trên máy người dùng (`OOS -> []`).

---

#### 🔹 GIAI ĐOẠN 1: `LFM3` Native LoRA SFT Đầy Đủ (`src/agent/lfm3_sft.py` nâng cấp)

Kế thừa trực tiếp `src/lfm3-post-training/lfm3_sft_with_trl.ipynb` với các thông số đã sửa hết lỗi non bước & cắt cụt:
- **LoRA Config (10 modules Tri-Hybrid):** `r = 32`, `lora_alpha = 64`, `lora_dropout = 0.05`, `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "in_proj", "out_proj", "w1", "w2", "w3"]`. *(Lưu ý: Nếu tác giả LFM3 khuyến nghị Full Fine-Tuning cho 1.7B ở pha SFT, trên A100/L4 24GB với `bfloat16` + `AdamW 8-bit` + `gradient_checkpointing` hoàn toàn chạy mượt Full SFT vì model chỉ 1.73B!)*
- **SFT Config:**
  - `num_train_epochs = 3.0`, **`max_steps = -1`** (bỏ hẳn giới hạn `50` bước cũ!).
  - **`max_length = 10240`** (hoặc `12288`), bao phủ trọn vẹn 100% prompt có `history` 12 lượt + 62 tool schemas mà không cắt mất bất kỳ ký tự nào.
  - `per_device_train_batch_size = 2`, `gradient_accumulation_steps = 8` (`effective_batch_size = 16`), `learning_rate = 2.0e-4`, `lr_scheduler_type = "cosine"`, `warmup_steps = 0.05`.
  - Định dạng `prompt` + `completion` pre-rendered bằng `tokenizer.apply_chat_template(..., tools=tools)` để `SFTTrainer` tính **Completion-Only Loss** chính xác 100% trên phần trả lời của Assistant.
- Lưu và gọi `trainer.model.merge_and_unload()` tạo checkpoint chuẩn **`lfm3_17b_sft_v2/merged`**.

---

#### 🔹 GIAI ĐOẠN 2: `LFM3` Grounded Decision DPO (`lfm3_dpo_with_trl.ipynb` + `When2Call`, NAACL 2025)

Đây chính là **mảnh ghép quan trọng nhất bị bỏ quên trước đó**, kết hợp trực tiếp công thức gốc của tác giả LiquidAI trong [`lfm3_dpo_with_trl.ipynb`](../../src/lfm3-post-training/lfm3_dpo_with_trl.ipynb) (dòng 324–341) với thiết kế biên quyết định 3 nhánh của **When2Call (`08_When2Call_NAACL2025/synthetic_data_gen/convert_raw_train_data_to_pref.py`)**:

1. **Thu thập lỗi tự thân (*On-Policy Mistakes*) từ chính `lfm3_17b_sft_v2/merged` trên tập `train_only`:**
   - Đúng theo nguyên lý của tác giả LiquidAI (*"the rejected answers are the model's own mistakes, sampled from LFM3-1.7B itself"*): cho `lfm3_17b_sft_v2/merged` sinh 6 mẫu (`temperature = 0.8`) + 1 mẫu greedy trên mỗi bước của `train_only`, chấm điểm bằng chính `src/eval/metrics.py` + `SafeEnv`:
     - **Nhánh 1 (`ask` — sửa lỗi `missed_ask`):** `chosen` = `<tool_call><function=sys.ask_missing_info>...</function></tool_call>` đúng thứ tự `params`; `rejected` = lỗi thực tế của LFM3 (tự bịa tham số mặc định để gọi hàm `write`, gọi `ask_missing_info` sai/thừa tên tham số, hoặc hỏi bằng lời văn bản mà không gọi tool `sys.ask_missing_info`).
     - **Nhánh 2 (`oos` — sửa lỗi `missed_oos`):** `chosen` = câu từ chối ngắn gọn không có `<tool_call>`; `rejected` = lỗi thực tế của LFM3 (cố tình gọi `notes.create_note`, `reminders.create_reminder` hoặc `sys.ask_missing_info`).
     - **Nhánh 3 (`act` & `final_stop` — sửa lỗi `false_ask`, `arg_key_mismatch` & lặp vô tận):** `chosen` = lệnh gọi đúng chỉ chứa các tham số được nhắc tới (hoặc dừng `final_stop` khi đã đủ trạng thái); `rejected` = hỏi ngược `sys.ask_missing_info` vô cớ (`false_ask`), truyền thừa tham số optional mặc định (làm trượt `set(p) == set(g)`!), đoán mò ID `x-ref` khi chưa gọi hàm `read`, hoặc tiếp tục gọi thêm hàm sau khi trạng thái đã hoàn tất.
2. **Huấn luyện bằng module mới `src/agent/lfm3_dpo.py` (kế thừa nguyên bản `lfm3_dpo_with_trl.ipynb`):**
   ```python
   dpo_config = DPOConfig(
       output_dir="./experiments/exp_lfm3_dpo_v2",
       num_train_epochs=1,
       per_device_train_batch_size=2,
       gradient_accumulation_steps=8,     # effective batch size = 16 pairs
       learning_rate=2e-5,
       loss_type=["sigmoid", "sft"],      # Công thức gốc của tác giả LiquidAI!
       loss_weights=[1.0, 0.5],           # Đẩy xa lỗi rejected + giữ chặt cú pháp XML chosen
       lr_scheduler_type="cosine",
       warmup_steps=0.1,
       beta=0.1,
       max_length=10240,
       bf16=use_bf16,
       fp16=not use_bf16,
   )
   ```
   - Nhờ `peft_config` (`r=16, alpha=32` trên 10 modules), `DPOTrainer` tự dùng `disable_adapter()` lấy `lfm3_17b_sft_v2/merged` làm Reference Model mà **không tốn thêm 1 byte VRAM nào** cho bản sao thứ hai! Kết thúc pha 2 gọi `merge_and_unload()` lưu thành **`lfm3_17b_dpo_v2/merged`**.

---

#### 🔹 GIAI ĐOẠN 3: Verifiable `SafeEnv` RL với Hàm Thưởng Chính Xác BTC + `ToolRL` + `GiGPO` + `LoopTool`

Khởi tạo từ **`lfm3_17b_dpo_v2/merged`** (neo KL `beta = 0.04` giữ vững biên quyết định `Ask`/`OOS` vừa học ở Pha 2) và sửa triệt để toàn bộ sai lệch của `lfm3_scorer.py` & `lfm3_reward.py`:

1. **Thay thế hoàn toàn `lfm3_scorer.py` bằng chuẩn `src/eval/metrics.py` + `ToolRL` (`04_ToolRL_NeurIPS2025/verl/utils/reward_score/rlla.py`):**
   - Loại bỏ `canonical()` (không tự điền `default`!), loại bỏ `norm()` bỏ dấu tiếng Việt trên các trường exact!
   - Một lệnh dự đoán khớp lệnh chuẩn **chỉ và chỉ khi** `src/eval/metrics.py:calls_match(pred, gold, catalog)` trả về `True` (`_clean` bỏ giá trị rỗng `(None, "", [], {})`, `set(p) == set(g)`, `token_f1 > 0.6` cho `"x-match": "fuzzy"`, và `p[k] == g[k]` cho mọi trường còn lại).
   - Phần thưởng dẫn hướng liên tục (Dense Shaping kiểu **ToolRL**) cho trường hợp chưa khớp hoàn toàn (trần tối đa `0.55` để luôn thấp hơn rõ rệt mức khớp hoàn toàn `1.0`):
     $$R_{\text{dense}} = 0.15 \cdot \text{F1}_{\text{tool\_name}} + 0.15 \cdot \text{Jaccard}_{\text{arg\_keys}} + 0.20 \cdot \text{MatchRate}_{\text{exact\_or\_fuzzy\_vals}} + 0.05 \cdot \mathbb{I}[\text{env\_valid}]$$

2. **Nâng cấp `lfm3_reward.py` sang Kiểm Chứng Thực Thi Thật & Cổng Cứng `state_hash` (`SafeEnv`):**
   - Xóa bỏ `except Exception: return True` tại `lfm3_reward.py:71`!
   - Thực thi thật từng lệnh trong `pred_calls`:
     - Nếu bất kỳ lệnh nào trượt `env.validate()` $\rightarrow$ phạt ngay **`reward = -0.10`** (hoặc `0.0`), tuyệt đối không thưởng phần trăm giả tạo.
     - Nếu hợp lệ, gọi thật `env.call()` trên bản sao `SafeEnv("train", sample_id, root)`! Đối với bước hoàn tất / bước có thao tác `write` cuối quỹ đạo, kiểm tra trực tiếp **`env.get_state_hash() == gold_state_hash`**: nếu làm sai lệch trạng thái môi trường so với nhãn chuẩn, khóa cứng trần phần thưởng xuống `<= 0.20` (không bao giờ nhận được `1.0`!).

3. **Mở rộng sang Multi-Turn On-Policy `GiGPO` Rollout (`05_GiGPO_verl-agent_NeurIPS2025/gigpo/core_gigpo.py`) & Lọc Mẫu Khó `LoopTool` (`07_LoopTool_ACL2026`):**
   - **Pha 3A (Fast Step-wise GRPO với `LoopTool` Filtering trên TRL `GRPOTrainer`):**
     - Dùng kỹ thuật lọc của **LoopTool** (`filter_grpo_sample.py`): chạy đánh giá nhanh `lfm3_17b_dpo_v2/merged` với $G=12$ generations trên `train_only`, loại bỏ các bước dễ mà cả 12/12 lần đều đạt `1.0` (`std == 0`), chỉ giữ lại các bước có sự phân hóa điểm thưởng (`0 < pass@12 < 1`) để huấn luyện `GRPOTrainer` với **`num_generations = 12`** (tận dụng ưu thế VRAM siêu nhẹ của `LFM3-1.7B`!).
   - **Pha 3B (End-to-End Multi-Turn `SafeEnv` Rollout với `GiGPO` Anchor-State Advantage):**
     - Cho `LFM3-1.7B` tự chạy trọn vẹn vòng lặp ReAct từ bước 1 đến bước kết thúc ($G=8$ quỹ đạo hoàn chỉnh / mẫu `train_only`, nhận `tool_result` thật từ `SafeEnv.call()` ở mỗi bước chứ không dùng gold prefix!).
     - Chấm điểm cuối quỹ đạo bằng chính hàm **`score_sample()` của BTC (`src/eval/metrics.py:82-107`)**, và áp dụng trực tiếp hàm `compute_gigpo_outcome_advantage` (`core_gigpo.py:138-172`) với khóa mỏ neo trạng thái tại đầu mỗi bước $t$:
       $$\text{AnchorState}_{i,t} = \big(\text{sample\_id},\; \text{env}_{i,t}.\text{get\_state\_hash}(),\; \text{frozenset}(\text{read\_calls}_{i,<t})\big)$$
     - Nhờ đó, `LFM3-1.7B` học được cách **tự phục hồi (self-correction)** ngay trong chuỗi đa bước nếu bước 1 gọi `read` hơi rộng hoặc nhận kết quả danh bạ trùng tên, đồng thời tối ưu trực tiếp chỉ số chấm thi cuối cùng (**Tool F1 có điều kiện `state_hash` khớp 100%**)!

---

## PHẦN IV. DANH SÁCH CÂU HỎI / HỖ TRỢ KỸ THUẬT NÊN KHAI THÁC TỪ TÁC GIẢ `LFM3`

Vì đội ngũ của bạn có kênh trao đổi hỗ trợ từ phía tác giả **LFM3**, dưới đây là **4 điểm kỹ thuật giá trị cao nhất** bạn có thể hỏi/nhờ tác giả tư vấn hoặc cung cấp snippet để tối đa hóa lợi thế cạnh tranh:

1. **Tăng tốc Generation Rollout cho GRPO/vLLM trên kiến trúc Tri-Hybrid LFM3:**
   - Trong `lfm3_grpo_with_trl.ipynb` (dòng 555), tác giả ghi chú: *"TRL can also generate with `use_vllm=True`, which is much faster when your vLLM version supports the model."* Hãy hỏi tác giả phiên bản `vllm` / wheel nội bộ hoặc cấu hình `torch.compile` / CUDA kernel cho lớp LIV short-conv (`in_proj`/`out_proj`) để tăng tốc độ sinh rollout RL gấp **5–8×** và giảm độ trễ thi đấu xuống `< 1s/mẫu`.
2. **Chiến lược tinh chỉnh tối ưu cho các lớp LIV Convolution vs Attention (`LoRA` + `gate_proj` vs `Full Fine-Tuning`):**
   - Hỏi tác giả với tập dữ liệu ~4,000–8,000 bước tool-calling tiếng Việt có schema phức tạp (~6,000–10,000 tokens), `LFM3-1.7B` đạt độ chính xác cao hơn khi chạy **LoRA (`r=32/64` có bật `gate_proj`)** hay **Full Fine-Tuning với lr phân tầng** (ví dụ lr nhỏ hơn cho LIV conv `in_proj`/`out_proj` để bảo toàn khả năng giữ ngữ cảnh dài)?
3. **Khai thác chế độ `enable_thinking=True` ngắn gọn với `tokenizer.parse_response`:**
   - Cả 3 notebook mẫu đang đặt `enable_thinking=False`. Hãy xác nhận với tác giả cấu trúc token đặc biệt chuẩn của `LFM3-1.7B` khi bật `enable_thinking=True` đi kèm `<tool_call>` để `tokenizer.parse_response()` vẫn bóc tách chính xác 100% khi mô hình sinh 1–2 câu suy luận ngắn kiểm tra ràng buộc `x-ref` và tham số thiếu trước khi gọi hàm!
4. **Điểm ngọt Hyperparameters khi nối chuỗi 3 pha `SFT -> DPO (sigmoid+sft) -> GRPO` trên LFM3:**
   - Hỏi tác giả khuyến nghị hệ số `loss_weights` trong `DPOConfig(loss_type=["sigmoid", "sft"])` và hệ số `beta` KL trong `GRPOConfig` khi khởi tạo từ mô hình đã qua SFT+DPO (thay vì khởi tạo từ base model như trong notebook đơn lẻ) để tránh hiện tượng over-sharpening phân phối xác suất.

---

## PHẦN V. LỘ TRÌNH TRIỂN KHAI THỰC NGHIỆM TRÊN `LFM3-1.7B` (TUÂN THỦ `AGENT_LOOP_PROTOCOL.md`)

Toàn bộ quá trình tuân thủ tuyệt đối **Zero-Compute trên máy Local** (chỉ sửa code và chạy `pytest tests/ -q` trên local; toàn bộ tạo transcript PhoWhisper, SFT, DPO, GRPO/GiGPO và đánh giá đều chạy trên **Colab GPU / Kaggle GPU** và đồng bộ qua **Hugging Face Hub**; tuyệt đối không chạm vào `dev_v1` ngoài 4 chỉ số vĩ mô từ Referee):

| Mốc | Hạng mục Triển khai Cụ thể (Chuyên biệt `LiquidAI/LFM3-1.7B`) | Kỳ vọng trên Sealed `dev_v1` |
|---|---|---|
| **Mốc 0 (Mở khóa ASR Champion cho LFM3)** | 1. Tạo `configs/e2e_phowhisper_lfm3_17b.yaml` (ghép `PhoWhisper 1.50% WER` + `LFM3-1.7B`, `repetition_penalty=1.1`), sửa `ASRConfig`, bỏ cắt trái ở `llm_client.py`, thêm gợi ý `[x-ref]`/`[fuzzy]` vào `tool_selector.py`.<br>2. Đo ngay mốc Zero-Shot thật của `PhoWhisper + LFM3-1.7B` trên Colab GPU. | Vượt xa mốc cũ `14.92%` (vốn bị hỏng bởi ChunkFormer 21.89% WER), xác lập đường cơ sở sạch (~32%–42%). |
| **Mốc 1 (`lfm3_data_v2` + SFT 3 Epochs Đầy Đủ)** | 1. Chạy `PhoWhisper` 4-beam trên 1,246 mẫu `train_only` (kể cả audio `history`).<br>2. Nâng cấp `scripts/build_lfm3_data.py` (DAG grouping, 100% `final_stop`, full distractors, 25% `Hammer` masking, `APIGen-MT` blueprints).<br>3. Cập nhật `lfm3_sft.py` (+`gate_proj`, `r=32, alpha=64`, `max_length=10240`, `3 epochs` đầy đủ, xóa `max_steps: 50`). | **Final Score $\rightarrow$ 55% – 65%**, triệt tiêu cắt cụt ngữ cảnh và giảm mạnh lỗi sai tên/khóa tham số. |
| **Mốc 2 (`LFM3` Sigmoid+SFT DPO + `When2Call`)** | 1. Viết `src/agent/lfm3_dpo.py` & `scripts/train_lfm3_dpo.py` theo chuẩn `lfm3_dpo_with_trl.ipynb` (`loss_type=["sigmoid", "sft"]`).<br>2. Lấy mẫu lỗi tự thân của `lfm3_17b_sft_v2` trên `train_only` tạo ~3,500 cặp đối chiếu 3 nhánh `Act / Ask / OOS / Stop`. | Đưa độ chính xác `ask` lên **> 65%**, `oos` lên **> 80%**, kéo `false_ask` từ 29 mẫu về **< 3 mẫu** (`Final Score -> 68% – 74%`). |
| **Mốc 3 (Verifiable `SafeEnv` ToolRL + `GiGPO` + `LoopTool`)** | 1. Thay thế `lfm3_scorer.py` bằng chuẩn chính xác `src/eval/metrics.py` + `ToolRL` dense shaping; nâng cấp `lfm3_reward.py` thực thi thật `SafeEnv.call()` và kiểm tra cứng `env.get_state_hash()`.<br>2. Chạy Pha 3A (`G=12` với lọc khó `LoopTool`) và Pha 3B (Multi-turn rollout với lợi thế mỏ neo trạng thái `GiGPO`). | **Chinh phục DoD:** **Final Score / Tool F1 > 80.0%**, **State Match $\ge$ 75.0%**, **Latency ~1.5s–2.5s/mẫu**, **Total Params = 3.27B < 6.0B**. |
