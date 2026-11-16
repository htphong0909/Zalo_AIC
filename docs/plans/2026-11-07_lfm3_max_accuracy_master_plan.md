# KẾ HOẠCH TỔNG THỂ TỐI ĐA HÓA ĐỘ CHÍNH XÁC (`APEX-LFM3` MASTER PLAN)
## Chỉnh sửa Toàn diện Pipeline, Tổng hợp Dữ liệu Đa tác tử & Post-Training Đa tầng cho `LiquidAI/LFM3-1.7B`

> **Ngày lập:** 2026-11-07  
> ** Mục tiêu tối thượng:** Đạt **tỉ lệ chính xác chính thức cao nhất có thể (`final_score` = `state_match`-gated `Tool-Call F1` hướng mốc `88% – 92%+`)** cho bài toán **Zalo AI Challenge 2026 — LifeMate**, tập trung độc quyền LLM suy luận vào **`LiquidAI/LFM3-1.7B` (`1.73B` Tri-Hybrid)**, không giới hạn độ phức tạp thuật toán hay tài nguyên GPU offline trong giai đoạn phát triển/huấn luyện, đồng thời **tuân thủ tuyệt đối 100% luật thi** để không bao giờ bị loại (disqualified).

---

## Mục lục

1. [Bản chất Hàm mục tiêu Chính thức, Thống kê Thực nghiệm Tập dữ liệu & Ranh giới Luật thi](#1-bản-chất-hàm-mục-tiêu-chính-thức-thống-kê-thực-nghiệm-tập-dữ-liệu--ranh-giới-luật-thi)
2. [Kiểm toán Toàn diện 15 Lỗi Tồn đọng & Nút thắt Độ chính xác trong Codebase hiện tại](#2-kiểm-toán-toàn-diện-15-lỗi-tồn-đọng--nút-thắt-độ-chính-xác-trong-codebase-hiện-tại)
3. [Đúc kết Tinh hoa Thuật toán từ 14 Bài báo (2024–2026), 12 Public Repos & 3 Official LiquidAI Notebooks](#3-đúc-kết-tinh-hoa-thuật-toán-từ-14-bài-báo-20242026-12-public-repos--3-official-liquidai-notebooks)
4. [Kiến trúc Đích `Apex-LFM3`: Nhà máy Dữ liệu Offline & Hệ thống Suy luận Đích](#4-kiến-trúc-đích-apex-lfm3-nhà-máy-dữ-liệu-offline--hệ-thống-suy-luận-đích)
5. [Kế hoạch Thực thi Chi tiết 8 Pha (Implementation-Ready Specs, Code, Equations & Tests)](#5-kế-hoạch-thực-thi-chi-tiết-8-pha-implementation-ready-specs-code-equations--tests)
6. [Bảng Dự phóng Độ chính xác theo Từng Pha & Quản trị Rủi ro](#6-bảng-dự-phóng-độ-chính-xác-theo-từng-pha--quản-trị-rủi-ro)

---

## 1. Bản chất Hàm mục tiêu Chính thức, Thống kê Thực nghiệm Tập dữ liệu & Ranh giới Luật thi

### 1.1. Giải phẫu Toán học của Điểm số Chính thức (`src/eval/metrics.py`)

Mọi quyết định thiết kế pipeline và hàm thưởng RL phải xuất phát trực tiếp từ hàm chấm điểm chính thức `score_sample` tại `src/eval/metrics.py:82-107`:

$$\text{Score}(y_{\text{pred}}, y_{\text{gold}}) = \begin{cases} \mathbb{I}[y_{\text{pred}} = \emptyset] \cdot \mathbb{I}[h_{\text{pred}} = h_{\text{gold}}], & \text{nếu } y_{\text{gold}} = \emptyset \text{ (OOS)} \\ \text{F1}_{\text{bipartite}}(y_{\text{pred}}, y_{\text{gold}}) \cdot \mathbb{I}[h_{\text{pred}} = h_{\text{gold}}], & \text{nếu } y_{\text{gold}} \neq \emptyset \text{ (Act / Ask)} \end{cases}$$

Bốn tính chất sống còn rút ra trực tiếp từ `src/eval/metrics.py:32-99`:
1. **Vách đá Tử thần `state_hash` (`metrics.py:89, 99`):** Chỉ cần **1 lệnh `write` sai tham số**, **1 lệnh `write` thừa**, hoặc **thiếu 1 lệnh `write`** làm `env.get_state_hash()` lệch 1 bit so với `gold_hash` $\implies$ toàn bộ mẫu nhận **`0.00` điểm** dù `Tool-Call F1 = 0.95`.
2. ** Khớp tập khóa tuyệt đối `set(p) == set(g)` (`metrics.py:41`):** Sau khi lọc bỏ các giá trị rỗng `(None, "", [], {})`, tập tên tham số dự đoán `set(p)` phải **trùng khít 100%** với `set(g)`. Nếu LLM tự ý điền thêm bất kỳ tham số tùy chọn (`optional`) nào mà người dùng không nói (ví dụ `note`, `priority`, `mode`, `location`, `repeat`), hoặc sanitizer xóa nhầm 1 tham số hợp lệ $\implies$ `calls_match = False` ngay lập tức (và nếu là lệnh `write`, làm hỏng luôn `state_hash` $\to$ `0.00`).
3. **Phạt Precision trên MỌI lệnh thực thi trong `SafeEnv` (`metrics.py:93-95`):** `pred_calls = env.get_tool_calls()` ghi nhận **tất cả các lệnh đã gọi vào `SafeEnv`**, bao gồm cả các lệnh `read` dò tìm bị rỗng (ví dụ `contacts.search(query="Hương dùm")` trả về `[]` trước khi gọi lại `contacts.search(query="Hương")`). Mỗi lệnh `read` thừa làm giảm trực tiếp `precision = matched / len(pred_calls)`!
4. **Khớp `sys.ask_missing_info` (`metrics.py:47-51`):** `tool_name` phải khớp chính xác và `params` được chấm như **list có thứ tự** (`p["params"] == g["params"]`) theo thứ tự khai báo trong `catalog`.

### 1.2. Thống kê Thực nghiệm trên `1,246` Mẫu `train_only` (Tuyệt đối không chạm `dev_v1`)

Kiểm tra lập trình trên `train_only` (`1,246` mẫu sau khi loại trừ mù `configs/splits/dev_v1.json`) và `data/train/train_tools_catalog.json` cho thấy bức tranh thực tế:

| Chỉ số | Giá trị thực tế | Hệ quả Kiến trúc bắt buộc |
|---|---|---|
| Quy mô Catalog | **`46` ứng dụng, `147` tools** | Không thể nhớ thuộc lòng bằng heuristic; cần Schema Grounding + `Hammer` Masking. |
| Phân bố kịch bản (`1,246` mẫu) | `Act = 982` (`78.8%`), `Ask = 174` (`14.0%`), `OOS = 90` (`7.2%`) | `Ask + OOS` chiếm **`21.2%`** tổng điểm — ranh giới từ chối/hỏi lại quyết định thắng bại. |
| Đa lượt hội thoại (`history`) | **`821 / 1,246` (`65.9%`)** có `history` | Đồng tham chiếu (`"chị ấy"`, `"cửa đó"`, `"số vừa nãy"`, lật ý lượt 2) là trạng thái phổ biến nhất. |
| Tổng lượng Audio trong `train_only` | **`4,042` file `.wav`** (`1,246` câu chính + `2,796` câu user trong `history`) | Kho báu **`4,042` mẫu âm thanh in-domain** để fine-tune Acoustic LoRA cho `PhoWhisper-large` (gấp `3.2x` nếu chỉ lấy câu chính!). |
| Đa ứng dụng (`multi_app`) | **`582 / 1,246` (`46.7%`)** gọi từ `2+` app khác nhau | Chuỗi phụ thuộc chéo (`weather -> calendar`, `tasks -> music`, `banking -> chat_ott`) đòi hỏi `<think>` lập kế hoạch + RL đa bước (`GiGPO`). |
| Độ sâu chuỗi tool call | `1 call: 297`, `2: 346`, `3: 289`, `4: 131`, `5: 84`, `6: 9` (tổng `2,854` gold calls) | Hơn `41%` mẫu cần $\ge 3$ lệnh; không dùng KV-cache/Prefix-cache sẽ chạm trần `30s`. |
| **Đuôi dài khan hiếm dữ liệu (Long-Tail Apps)** | **25/46 ứng dụng có `< 35` mẫu** (`fuel_price: 9`, `tv: 15`, `lottery: 17`, `calculator: 18`, `shopping_list: 19`, `exchange_rate: 20`, `podcast_radio: 20`, `library: 20`, `home_energy: 22`, `fitness: 22`, `news: 22`, `household_chores: 23`, `bike_rental: 23`, `medication: 23`...) | Mô hình `1.7B` chắc chắn underfit 25 app hiếm này nếu chỉ train trên `1,246` mẫu gốc $\implies$ **Bắt buộc tổng hợp thêm `+3,500` quỹ đạo `APIGen-MT` có kiểm chứng `SafeEnv`** tập trung vào nhóm đuôi dài! |
| Top bẫy `sys.ask_missing_info` | `phone.call` (`32`), `sms.send` (`22`), `ewallet.transfer` (`18`), `video_meeting.create` (`6`), `reminders.create` (`5`), `chat_ott.send_message` (`5`), `appliances.stop` (`5`), `fitness.set_goal` (`5`) | Trung bình 65% lỗi `missed_ask` đến từ trùng tên danh bạ (`2+ Nguyễn Văn Hùng`) hoặc thiếu số tiền/giờ/người nhận. |

### 1.3. Khai thác Tối đa Quyền Tự do Offline vs. Tuân thủ Tuyệt đối Luật Thi Suy luận

Để đạt **độ chính xác cao nhất mà không rủi ro phạm quy**, kế hoạch tách bạch sắc lẹm hai môi trường:

| Chiều | Giai đoạn Offline (Data Synthesis, Distillation, RL Training) | Giai đoạn Submission / Inference (Docker Offline trên A30) |
|---|---|---|
| **Mô hình & Tài nguyên** | **Không giới hạn:** Được phép dùng Teacher LLM mạnh để chưng cất `<think>` (`Tool-N1`), sinh kịch bản mới (`APIGen-MT`), chấm thẩm định quỹ đạo (`ActionStudio Critic`), chạy multi-GPU Full-FT / DoRA / `GiGPO` / `LoopTool` 3 vòng. | **Tuân thủ tuyệt đối `< 6B` tổng & `< 5B` LLM:** `PhoWhisper-large` (`1.54B`) + `LiquidAI/LFM3-1.7B` (`1.73B`) + `Action/Boundary Cross-Encoder Verifier` (`0.28B–0.56B`) = **`3.55B – 3.83B << 6.00B`**. Không dùng `Qwen3.5`. |
| **Tương tác `SafeEnv` & Dữ liệu** | Chỉ chạy thông qua `SafeEnv` (`env.validate`, `env.call`, `env.get_state_hash`) trên **100% mẫu thuộc `train_only` (`1,246` sandboxes)**. **CẤM TUYỆT ĐỐI** đọc trực tiếp `*_environment/state.json` và **CẤM TUYỆT ĐỐI** đụng vào `dev_v1.json` (`299` mẫu mù). | Offline 100% (không internet, không API ngoài), `temperature = 0.0` / greedy deterministic, `max_seq_len <= 16,384`, thời gian trung bình **`~2.3s/mẫu` (`<= 30s/mẫu`)** trên 1x NVIDIA A30 24GB. |

---

## 2. Kiểm toán Toàn diện 15 Lỗi Tồn đọng & Nút thắt Độ chính xác trong Codebase Hiện tại

Qua đối chiếu từng dòng code hiện tại với trace thực nghiệm trên Kaggle GPU (`experiments/exp_20261107_lfm3_smoke/kaggle_output/`) và chạy kiểm thử lập trình trực tiếp trên toàn bộ `1,246` mẫu `train_only`, dưới đây là **15 lỗi nghiêm trọng đang kìm hãm trực tiếp điểm số**:

### Nhóm A — Các lỗi Phá hủy Dữ liệu & Huấn luyện (`build_lfm3_data.py`, `lfm3_sft.py`, `lfm3_dpo.py`, `lfm3_grpo.py`, `lfm3_reward.py`)

1. **[CRITICAL] Nhân đôi Special Tokens `<|startoftext|>` & `<|im_end|>` làm hỏng phân phối BOS/EOS và DPO Log-Ratio (`scripts/build_lfm3_data.py:163-179`, `src/agent/lfm3_sft.py`, `src/agent/lfm3_dpo.py`):**
   - **Hiện trạng:** `render_step_prompt_completion` gọi `tokenizer.apply_chat_template(..., tokenize=False)` tạo ra chuỗi `prompt` bắt đầu bằng `"<|startoftext|><|im_start|>system..."` và chuỗi `completion` kết thúc bằng `"<|im_end|>\n"`. Khi đưa dạng chuỗi thô `{"prompt": str, "completion": str}` vào `SFTTrainer` / `DPOTrainer` / `GRPOTrainer`, TRL gọi lại `tokenizer(prompt)` (mặc định `add_special_tokens=True`) $\implies$ **chèn thêm thẻ `<|startoftext|>` thứ hai ở đầu**, đồng thời nối thêm `eos_token_id` (`<|im_end|>`) sau `"<|im_end|>\n"` $\implies$ **nhân đôi EOS**!
   - **Hậu quả:** Làm méo xác suất dừng lượt (`<|im_end|>`), gây hiện tượng lặp vô tận hoặc cắt cụt XML `<tool_call>`, và làm sai lệch tổng log-probability của chuỗi `chosen`/`rejected` trong DPO.
   - **Cách triệt tiêu:** Chuyển sang định dạng hội thoại có cấu trúc chuẩn của tác giả LiquidAI (`lfm3_sft_with_trl.ipynb:197-203` & `lfm3_dpo_with_trl.ipynb:188-194`) hoặc tiền mã hóa (`input_ids`, `labels` với `labels = -100` trên toàn bộ `prompt`) kèm unit test kiểm tra cấp độ token ID: đúng 1 `bos_token_id` ở vị trí `0`, đúng 1 `eos_token_id` ở cuối lượt assistant, không bao giờ có `[bos, bos]` hay `[eos, newline, eos]`.

2. **[CRITICAL] Mất hoàn toàn cổng kiểm tra `state_hash` trên 100% bước `write` trong GRPO (`src/agent/sft_dataset.py:203-204` & `src/agent/lfm3_reward.py:170`):**
   - **Hiện trạng:** Trong `unroll_trajectory_steps`, cờ `is_terminal` chỉ bằng `True` ở bước văn bản kết thúc cuối cùng (`final_stop`, nơi `step_calls == []`), còn tại bước sinh lệnh `write` liền trước đó thì `is_terminal == False`! Do đó trong `lfm3_reward.py:170`, `check_hash = str(g_hash) if (is_term and g_hash) else None` luôn bằng **`None` trên mọi bước `write`**!
   - **Hậu quả:** GRPO thưởng điểm dương cho các lệnh `write` gần giống bề mặt (ví dụ thừa tham số optional làm đổi trạng thái ngầm) mà không hề biết lệnh đó làm nổ `state_hash`!
   - **Cách triệt tiêu:** Tính và lưu **`step_gold_state_hash`** sau khi thực thi `prefix_calls + step_calls` trong `SafeEnv` cho **mọi bước** trong `unroll_trajectory_steps`, bật kiểm chứng `pred_step_hash == step_gold_state_hash` ở Stage 3A, và nâng cấp Stage 3B lên **Online Multi-Turn `SafeEnv` Rollout (`GiGPO`)** chấm trực tiếp bằng `score_sample` cuối phiên.

3. **[CRITICAL] Chưa lọc `<|pad|>` trước `tokenizer.parse_response` & Trần `max_completion_length = 256` quá hẹp (`src/agent/lfm3_reward.py:20-34`, `src/agent/lfm3_grpo.py:28`):**
   - **Hiện trạng:** Notebook chính thức của LiquidAI (`lfm3_grpo_with_trl.ipynb:419`) bắt buộc lọc `output = output[output != tokenizer.pad_token_id]` trước khi gọi `tokenizer.parse_response(output)`. Code hiện tại truyền thẳng `completion_ids` còn nguyên `<|pad|>` bên phải vào parser; ngoài ra khi `raw_calls` rỗng (`OOS` / `final_stop`), hàm rơi xuống nhánh fallback thay vì xác nhận hợp lệ `([], True)` khi không có thẻ `<tool_call>` hỏng. Đồng thời `max_completion_length = 256` sẽ cắt đứt chuỗi gồm `<think>` (`60` tokens) + 3 lệnh `read`/`write` song song (`~280` tokens).
   - **Cách triệt tiêu:** Lọc sạch `pad_token_id` trước `parse_response`, xử lý chuẩn xác trường hợp `0` tool call hợp lệ vs thẻ `<tool_call>` bị cụt, và nâng `max_completion_length = 640` (`max_new_tokens = 640` khi suy luận).

4. **[CRITICAL] Vô hiệu hóa `<think>` (`enable_thinking=False`) làm mù khả năng tự sửa ý, tính thời gian & kiểm tra điều kiện của mô hình `1.7B` (`sft_dataset.py`, `llm_client.py`):**
   - **Bằng chứng thực nghiệm từ Smoke Test (`train_1096` & `train_0282`):** Với `<think></think>` rỗng (0 token nháp), `LFM3-1.7B` phải nhảy thẳng vào tên hàm ở token đầu tiên. Trên `train_1096`, mô hình không thể vừa xử lý câu tự sửa giờ (`"16:30 đến 17:15, à nhầm kết thúc lúc 17:05"` $\to$ `35` phút), vừa cộng học phí 2 môn (`1.8M + 1.5M = 3.3M`), vừa đối chiếu số dư ví (`2.9M < 3.3M` $\to$ không đủ trả cả hai!).
   - **Cách triệt tiêu:** Bật **`enable_thinking=True`** đồng bộ ở SFT, DPO, GRPO và Inference với **4-Slot Micro-`<think>` cô đọng (`35–85` tokens)** được chưng cất có kiểm chứng thực thi theo **Tool-N1 (ICLR 2026)** và **xLAM-2-fc-r / APIGen-MT (NeurIPS 2025)**.

5. **[HIGH] Mâu thuẫn Nội tại trong `apply_hammer_masking` & Thiếu Xáo trộn Thứ tự Tool (`scripts/build_lfm3_data.py:226-302`):**
   - **Hiện trạng:** Hàm chỉ đổi tên hàm/tham số trong `<tools>` (`smart_home.get_status -> fn_a8x2`) nhưng **giữ nguyên văn tên thật** trong `SYSTEM_PROMPT` (`prompts.py`: `"smart_home.get_status"`, `"banking.transfer"`, `"device_id"`) và ngay bên trong chuỗi `description` của tham số (`"[x-ref: smart_home.devices]"`), đồng thời **không xáo trộn danh sách `<tools>`** (`rng.shuffle(tools)` như `03_Hammer/train/data_processing.py:96,130`).
   - **Cách triệt tiêu:** Đồng bộ bản đồ ánh xạ tên giả trên cả `<tools>`, chuỗi `description`/`x-ref`, và bản `SYSTEM_PROMPT` trung tính hóa schema khi bật Hammer Masking, kết hợp xáo trộn ngẫu nhiên thứ tự `<tools>` ở 100% mẫu huấn luyện để triệt tiêu thiên kiến vị trí trong khối `LivConvolution` của `LFM3`.

6. **[HIGH] Cặp Phản ví dụ DPO Giả tạo (`_dummy_value_for_spec`), Sai trọng số Loss & Thiếu Phản ví dụ Hỏi bằng Text Thường (`scripts/build_lfm3_data.py:329-454`, `configs/lfm3_dpo.yaml`):**
   - **Hiện trạng:** Các câu `rejected` dùng chuỗi cứng `"id_unknown_01"`, `100000` quá dễ phân biệt; thiếu hẳn lỗi phổ biến nhất của LLM nhỏ theo **When2Call (NAACL 2025)**: **hỏi ngược người dùng bằng câu tiếng Việt thường thay vì gọi `sys.ask_missing_info`**, cũng như lỗi **điền thừa tham số optional**; đồng thời `loss_weights: [1.0, 0.5]` lệch với cấu hình chuẩn đã kiểm chứng của tác giả LiquidAI là `loss_weights: [1.0, 1.0]` (`lfm3_dpo_with_trl.ipynb:331`).
   - **Cách triệt tiêu:** Sinh **On-Policy Hard Negatives** trực tiếp từ `N=8` rollout của checkpoint Stage-1 SFT (`T=0.8` + greedy) bao phủ đủ **5 lớp lỗi ranh giới** và đặt `loss_weights=[1.0, 1.0]`.

7. **[HIGH] Thiếu Hoàn toàn Quỹ đạo Tự phục hồi Lỗi (`Error-Recovery Trajectories`) & Hàm Fallback `simulate_ctc_transcript` Lạc hậu (`sft_dataset.py`, `build_lfm3_data.py:70`):**
   - **Hiện trạng:** 100% dữ liệu SFT là đường đi hoàn hảo ngay lần đầu; khi gặp truy vấn trả về `[]` (do lệch dấu ASR) hoặc bị guardrail chặn `_xref_error`, `LFM3-1.7B` dễ rơi vào vòng lặp vô tận (như `train_0282`). Ngoài ra `simulate_ctc_transcript` mô phỏng lỗi CTC kiểu cũ (`"sms" -> "a ét mờ ét"`, chữ thường không dấu câu) trái ngược hoàn toàn đặc trưng đầu ra có viết hoa/dấu câu của `PhoWhisper-large`.
   - **Cách triệt tiêu:** Bổ sung **15% quỹ đạo SFT tự sửa lỗi trong vòng lặp** (thử lại `contacts.search` bằng tên ngắn khi rỗng, gọi `read` lấy ID khi bị chặn `_xref_error`, chuyển sang `sys.ask_missing_info` khi kết quả `read` có $\ge 2$ thực thể trùng tên) và thay `simulate_ctc_transcript` bằng bộ nhiễu âm vị chuẩn `PhoWhisper` (cùng vần/thanh điệu vùng miền Bắc–Trung–Nam).

---

### Nhóm B — Các lỗi Guardrail, Sanitizer & Suy luận (`tool_caller.py`, `react_loop.py`, `llm_client.py`)

8. **[CRITICAL — ĐÃ CHỨNG MINH ĐỊNH LƯỢNG] `sanitize_tool_call` ĐANG TỰ TAY PHÁ HỎNG `65` Gold Tool Calls (`62` mẫu = `4.98%` tập train) và ĐÁNH SẬP ĐIỂM CHÍNH THỨC trên `15` Mẫu Chuẩn Vàng (`-7.25` điểm gốc!) (`src/agent/tool_caller.py:144-238`):**
   - **Kiểm chứng thực nghiệm trực tiếp trên `1,246` mẫu `train_only`:** Ngay cả khi mô hình dự đoán **đúng 100% nhãn vàng**, chạy qua `sanitize_tool_call` hiện tại làm biến dạng `65 / 2,854` lệnh gốc và làm tụt điểm chính thức trên `15` mẫu (trong đó **3 mẫu bị giết về `0.00` điểm**):
     - `tool_caller.py:238` (`re.sub(r"[.!?]+", "", cleaned_content)`): Xóa sạch mọi dấu chấm `.` bên trong nội dung tin nhắn! Trên `train_1561`, đường link họp `"https://meet.example.test/lxq-jnpn-lga"` bị biến thành `"https://meetexampletest/lxq-jnpn-lga"` làm tụt F1; trên `train_0751` & `train_0830`, số tiền `"50.000"` biến thành `"50000"` và xóa dấu chấm cuối câu trên **50 mẫu** `chat_ott.send_message` / `sms.send`!
     - `tool_caller.py:199-201`: Xóa `mode="both"` của `robot_vacuum.start` khi người dùng nói *"vừa hút vừa lau"* (`train_1370`, `train_1646` — không có từ `"chế độ"`) $\implies$ **lệch `state_hash`, điểm sập từ `1.00` về `0.00`!**
     - `tool_caller.py:195-197`: Xóa `since` của `home_security.list_alerts` khi người dùng nói *"từ mười hai giờ trưa"* (`train_1394` — không có cụm `"từ lúc"`) $\implies$ **điểm sập từ `1.00` về `0.00`!**
     - `tool_caller.py:173-176`: Xóa trắng `args = {}` của `smart_home.get_status` trên `5` mẫu (`train_0955`, `train_0994`, `train_1009`, `train_1164`, `train_1722`) làm trượt `set(p) == set(g)`.
     - `tool_caller.py:167-171, 147-149, 178-181, 203-205`: Xóa oan `location` (`train_0037`, `train_1340`), xóa `label` báo thức được nhắc từ `history` lượt trước (`train_1318`), xóa `mode="car"` (`train_1322`, `train_1433`), xóa `date_from/date_to` (`train_1672`).
   - **Cách triệt tiêu:** **Xóa bỏ toàn bộ các luật `args.pop(...)` / `re.sub` đoán mò bằng từ khóa cứng!** Thay bằng nguyên tắc bất di bất dịch **"Zero-Gold-Corruption Invariant"**: mọi luật chuẩn hóa định dạng trong `sanitize_tool_call` phải vượt qua bài kiểm thử tự động `test_zero_gold_corruption_on_all_train_only` (**`0 / 2,854` gold calls bị biến đổi sai**), và giao việc chọn đúng tham số optional cho mô hình `LFM3-1.7B` đã qua Post-Training + Boundary Verifier.

9. **[HIGH] Phạt Precision từ Hậu tố Khẩu ngữ trong `contacts.search` & Gọi Lặp Lệnh `read` đã có trong `sample.history` (`src/agent/tool_caller.py:220-235`, `src/agent/react_loop.py:323`):**
   - **Hiện trạng:** Trên `train_0089`, `contacts.search(query="Hương dùm")` lọt qua `SafeEnv`, trả về `[]`, làm `unmatched_pred += 1` và mất trắng `-0.20` điểm (`1.00 -> 0.80`). Ngoài ra `seen_read_calls` trong `react_loop.py:323` khởi tạo rỗng `set()`, không nạp các lệnh `read` đã chạy sẵn trong `sample.history` có kết quả rõ ràng.
   - **Cách triệt tiêu:** Tách chuẩn hóa hậu tố khẩu ngữ tiếng Việt cuối câu truy vấn tìm kiếm danh bạ (`dùm/giùm/nha/nhé/giúp/hộ/với/đi/ạ` đi kèm đại từ `em/anh/chị/tôi/mình`, kiểm chứng `0` va chạm với tên thật trong `train_only`) và hỗ trợ cơ chế tra cứu/khử trùng lặp thông minh với `history`.

10. **[HIGH] Kiểm tra `x-ref` Thô theo Cấp Ứng dụng thay vì Cấp Bảng `<app>.<collection>` (`src/agent/react_loop.py:80-94, 239-270`):**
    - **Hiện trạng:** `known_ids_by_app` gom chung mọi ID theo tên `app` (`"music"`, `"smart_home"`), cho phép mô hình điền nhầm `playlist_id` vào tham số `track_id` (`[x-ref: music.tracks]`) hoặc `scene_id` vào `device_id` (`[x-ref: smart_home.devices]`).
    - **Cách triệt tiêu:** Ánh xạ chính xác đầu ra của từng hàm `read` vào đúng bảng `<app>.<collection>` tương ứng với nhãn `x-ref` trong `catalog`.

11. **[HIGH] Không Tái sử dụng KV-Cache / Prefix Caching Qua Các Bước ReAct (`src/agent/llm_client.py:130-198`):**
    - **Hiện trạng:** Mỗi bước ReAct trong cùng 1 mẫu đều nạp và tính lại từ đầu `~6,200` token của `SYSTEM_PROMPT + <tools> + history`. Trên `train_0282` (4 bước), thời gian đội lên **`26.48s`** và đụng ngưỡng cắt an toàn `23s`!
    - **Cách triệt tiêu:** Tích hợp **`vLLM` với `enable_prefix_caching=True`** (hoặc `Transformers` `DynamicCache` giữ trạng thái `LivConvolution` + KV-cache của prefix giữa các lượt trong cùng 1 mẫu). Tốc độ từ lượt ReAct thứ 2 tăng **`5x–7x`**, kéo thời gian trung bình toàn mẫu xuống **`~2.0s–3.0s`** trên A30, mở ra ngân sách thời gian dư dả cho cả `<think>` lẫn bước hậu kiểm Verifier!

12. **[MEDIUM] Bỏ phí `2.73B` Ngân sách Tham số Hợp lệ dưới Trần `< 6.0B` cho Bước Hậu kiểm Trước Khi Ghi (`Pre-Flight Action & Boundary Verifier`):**
    - **Hiện trạng:** Hệ thống mới dùng `3.27B / 6.00B` (`54.5%` hạn mức cho phép). Mọi quyết định nhạy cảm (`Act` vs `Ask` vs `OOS`, có đưa tham số optional vào lệnh `write` hay không) đều đặt cược 100% vào một lần sinh duy nhất.
    - **Cách triệt tiêu:** Huấn luyện thêm một mô hình **Cross-Encoder Verifier siêu nhẹ (`BAAI/bge-reranker-v2-m3` `0.56B` hoặc `mDeBERTa-v3-base` `0.28B`)** tổng cộng vẫn chỉ **`3.55B–3.83B << 6.00B`**, chuyên chấm xác suất hợp lệ của `(transcript + context + read_obs, candidate_tool_call)` trước khi cho phép lệnh `write` chạm vào `SafeEnv`!

---

### Nhóm C — Các lỗi Tầng Nhận dạng Giọng nói ASR (`whisper_engine.py`, `run_asr.py`) & Quản trị Đánh giá

13. **[CRITICAL] Lỗi Rơi Ngữ cảnh Âm thầm trong `WhisperASREngine.transcribe_batch` (`src/asr/whisper_engine.py:378-380`):**
    - **Hiện trạng:** Dòng `378-380` viết: `same_context = all(c == first_ctx for c in chunk_contexts); effective_ctx = first_ctx if same_context else None`. Vì mỗi mẫu trong một batch (`batch_size=8`) có `user_profile` và `history` khác nhau, `same_context` **LUÔN BẰNG `False`**, dẫn đến `effective_ctx` bị gán về **`None` âm thầm cho 100% mẫu khi chạy theo batch**! Toàn bộ `build_sample_asr_context` trong `scripts/run_asr.py` thực chất bị vô hiệu hóa hoàn toàn khi `batch_size > 1`!
    - **Cách triệt tiêu:** Sửa `whisper_engine.py` để truyền mảng `prompt_ids` riêng biệt cho từng mẫu hoặc nhóm theo mẫu khi có `extra_context` cá nhân hóa, đồng thời bổ sung từ vựng tên riêng/thuật ngữ của `installed_apps` vào `prompt_ids`.

14. **[HIGH] Giải mã Tham lam `num_beams=1`, Chưa Tận dụng `4,042` Audio In-Domain & Thiếu N-Best Phonetic Grounding theo `AURA` (`whisper_engine.py`, `run_asr.py`):**
    - **Bằng chứng thực nghiệm (`train_1096`, `loop_042`):** `PhoWhisper` chạy `num_beams=1` nghe nhầm đồng âm tiếng Việt quyết định nghĩa điều kiện: `"đủ" -> "đụng"`, `"tiếng Hàn" -> "lẫn hàng"`, `"tắt" -> "bắt"`, `"giờ hẹn" -> "giờ huỷ"`.
    - **Cách triệt tiêu:** (a) Fine-tune **5-Fold OOF + Full Hyper-Regularized Acoustic LoRA (`r=8, alpha=16`)** cho `PhoWhisper-large` trên toàn bộ **`4,042` câu audio** (`1,246` chính + `2,796` `history`); (b) giải mã tất định `do_sample=False, num_beams=5, num_return_sequences=3`; (c) áp dụng kỹ thuật **Uncertainty-Aware Phonetic Diff (`AURA`, IEEE ASRU 2025)**: chỉ khi `Beam-2` khác `Beam-1` ở cụm từ khóa thực thể/số/phủ định (với biên độ logprob hẹp), đính kèm ghi chú gọn `[Gợi ý đồng âm ASR: ...]` để `LFM3-1.7B` đối chiếu với kết quả `SafeEnv` (`learning.get_courses` thấy `"Tiếng Hàn Sơ Cấp"` $\implies$ tự giải được `"lẫn hàng"` là `"lẫn Hàn"`!).

15. **[HIGH] Thiếu Giao thức Phân chia K-Fold Phân tầng Đa nhãn Nội bộ (`Stratified 5-Fold CV`) trên `train_only`:**
    - **Hiện trạng:** `dev_v1.json` (`299` mẫu) là tập mù bất khả xâm phạm, nhưng bên trong `1,246` mẫu `train_only` chưa có tập `val_fold` chuẩn hóa để chọn siêu tham số, dừng sớm DPO/RL, huấn luyện Cross-Encoder Verifier out-of-fold, và ghép súp trọng số (**Model Soup**).
    - **Cách triệt tiêu:** Tạo cố định `configs/splits/train_only_5folds.json` phân tầng đồng thời theo `(category, is_multi_app, has_history, rare_app_bucket)`: dùng Fold 0 (`249` mẫu) làm chốt kiểm định nhanh cho mọi ablation, chạy OOF 5-Fold cho ASR + Verifier để **0% rò rỉ**, và cuối cùng hợp nhất **5-Fold Model Soup / Full `1,246` Retrain** cho checkpoint thi đấu!

---

## 3. Đúc kết Tinh hoa Thuật toán từ 14 Bài báo (2024–2026), 12 Public Repos & 3 Official LiquidAI Notebooks

Không dừng ở trích dẫn lý thuyết, chúng ta trích xuất **trực tiếp các cơ chế toán học và module mã nguồn** đã được chứng minh hiệu quả nhất trên mô hình nhỏ `<6B` từ kho tài liệu đã thu thập tại `references/tool_calling_papers/`, `references/tool_calling_repos/` và `src/lfm3-post-training/`:

| Nguồn (Paper / Repo / Notebook) | Cơ chế Cốt lõi & File Mã nguồn Tham chiếu | Cách Chuyển hóa Tối ưu vào `Apex-LFM3` |
|---|---|---|
| **Official LiquidAI `LFM3` Post-Training** (`src/lfm3-post-training/lfm3_{sft,dpo,grpo}_with_trl.ipynb`) | - Kiến trúc Tri-Hybrid: 10 module mục tiêu (`q/k/v/o/gate_proj`, `in/out_proj` của `LivConvolution`, `w1/w2/w3` của `SwiGLU`).<br>- DPO kết hợp `loss_type=["sigmoid", "sft"], loss_weights=[1.0, 1.0], beta=0.1` từ on-policy rollouts.<br>- GRPO: oversample lớp từ chối/hỏi lại, lọc `pad_token_id` trước `tokenizer.parse_response`. | Nền tảng gốc cho cả 3 tầng huấn luyện; nâng cấp thêm **Full-FT Differential-LR vs Rank-64 DoRA ablation** để vắt kiệt dung lượng biểu diễn của `1.73B` tham số. |
| **Tool-N1** (ICLR 2026) — `01_Tool-N1/data_process/distill_data.py` & `qwen_rl.sh` | Chưng cất suy luận `<think>` bằng **Rejection Sampling có kiểm chứng kết quả gọi hàm**: sinh nhiều suy luận từ Teacher cho mỗi bước, **chỉ giữ lại mẫu mà `<tool_call>` sinh ra khớp chính xác nhãn chuẩn**, sau đó chạy RLEF hai giai đoạn. | Sinh **4-Slot Micro-`<think>` (`35–85` tokens)** cho toàn bộ các bước của `train_only` + dữ liệu tổng hợp; loại bỏ 100% `<think>` gây ảo giác tham số. |
| **xLAM-2-fc-r / APIGen-MT** (NeurIPS 2025) — `06_xLAM_APIGen-MT/actionstudio/` (`trajectory_critic.py`) | **Agentic Feedback Loop 2 Pha:** (1) Từ trạng thái môi trường thật + đồ thị phụ thuộc tool, sinh ngược **Task Blueprint** rồi mới diễn dịch thành hội thoại đa lượt; (2) Kiểm duyệt kép (Rule-based execution check + Semantic Trajectory Critic). Giúp `xLAM-2-1B/3B` vượt GPT-4o trên $\tau$-bench! | Chạy `read` hợp pháp trên các sandbox `train_only` để lấy thực thể thật, sinh **+3,500 quỹ đạo đa lượt tiếng Việt** phủ dày **25 ứng dụng đuôi dài** (`fuel_price`, `lottery`, `tv`...) + kịch bản tự sửa ý & điều kiện rẽ nhánh, thực thi thật trong `SafeEnv` để lấy `gold_state_hash` chuẩn 100%! |
| **Hammer & HammerBench** (ICLR 2025 / ACL 2025) — `03_Hammer/train/data_processing.py` | Đổi tên ngẫu nhiên tên hàm/tham số (`mask_Identifier`) + **xáo trộn ngẫu nhiên thứ tự `<tools>` (`random.shuffle`)** + trộn mẫu Irrelevant/Missing để buộc SLM đọc kỹ `description` thay vì học vẹt tên hàm. | Sửa triệt để lỗi mâu thuẫn prompt khi mask: đồng bộ đổi tên trên cả `<tools>`, `x-ref` và prompt trung tính ở `30%` bản sao SFT; bật `rng.shuffle(tools)` trên `100%` bản sao huấn luyện. |
| **When2Call** (NAACL 2025) — `08_When2Call/synthetic_data_gen/convert_raw_train_data_to_pref.py` | Phân tách 4 hành vi quyết định (`Direct Answer/OOS`, `Tool Call`, `Clarification/Ask`, `Follow-up`) và xây dựng cặp đối chiếu DPO phạt nặng hành vi **đoán mò khi thiếu thông tin** hoặc **hỏi lại bằng văn bản thường khi đã có công cụ hỏi chuẩn**. | Xây dựng kho **On-Policy Multi-Negative DPO (~12,000 cặp)** phủ kín 5 lỗi chí mạng: `missed_ask`, `false_ask`, `plain_text_ask`, `missed_oos`, và `hallucinated_optional_arg`. |
| **ToolRL** (NeurIPS 2025 Oral) — `04_ToolRL/verl/utils/reward_score/rlla.py` | Hàm thưởng RL phân rã chi tiết, chuẩn hóa thang cố định `[-3, +3]` (Format + Tool Name Jaccard + Param Keys Exact + Param Values Match), dùng ghép cặp hai phía tối ưu (`bipartite matching`), tránh sụp đổ chính sách ở mô hình `<4B`. | Cài đặt hàm thưởng bước mật độ cao (Dense Step Reward) kết hợp **cổng cứng `step_gold_state_hash`** cho Stage 3A GRPO. |
| **GiGPO** (NeurIPS 2025) — `05_GiGPO_verl-agent/gigpo/core_gigpo.py` | **Group-in-Group Policy Optimization:** Gộp lợi thế toàn cục cấp quỹ đạo ($A^E$, từ kết quả cuối phiên) với lợi thế vi mô cấp bước ($A^S$, gom nhóm các hành động xuất phát từ cùng **`AnchorState`** qua bảng băm trạng thái môi trường) + Asymmetric PPO Clip `[0.20, 0.28]`: $A_{i,t} = A^E_i + \omega A^S_{i,t}$. | **Vũ khí tối thượng cho Stage 3B:** Chạy rollout đa bước trực tiếp trong `SafeEnv`! Định nghĩa `AnchorState = (sample_id, env.get_state_hash(), read_obs_digest)` để gán tín hiệu thưởng/phạt chính xác đến **từng bước `read` trung gian và bước `write` quyết định** dựa trên điểm `score_sample` chính thức cuối phiên! |
| **LoopTool** (ICLR 2026) — `07_LoopTool/dataloop/` (`greedy_capability_prob.py`, `filter_grpo_sample.py`, `error_data_expansion.py`) | Vòng lặp khép kín đồng tiến hóa Dữ liệu–Mô hình: (1) Dò năng lực bằng Greedy vs `pass@k`; (2) **Dynamic GRPO Filtering** loại bỏ mẫu quá dễ (`pass@k = 1.0`) và mẫu nhiễu vô nghiệm (`pass@k = 0.0`); (3) Nhân rộng `5x` dữ liệu sửa sai đúng cụm lỗi mô hình vừa vấp (`Error-Driven Data Expansion`). | Chạy **3 vòng lặp `LoopTool`** sau chuỗi SFT $\to$ DPO $\to$ GiGPO ban đầu để quét sạch mọi góc khuất lỗi còn sót lại của `LFM3-1.7B`! |
| **AURA** (IEEE ASRU 2025) & **SimpleTool** (ICML 2025) | `AURA`: Khai thác độ bất định âm học N-best từ ASR đưa vào tầng quyết định tool để chống nhầm đồng âm/thực thể. `SimpleTool`: Rút gọn biểu diễn schema trực diện, đặt ràng buộc kiểu/giá trị sát tham số giúp SLM sinh nhanh và chính xác hơn. | Đưa chênh lệch âm vị có điều kiện từ `PhoWhisper` `5-beam` (`Beam-1` vs `Beam-2`) vào user context và tối ưu hóa mô tả schema gọn sắc nét trong `tool_selector.py`. |

---

## 4. Kiến trúc Đích `Apex-LFM3`: Nhà máy Dữ liệu Offline & Hệ thống Suy luận Đích

### 4.1. Sơ đồ Tổng thể Hai Pha (Offline Unlimited Training Factory $\longrightarrow$ Deployed `<6B` Deterministic Engine)

```
====================================================================================================
[A] NHÀ MÁY DỮ LIỆU & HUẤN LUYỆN ĐA TẦNG OFFLINE (CHỈ DÙNG 1,246 MẪU TRAIN_ONLY — 0% RÒ RỈ DEV_V1)
====================================================================================================
  4,042 Train Audio (.wav) ──► [5-Fold OOF PhoWhisper-Large LoRA (r=8)] ──► Real OOF N-Best Transcripts
                                                                                      │
  1,246 Train Sandboxes ─────► [SafeEnv Read Explorer + APIGen-MT Blueprint] ─────────┼──► +3,500 Quỹ đạo Đa lượt
  (46 apps, ưu tiên 25 app hiếm, multi-app, tự sửa ý, điều kiện toán/giờ)             │    (Kiểm chứng SafeEnv Hash)
                                                                                      ▼
                        [Tool-N1 Rejection-Sampled 4-Slot Micro-<think> Distiller + Recovery Augmenter]
                                                      │
         ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
         ▼                                            ▼                                            ▼
  [STAGE 1: CURRICULUM SFT]              [STAGE 2: ON-POLICY DPO+SFT]             [STAGE 3A -> 3B: RLVR & GiGPO]
  - LiquidAI/LFM3-1.7B                   - Rollout N=8 từ Stage-1                 - 3A: Dynamic-Filtered Step GRPO
  - ~38,000 bước (Hammer Mask 30%        - 5 lớp Hard Negatives (When2Call          (ToolRL + step_gold_state_hash)
    + Shuffle Tools 100% + Recovery)       + Optional-Arg Trap)                   - 3B: Live Multi-Turn SafeEnv GiGPO
  - Ablation: Diff-LR Full-FT            - loss=["sigmoid","sft"], w=[1.0,1.0]      (AnchorState = state_hash,
    vs Rank-64 DoRA (10 modules)                                                     Macro = Official score_sample)
         └────────────────────────────────────────────┴────────────────────────────────────────────┘
                                                      │
                                                      ▼
         [STAGE 4: LOOPTOOL (3 VÒNG KHÉP KÍN: PROBING -> 5x ERROR EXPANSION -> WARM RL) + 5-FOLD MODEL SOUP]
                                                      │
                                                      ▼
         [STAGE 5: OUT-OF-FOLD ACTION & OPTIONAL-ARG BOUNDARY VERIFIER (mDeBERTa-v3 / BGE-Reranker 0.28B-0.56B)]

====================================================================================================
[B] HỆ THỐNG SUY LUẬN ĐÍCH TRONG DOCKER OFFLINE (TẤT ĐỊNH 100%, TỔNG 3.55B-3.83B < 6B, ~2.3s/MẪU A30)
====================================================================================================
  Audio (.wav) ─► [PhoWhisper-Large + LoRA (1.54B)] ─► Beam-1 + Conditional AURA Phonetic Alt
                  (num_beams=5, dynamic app/profile prompt_ids)
                                    │
                                    ▼
                  [LiquidAI/LFM3-1.7B Merged Soup (1.73B)]  ◄─── vLLM Prefix Cache / DynamicCache
                  - Sinh <think> cô đọng (40-75 tokens) + <tool_call> (greedy T=0.0)
                                    │
                                    ▼
                  [Pre-Flight Safe Gate + Collection-Exact x-ref + Boundary Verifier (0.28B-0.56B)]
                  - Chặn đọc trùng history, lọc từ đệm danh bạ (0% hỏng nhãn vàng)
                  - Kiểm định ranh giới Act/Ask/OOS & tham số Optional trước khi gọi lệnh WRITE
                                    │
                                    ▼
                  [SafeEnv Execution] ──► (Nếu còn bước tiếp: nạp Observation vào KV-Cache, lặp ReAct)
                                      ──► Xuất tool_calls + final state_hash
```

### 4.2. Kiểm toán Ngân sách Tham số & Độ trễ Suy luận trên NVIDIA A30 (24GB VRAM)

| Thành phần Suy luận | Mô hình Cụ thể | Số Tham số | VRAM (`bf16`) | Độ trễ TB / Mẫu (A30) | Tuân thủ Luật thi |
|---|---|---|---|---|---|
| **1. Acoustic ASR** | `PhoWhisper-large` + Merged Domain LoRA | **`1.54B`** | `~3.2 GB` | `0.38s` (`num_beams=5`) | Hợp lệ |
| **2. Core Agent LLM** | `LiquidAI/LFM3-1.7B` (Merged 5-Fold Soup, **không dùng Qwen3.5**) | **`1.73B`** | `~3.6 GB` | `1.85s` (3.2 bước ReAct với Prefix Cache + `<think>` 60 tok) | `< 5.0B` LLM (`1.73B`) |
| **3. Boundary Verifier** | Fine-tuned `BAAI/bge-reranker-v2-m3` (hoặc `mDeBERTa-v3-base`) | **`0.56B`** (`0.28B`) | `~1.2 GB` | `0.06s` (chỉ chạy trên bước ranh giới / `write`) | Hợp lệ |
| **4. Môi trường & Parser** | `SafeEnv` + `Collection-Exact Guard` | `0.00B` | CPU | `0.03s` | Chỉ gọi hàm hợp pháp |
| **TỔNG HỆ THỐNG** | **Offline Deterministic Pipeline** | **`3.83B`** (`3.55B`) | **`~11.5 GB / 24 GB`** | **`~2.32s / 30.0s`** | **ĐẠT TUYỆT ĐỐI (`<< 6.0B`, `<< 30s`)** |

---

## 5. Kế hoạch Thực thi Chi tiết 8 Pha (Implementation-Ready Specs, Code, Equations & Tests)

---

### PHA 0 — Thiết lập K-Fold Phân tầng Chống Rò rỉ & Phẫu thuật Triệt để 15 Lỗi Gốc Pipeline
**Mục tiêu:** Dựng nền móng đánh giá 0% rò rỉ dữ liệu, xóa bỏ 100% lỗi tự hủy điểm của `sanitize_tool_call`, sửa sạch lỗi nhân đôi token `<|startoftext|>`/`<|im_end|>`, sửa `step_gold_state_hash`, và siết chặt `x-ref` theo bảng.

#### Task 0.1: Tạo Bộ Chia 5-Fold Phân tầng Đa nhãn Cố định (`scripts/make_folds.py` & `configs/splits/train_only_5folds.json`)
- **Files:**
  - Create: `scripts/make_folds.py`
  - Output: `configs/splits/train_only_5folds.json`
  - Test: `tests/test_folds_no_leak.py`
- **Thiết kế chi tiết:**
  - Nạp `train.jsonl` và loại trừ mù các ID trong `configs/splits/dev_v1.json` (tuyệt đối không đọc nội dung mẫu `dev_v1`), thu được đúng `1,246` mẫu `train_only`.
  - Gán khóa phân tầng kết hợp cho mỗi mẫu: `strata_key = f"{category}_{is_multi_app}_{has_history}_{rarest_app_name}"` (ưu tiên phân bổ đều `9` mẫu `fuel_price`, `15` mẫu `tv`, `17` mẫu `lottery`... vào cả 5 fold, mỗi fold `~249` mẫu).
  - Unit test `tests/test_folds_no_leak.py` kiểm chứng: (1) giao của 5 fold với `dev_v1` bằng $\emptyset$; (2) hợp của 5 fold bằng đúng `1,246` mẫu; (3) mọi ứng dụng trong 46 ứng dụng đều xuất hiện ở cả tập train lẫn tập val của từng fold.

#### Task 0.2: Tái cấu trúc `sanitize_tool_call` theo Tiên đề "Zero-Gold-Corruption" (`src/agent/tool_caller.py`)
- **Files:**
  - Modify: `src/agent/tool_caller.py`
  - Test: `tests/test_zero_gold_corruption.py`
- **Thiết kế chi tiết:**
  - **Xóa bỏ toàn bộ** các nhánh `args.pop(...)` và `re.sub(r"[.!?]+", ...)` phá hoại ở dòng `144–206` và `236–238` (`alarm.create`, `reminders.create`, `bill_payment.list_bills`, `smart_home.light_set`, `calendar.create_event`, `calendar.update_event`, `weather.*`, `smart_home.get_status`, `calendar.list_events`, `robot_vacuum.start`, `home_security.list_alerts`, `maps.get_route`, `sms.send`, `chat_ott.send_message`).
  - **Chỉ giữ lại và nâng cấp các phép chuẩn hóa bảo toàn ngữ nghĩa 100%:**
    1. Chuẩn hóa Unicode `NFC` và cắt khoảng trắng thừa đầu/cuối chuỗi (`str.strip()`).
    2. Ánh xạ đại từ địa chỉ rõ ràng (`"nhà"`, `"nhà riêng"`, `"home"` $\to$ `user_profile["home_address"]`; `"công ty"`, `"văn phòng"`, `"work"` $\to$ `user_profile["work_address"]`) trên các trường địa chỉ.
    3. Bộ lọc từ đệm lịch sự cuối truy vấn **riêng cho `contacts.search(query=...)`**:
       ```python
       _CONTACT_POLITE_SUFFIX_RE = re.compile(
           r"\s+(?:giúp|giùm|dùm|hộ)(?:\s+(?:em|anh|chị|tôi|mình|con|cháu))?(?:\s+(?:với|nha|nhé|đi|ạ|cái))?$"
           r"|\s+(?:nha|nhé|ạ|với\s+ạ)$",
           flags=re.IGNORECASE,
       )
       ```
       (Cắt `"Hương dùm"` $\to$ `"Hương"`, `"chị Lan giúp em nhé"` $\to$ `"chị Lan"`, nhưng không bao giờ chạm vào các tên thật như `"Anh Dũng"`, `"Cô Hạnh"`).
  - **Cổng kiểm định bắt buộc (`tests/test_zero_gold_corruption.py`):** Chạy `sanitize_tool_call` trên toàn bộ **`2,854` gold tool calls của `1,246` mẫu `train_only`**, khẳng định `assert corrupted_count == 0` và `assert official_score_loss == 0.0`!

#### Task 0.3: Nâng cấp Guardrail `x-ref` Cấp Bảng `<app>.<collection>` & Khử Trùng lặp Lệnh `read` từ `history` (`src/agent/react_loop.py`)
- **Files:**
  - Modify: `src/agent/react_loop.py`
  - Test: `tests/test_collection_xref_guard.py`
- **Thiết kế chi tiết:**
  - Thay `known_ids_by_app: dict[str, set[str]]` bằng `known_ids_by_collection: dict[str, set[str]]` khớp trực tiếp với khóa `x-ref` trong `train_tools_catalog.json` (ví dụ `music.list_tracks` ghi vào `"music.tracks"`, `music.list_playlists` ghi vào `"music.playlists"`, `smart_home.get_status` ghi `devices` vào `"smart_home.devices"` và `scenes` vào `"smart_home.scenes"`).
  - Khi khởi tạo vòng lặp `run_sample`, quét toàn bộ các cặp `tool_calls` + `tool_observations` trong `sample.history` để:
    1. Nạp sẵn các ID hợp lệ đã biết từ `history` vào `known_ids_by_collection`.
    2. Đưa chữ ký `(call.name, canonical_json(call.arguments))` của các lệnh `read` tĩnh đã chạy trong lượt sát trước vào `history_read_cache` — nếu ở bước 1 mô hình gọi lại y hệt lệnh `read` vừa có sẵn kết quả đầy đủ trong `history` mà không có lệnh `write` nào xen giữa, trả lại ngay observation từ `history` cho mô hình **mà không tốn thêm 1 lượt gọi thừa vào `SafeEnv`** (bảo toàn `precision = 1.00`!).

#### Task 0.4: Sửa Lỗi Nhân đôi BOS/EOS, Lọc `<|pad|>` & Gắn `step_gold_state_hash` cho Mọi Bước (`src/agent/sft_dataset.py`, `scripts/build_lfm3_data.py`, `src/agent/lfm3_sft.py`, `src/agent/lfm3_dpo.py`, `src/agent/lfm3_reward.py`)
- **Files:**
  - Modify: `src/agent/sft_dataset.py`, `scripts/build_lfm3_data.py`, `src/agent/lfm3_sft.py`, `src/agent/lfm3_dpo.py`, `src/agent/lfm3_grpo.py`, `src/agent/lfm3_reward.py`
  - Test: `tests/test_lfm3_token_hygiene.py`, `tests/test_lfm3_reward_state_hash.py`
- **Thiết kế chi tiết:**
  1. **Chuẩn hóa Tokenization cho `LFM3-1.7B`:**
     - Lưu bản ghi SFT/DPO/GRPO dưới dạng danh sách messages chuẩn OpenAI/LiquidAI kèm schema `tools` (hoặc hàm tiền xử lý tạo trực tiếp `input_ids`, `attention_mask`, `labels` bằng `tokenizer.apply_chat_template(..., tokenize=True, enable_thinking=True)` một lần duy nhất, gán `labels[:len(prompt_ids)] = -100`).
     - Test bắt buộc với tokenizer thật hoặc mock cấu trúc `LFM3`: `input_ids.count(bos_token_id) == 1`, phần completion kết thúc chính xác bằng `[..., im_end_id, newline_id]` (hoặc `im_end_id` chuẩn của `LFM3`), `0` token prompt bị lọt vào hàm loss (`all(x == -100 for x in labels[:prompt_len])`).
  2. **Gắn `step_gold_state_hash` trên từng bước unroll (`sft_dataset.py`):**
     - Trong `unroll_trajectory_steps(traj, split, root)`: khởi tạo một phiên `SafeEnv(split, sample_id, root)`, thực thi tuần tự `prefix_calls` $\to$ ghi nhận `pre_step_state_hash`, thực thi tiếp `step_calls` $\to$ ghi nhận **`step_gold_state_hash = env.get_state_hash()`**.
     - Trong `lfm3_reward.py`: ở mọi bước (`read`, `write`, `ask`, `oos`, `final_stop`), thực thi `prefix_calls + pred_calls` trên `SafeEnv` và kiểm tra `pred_step_hash == step_gold_state_hash`! Nếu bước `write` sai `state_hash`, áp phạt cứng tức thì thay vì đợi đến `final_stop`.
  3. **Sửa `extract_predicted_calls` (`lfm3_reward.py:20-34`):**
     - Lọc bỏ toàn bộ `pad_token_id` trước khi gọi `tokenizer.parse_response(clean_ids)`.
     - Phân biệt rõ 3 trạng thái: (a) có `<tool_call>` hợp lệ $\to$ `(parsed_calls, True)`; (b) không chứa chuỗi `"<tool_call>"` hay `"</tool_call>"` (tức phản hồi text/OOS/final_stop sạch) $\to$ `([], True)`; (c) có dấu hiệu thẻ `<tool_call>` bị gãy/cụt JSON/XML $\to$ `([], False)`.
     - Nâng `max_completion_length = 640` trong `configs/lfm3_grpo.yaml` và `src/agent/lfm3_grpo.py`.

---

### PHA 1 — Hoàn thiện Tầng Âm học (`PhoWhisper-Large` Domain LoRA + Context Biasing + `AURA` N-Best Grounding)
**Mục tiêu:** Đưa WER thực tế về **`<= 0.8%`** và cung cấp thông tin đồng âm có kiểm soát để triệt tiêu các ca chết oan kiểu `"đụng/đủ"`, `"hàng/Hàn"`, `"bắt/tắt"`, `"huỷ/hẹn"`.

#### Task 1.1: Sửa Lỗi Rơi Ngữ cảnh Batch & Bổ sung Từ vựng Ứng dụng Động (`src/asr/whisper_engine.py`, `scripts/run_asr.py`)
- **Files:**
  - Modify: `src/asr/whisper_engine.py`, `scripts/run_asr.py`
  - Test: `tests/test_whisper_context_biasing.py`
- **Thiết kế chi tiết:**
  - Sửa `WhisperASREngine.transcribe_batch` (`whisper_engine.py:378-380`): khi các mẫu trong chunk có `extra_context` khác nhau, chạy giải mã theo từng mẫu với `prompt_ids` riêng (với `PhoWhisper` `1.54B` trên GPU, `num_beams=5` chỉ mất `~0.35s/câu`, vẫn đạt `1,246` câu trong `< 8 phút`), **tuyệt đối không gán `effective_ctx = None`**.
  - Nâng cấp `build_sample_asr_context(sample, catalog)` để ghép nối ngắn gọn (`<= 180` tokens):
    1. Tên danh bạ / người nhận / tên phòng / khóa học xuất hiện trong `sample.history`.
    2. `user_profile` (`name`, `home_address`, `work_address`, `city`).
    3. Từ khóa miền chuyên biệt của `sample.context["installed_apps"]` (ví dụ có `learning` $\to$ `"khóa học, tiếng Hàn, tiếng Nhật, học phí"`; có `ewallet`/`banking` $\to$ `"chuyển khoản, số dư, ví điện tử"`).

#### Task 1.2: Huấn luyện 5-Fold OOF + Full Hyper-Regularized Acoustic LoRA trên `4,042` Câu Audio (`scripts/train_phowhisper_lora.py`)
- **Files:**
  - Create: `scripts/train_phowhisper_lora.py`, `configs/asr_phowhisper_lora.yaml`
- **Thiết kế chi tiết (kế thừa bài học từ `loop_042` để chống overfit thanh điệu):**
  - Trích xuất toàn bộ **`4,042` cặp `(audio_wav, gold_transcript)`** hợp lệ từ `1,246` mẫu `train_only` (`1,246` file chính + `2,796` file lượt người dùng trong `sample.history`).
  - Cấu hình **Dual-Tower Hyper-Regularized LoRA** trên `kzap201/phowhisper-large-zaloai`:
    - `r = 8, lora_alpha = 16, lora_dropout = 0.15` gắn vào cả Encoder self-attention lẫn Decoder cross/self-attention (`q_proj, k_proj, v_proj, out_proj`).
    - Bật **SpecAugment** mạnh (`mask_time_prob=0.08`, `mask_feature_prob=0.08`), `label_smoothing_factor=0.05`, `lr=3e-5`, `2 epochs`, early stopping theo `val_fold` CER.
  - **Quy trình chống rò rỉ dữ liệu cho LLM:**
    - Chạy **5-Fold Out-Of-Fold (OOF)**: mô hình ASR train trên 4 fold sẽ dịch âm thanh của fold còn lại $\implies$ tạo tập **`train_only_oof_nbest.jsonl`** mang **phân phối nhiễu ASR thực tế y hệt lúc thi** để huấn luyện `LFM3-1.7B`!
    - Sau đó train 1 bản **Full `4,042`-Audio Merged LoRA** đẩy lên Hugging Face Hub để dùng khi suy luận thật.

#### Task 1.3: Tích hợp `AURA` Uncertainty-Aware Phonetic Diff (`src/asr/aura_nbest.py`)
- **Files:**
  - Create: `src/asr/aura_nbest.py`
  - Modify: `src/agent/prompts.py`
  - Test: `tests/test_aura_nbest_diff.py`
- **Thiết kế chi tiết:**
  - Giải mã tất định `do_sample=False, num_beams=5, num_return_sequences=3, return_dict_in_generate=True, output_scores=True`.
  - Chỉ kích hoạt chú thích phụ khi **đồng thời thỏa mãn 2 điều kiện**:
    1. Khoảng cách xác suất chuẩn hóa giữa `Beam-1` và `Beam-2` hẹp: $\Delta \log p = \log p(b_1) - \log p(b_2) < 0.45$.
    2. Phần từ khác nhau giữa $b_1$ và $b_2$ rơi vào từ thực nội dung (không phải dấu câu/viết hoa đơn thuần), ví dụ `b1 = "...đụng trả cả hai môn tiếng Nhật lẫn hàng..."` vs `b2 = "...đủ trả cả hai môn tiếng Nhật lẫn Hàn..."`.
  - Khi kích hoạt, nối thêm đúng 1 dòng siêu gọn dưới câu nói người dùng:
    `[ASR Beam-2 đồng âm cần đối chiếu ngữ cảnh: "đụng ... lẫn hàng" <-> "đủ ... lẫn Hàn"]`
  - Nhờ dòng này + `<think>`, khi `LFM3-1.7B` gọi `learning.get_courses` và nhìn thấy khóa học `"Tiếng Hàn Sơ Cấp"`, nó chọn đúng 100% ý nghĩa gốc mà không bị đánh lừa bởi lỗi nghe nhầm đơn lẻ của Beam-1!

---

### PHA 2 — Nhà máy Tổng hợp Dữ liệu Đa tác tử & Chưng cất `<think>` Có Kiểm chứng (`~38,000+` Bước)
**Mục tiêu:** Giải quyết tận gốc bài toán khan hiếm dữ liệu trên 25 ứng dụng đuôi dài, trang bị tư duy `<think>` sắc bén và khả năng tự sửa lỗi trong vòng lặp cho `LiquidAI/LFM3-1.7B`.

#### Task 2.1: Thiết kế Chuẩn 4-Slot Micro-`<think>` & Chưng cất Có Kiểm chứng theo `Tool-N1` (`src/data_gen/distill_think.py`)
- **Files:**
  - Create: `src/data_gen/distill_think.py`
  - Modify: `src/agent/sft_dataset.py`, `src/agent/prompts.py`
- **Định dạng 4-Slot Micro-`<think>` chuẩn hóa (`35–85` tokens, cố định cấu trúc ngắn gọn để tối ưu tốc độ & độ chính xác):**
  ```xml
  <think>
  1. Ý định: Đổi lịch học sang 16:30-17:05 (tự sửa từ 17:15 -> 35 phút); kiểm tra ví nếu đủ trả cả Nhật + Hàn thì đóng.
  2. Dữ liệu: Đã có courses (Nhật 1.8M + Hàn 1.5M = 3.3M), ví còn 2.9M (< 3.3M -> không đủ đóng cả hai).
  3. Quyết định: Chỉ gọi learning.set_study_goal(minutes_per_day=35) và cập nhật lịch; KHÔNG gọi thanh toán.
  </think>
  ```
- **Thuật toán Rejection Sampling từ `01_Tool-N1/data_process/distill_data.py`:**
  - Với mỗi bước `(messages_prefix, tools, gold_step_calls)` trong tập dữ liệu:
    1. Teacher model sinh suy luận `<think>` + dự đoán `<tool_call>` ở `N=4` mẫu thử.
    2. **Bộ lọc cứng (Hard Verifier):** Chỉ chấp nhận mẫu suy luận mà `<tool_call>` Teacher tự sinh ra **khớp 100% `calls_match` và `step_gold_state_hash`** với nhãn chuẩn!
    3. Nếu cả 4 lần sinh tự do chưa khớp, cung cấp gợi ý cấu trúc ẩn để sinh lời giải thích ngược (rationalization) nhưng kiểm tra nghiêm ngặt **không chứa từ khóa lộ nhãn giả tạo** ("vì đáp án là...") và độ dài `<= 95` tokens.

#### Task 2.2: Mở rộng `+3,500` Quỹ đạo Đa lượt Kiểm chứng `SafeEnv` theo `APIGen-MT` (`src/data_gen/apigen_mt_synthesizer.py`)
- **Files:**
  - Create: `src/data_gen/apigen_mt_synthesizer.py`, `src/data_gen/trajectory_critic.py`
- **Quy trình Reverse-Blueprint 4 bước (tuân thủ tuyệt đối Red Line — KHÔNG ĐỌC `state.json`):**
  1. **Legal Sandbox Exploration:** Khởi tạo `SafeEnv('train', base_sample_id, 'data')` từ một trong `1,246` sandbox của `train_only`. Chỉ gọi các hàm `read` công khai (`list_*`, `get_*`, `search`) thông qua `env.call(...)` để thu thập các thực thể có thật hợp pháp trong sandbox đó (hóa đơn, thiết bị, khóa học, vé, danh bạ, giá xăng, kết quả xổ số...).
  2. **Targeted Blueprint Sampling (tập trung lấp đầy điểm yếu):**
     - **50% (`~1,750` mẫu):** Nhắm thẳng vào **25 ứng dụng đuôi dài `< 35` mẫu** (`fuel_price`, `tv`, `lottery`, `calculator`, `shopping_list`, `exchange_rate`, `podcast_radio`, `library`, `home_energy`, `fitness`, `news`, `household_chores`, `bike_rental`, `medication`...), đảm bảo mỗi ứng dụng có ít nhất **`80–120` quỹ đạo huấn luyện đa dạng**!
     - **30% (`~1,050` mẫu):** Kịch bản **Đa ứng dụng có điều kiện & Tự sửa ý giữa câu** (*"nếu trời mưa thì..., không mưa thì..."*, *"tắt đèn phòng khách, à quên mở rèm thôi"*, cộng trừ số tiền/thời gian).
     - **20% (`~700` mẫu):** Kịch bản **Ranh giới `Ask` (trùng tên danh bạ, thiếu tham số bắt buộc) & `OOS` bẫy sát nghĩa** (ứng dụng liên quan bị gỡ khỏi `installed_apps`).
  3. **Spoken Vietnamese + Phonetic ASR Rendering:** Biến Blueprint thành câu nói tự nhiên đa phương ngữ Bắc–Trung–Nam (`"rứa"`, `"chừ"`, `"giùm ảnh"`, `"mạ dặn"`) kèm nhiễu âm vị kiểu `PhoWhisper` và lịch sử hội thoại `history` nhiều lượt.
  4. **Dual Verification (`SafeEnv` Execution + Semantic Critic theo `06_xLAM_APIGen-MT`):**
     - Thực thi chuỗi lệnh chuẩn của Blueprint trong một bản `SafeEnv` sạch: mọi lệnh phải pass `env.validate` và `env.call` không lỗi, trích xuất `gold_state_hash = env.get_state_hash()`.
     - Chạy `trajectory_critic.py` kiểm tra tính nhất quán chặt chẽ: (a) không thừa tham số optional nào không được nhắc trong câu nói; (b) không thiếu lệnh `read` lấy ID `x-ref`; (c) nếu mơ hồ phải gọi đúng `sys.ask_missing_info`. Loại bỏ mọi mẫu điểm thẩm định `< 1.0`.

#### Task 2.3: Bổ sung Quỹ đạo Tự sửa lỗi Trong Vòng lặp & Sửa `Hammer` Masking/Shuffling (`src/data_gen/recovery_augmenter.py`, `scripts/build_lfm3_data.py`)
- **Files:**
  - Create: `src/data_gen/recovery_augmenter.py`
  - Modify: `scripts/build_lfm3_data.py`
  - Test: `tests/test_hammer_mask_consistency.py`
- **3 mẫu Tự phục hồi (Error-Recovery) bơm vào 15% dữ liệu SFT:**
  1. *Guardrail Repair:* Bước trước gọi nhầm `write` với ID bịa $\to$ nhận `{"ok": false, "error": "Unknown x-ref ID..."}` $\to$ bước hiện tại `<think>` nhận ra lỗi, gọi đúng hàm `read` để lấy ID thật rồi mới `write`.
  2. *Empty Search Retry:* Bước trước `contacts.search(query="Chị Hương Phòng Kế Toán")` trả về `[]` $\to$ bước hiện tại `<think>` rút gọn truy vấn thành `contacts.search(query="Hương")`.
  3. *Homonym Disambiguation:* `contacts.search(query="Hùng")` trả về 2 liên hệ `Nguyễn Văn Hùng` và `Trần Quốc Hùng` mà `history`/`user_profile` không phân biệt được $\to$ gọi ngay `sys.ask_missing_info(tool_name="phone.call", params=["contact_id"])` thay vì chọn bừa người đầu tiên!
- **Sửa triệt để `apply_hammer_masking` (`build_lfm3_data.py`):**
  - Luôn chạy `rng.shuffle(tools)` trên **100% bản ghi huấn luyện** (cả bản gốc lẫn bản mask) để xóa sạch thiên kiến thứ tự tool.
  - Trên `30%` bản sao bật Hammer Masking (`sys.ask_missing_info` giữ nguyên tên hệ thống, nhưng tham số `tool_name` bên trong nó cũng được đổi sang tên hàm mask tương ứng!), thay đồng bộ tên hàm/tham số trong `<tools>`, trong nhãn `[x-ref: ...]`, và dùng bản `SYSTEM_PROMPT_SCHEMA_AGNOSTIC` không nhắc cứng tên hàm cụ thể.

---

### PHA 3 — Stage 1: Curriculum SFT Dung lượng Cao cho `LiquidAI/LFM3-1.7B`
**Mục tiêu:** Đưa `LiquidAI/LFM3-1.7B` làm chủ hoàn toàn cú pháp `<think>...<tool_call>`, 147 tool schemas, đồng tham chiếu đa lượt và 25 ứng dụng hiếm.

#### Task 3.1: Đối chứng Khoa học Full Fine-Tuning Phân tầng LR vs. Rank-64 10-Module DoRA (`src/agent/lfm3_sft.py`, `configs/lfm3_sft_max.yaml`)
- **Files:**
  - Modify: `src/agent/lfm3_sft.py`
  - Create: `configs/lfm3_sft_max.yaml`, `configs/lfm3_sft_fullft.yaml`
- **Vì sao cần chạy song song 2 nhánh trên Fold-0 (`val_internal`)?**
  - Với kích thước nhỏ gọn `1.73B` (`~3.46 GB` ở `bf16`), **Full Fine-Tuning** với `AdamW` 8-bit / ZeRO-2 chạy cực nhanh trên GPU 24GB–40GB và không bị giới hạn hạng ma trận khi phải học đồng thời 46 ứng dụng + tiếng Việt khẩu ngữ; tuy nhiên các lớp tích chập ngắn **`LivConvolution` (`in_proj`, `out_proj`, `conv`)** trong kiến trúc Tri-Hybrid của `LFM3` nhảy cảm với learning rate lớn hơn các lớp `GQA` / `SwiGLU`.
  - Do đó chúng ta cài đặt cả 2 chế độ trong `src/agent/lfm3_sft.py` và chọn bản thắng trên Fold-0:
    1. **Chế độ A — Differential-LR Full Fine-Tuning:**
       - Nhóm tham số `GQA + SwiGLU` (`q/k/v/o/gate_proj`, `w1/w2/w3`): `lr = 2.0e-5`.
       - Nhóm tham số `LivConvolution + Embeddings` (`in_proj`, `out_proj`, `conv`, `embed_tokens`, `lm_head`): `lr = 5.0e-6` (`0.25x`), `weight_decay = 0.01`, `neftune_noise_alpha = 5.0`.
    2. **Chế độ B — High-Rank DoRA (`use_dora=True`, `r=64`, `lora_alpha=128`, `lora_dropout=0.05`):**
       - Phủ đủ 10 module chuẩn LiquidAI (`q_proj, k_proj, v_proj, o_proj, gate_proj, in_proj, out_proj, w1, w2, w3`), `lr = 1.5e-4`, cosine warmup `5%`.
- **Chiến lược Curriculum 2 Giai đoạn trong SFT:**
  - **Epoch 1–2:** Học trên tập kết hợp `train_only` (cả transcript sạch lẫn OOF `PhoWhisper` transcript + `+3,500` quỹ đạo `APIGen-MT` + `30%` Hammer Masking + `15%` Recovery), `max_length = 12288`.
  - **Epoch 3 (Annealing về Phân phối Thi Thật):** Chỉ học trên các quỹ đạo chuẩn mang phân phối OOF `PhoWhisper` + `AURA` phonetic hint (tắt Hammer Masking ở nửa cuối epoch 3 để phân phối tên hàm hội tụ sắc nét 100%).

---

### PHA 4 — Stage 2: On-Policy Multi-Negative `When2Call` + Boundary DPO (`Sigmoid + SFT`)
**Mục tiêu:** Khắc sâu ranh giới quyết định `Act` vs `Ask` vs `OOS` và diệt tận gốc thói quen tự bịa thêm tham số `optional` làm hỏng `set(p) == set(g)` & `state_hash`.

#### Task 4.1: Khai phá Phản ví dụ On-Policy 5 Lớp từ Checkpoint Stage-1 (`scripts/mine_onpolicy_dpo.py`)
- **Files:**
  - Create: `scripts/mine_onpolicy_dpo.py`
  - Modify: `src/agent/lfm3_dpo.py`, `configs/lfm3_dpo_max.yaml`
- **Quy trình Khai phá On-Policy (theo `lfm3_dpo_with_trl.ipynb` + `08_When2Call`):**
  - Dùng chính checkpoint `LFM3-1.7B-SFT` sinh `1` mẫu greedy (`T=0.0`) + `7` mẫu ngẫu nhiên (`T=0.8, top_p=0.95`) cho mọi bước trong tập `train_only` + tập ranh giới tổng hợp.
  - Chấm từng mẫu sinh ra bằng `SafeEnv` + `lfm3_scorer.py`. Mọi mẫu sinh ra có `step_score < 1.0` hoặc làm sai `step_gold_state_hash` được ghép cặp trực tiếp với `chosen` (có `<think>` chuẩn) tạo thành **Hard On-Policy Preference Pairs**!
  - Đối với các bước mà cả 8 mẫu đều đúng (nhưng vẫn cần gia cố ranh giới), bổ sung phản ví dụ bán tổng hợp có cấu trúc `<think>` hợp lý bề mặt thuộc **đúng 5 lớp lỗi chí mạng**:
    1. **`missed_ask_guess`:** Thiếu thông tin / trùng 2 tên danh bạ nhưng tự đoán ID hoặc số tiền để gọi tool.
    2. **`plain_text_ask` (`When2Call` Negative):** Nhận ra thiếu thông tin trong `<think>` nhưng lại trả lời câu hỏi tiếng Việt dạng text thường thay vì gọi `<tool_call>sys.ask_missing_info(...)`.
    3. **`false_ask`:** Đã có đủ thông tin trong `user_profile` (`home_address`) hoặc `history` lượt trước nhưng lại gọi nhầm `sys.ask_missing_info`.
    4. **`missed_oos`:** Ứng dụng không được cài đặt nhưng cố ép sang công cụ gần giống (ví dụ đòi đặt vé máy bay khi chỉ có `ride_hailing` / `maps`).
    5. **`optional_arg_overfill`:** Gọi đúng tên tool `write`/`read` nhưng điền thừa tham số tùy chọn (`note`, `repeat`, `priority`, `color`, `location`, `mode`) không có căn cứ trong lời nói!
- **Cấu hình Huấn luyện Chuẩn Tác giả LiquidAI (`lfm3_dpo_with_trl.ipynb:330-345`):**
  ```python
  DPOConfig(
      loss_type=["sigmoid", "sft"],
      loss_weights=[1.0, 1.0],   # Sửa từ [1.0, 0.5] về chuẩn [1.0, 1.0] giữ vững logprob sinh XML
      beta=0.1,
      learning_rate=5e-6,        # (hoặc 2e-5 với DoRA/LoRA)
      num_train_epochs=2,
      max_length=12288,
      max_prompt_length=11520,
      bf16=True,
  )
  ```

---

### PHA 5 — Stage 3: Học Tăng cường Có Kiểm chứng Hai Tầng (`ToolRL` Dense Step GRPO $\longrightarrow$ Online Multi-Turn `SafeEnv` `GiGPO`)
**Mục tiêu:** Tối ưu hóa trực tiếp mô hình trên hàm mục tiêu thực thi thật của cuộc thi (`F1 * I[state_hash_match]`), giải quyết triệt để bài toán phân bổ tín dụng qua nhiều lượt `read -> write`!

#### Task 5.1: Stage 3A — Dynamic-Filtered Step GRPO với Hàm thưởng `ToolRL` + `step_gold_state_hash` (`src/agent/lfm3_grpo.py`, `src/agent/lfm3_reward.py`)
- **Files:**
  - Modify: `src/agent/lfm3_reward.py`, `src/agent/lfm3_grpo.py`
  - Create: `scripts/filter_grpo_dynamic.py`, `configs/lfm3_grpo_stage3a.yaml`
- **Bộ lọc Động theo Năng lực (`07_LoopTool/dataloop/filter_grpo_sample.py`):**
  - Trước khi chạy GRPO, probe `pass@16` (`T=0.9`) bằng checkpoint Stage-2 DPO: **chỉ giữ lại các bước có $0 < \text{pass@16} < 1.0$** (cộng thêm `2x` oversample các ca `ask`/`oos` từng mắc lỗi > 0 lần). Việc loại bỏ các bước có phương sai phần thưởng bằng `0` giúp tăng gấp `3x` hiệu suất gradient của GRPO!
- **Công thức Hàm thưởng Bước Mật độ Cao (`ToolRL` + Cổng `SafeEnv` Hash):**
  Với mỗi completion dự đoán $\hat{y}$ ở bước $t$:
  $$R_{\text{step}}(\hat{y}) = R_{\text{format}}(\hat{y}) + R_{\text{think}}(\hat{y}) + R_{\text{match}}(\hat{y}, y^*) + R_{\text{env\_hash}}(\hat{y})$$
  Trong đó:
  - $R_{\text{format}} \in \{-1.5, +0.2\}$: `parse_ok == False` bị phạt `-1.5`; đúng cấu trúc `<think>...</think>` + XML `<tool_call>` thưởng `+0.2`.
  - $R_{\text{think}} \in [-0.3, +0.1]$: thưởng `+0.1` nếu độ dài `<think>` nằm trong vùng vàng `[25, 95]` tokens; phạt tuyến tính nếu rỗng (`0` token trên ca đa bước) hoặc lan man `> 120` tokens (bảo vệ latency!).
  - $R_{\text{match}} \in [0.0, +2.0]$: Điểm ghép cặp lưỡng phân liên tục theo `ToolRL` (`04_ToolRL/verl/utils/reward_score/rlla.py`): $0.4 \cdot \text{Jaccard}_{\text{name}} + 0.6 \cdot \mathbb{I}[\text{keys\_exact}] + 1.0 \cdot \text{ParamValueScore}$ (khớp tuyệt đối quy tắc `fuzzy` vs `exact` của `metrics.py`).
  - $R_{\text{env\_hash}} \in \{-1.5, 0.0, +1.5\}$: Thực thi `prefix_calls + pred_calls` trong `SafeEnv`:
    - Nếu lệnh gây exception trong `SafeEnv` $\implies -1.5$.
    - Nếu là bước `write` / `ask` / `oos` / `final_stop` mà `env.get_state_hash() == step_gold_state_hash` $\implies +1.5$; ngược lại nếu lệch `state_hash` $\implies -1.0$!

#### Task 5.2: Stage 3B — Online Multi-Turn Stateful `GiGPO` Trực tiếp Trong `SafeEnv` (`src/agent/lfm3_gigpo.py`)
- **Files:**
  - Create: `src/agent/lfm3_gigpo.py`, `configs/lfm3_gigpo_stage3b.yaml`
  - Test: `tests/test_gigpo_anchor_advantage.py`
- **Tại sao Stage 3B (`GiGPO`, NeurIPS 2025) tạo ra bước nhảy vọt lớn nhất về độ chính xác?**
  - Step-wise GRPO ở Stage 3A vẫn conditioned trên `gold prefix_calls`. Khi chạy thật, mô hình có thể chọn thứ tự gọi `read` khác (ví dụ gọi 2 lệnh `read` tuần tự thay vì song song, hoặc thử lại sau 1 truy vấn chưa đủ). Chỉ có **Online Multi-Turn Rollout từ đầu đến cuối trong `SafeEnv`** mới tối ưu trực tiếp `score_sample` cuối phiên!
  - Tuy nhiên, nếu dùng GRPO cấp quỹ đạo thông thường, mọi bước trong cùng 1 phiên nhận chung 1 lợi thế $A^E_i$, khiến mô hình không biết bước `read` thứ 1 đúng nhưng bước `write` thứ 3 mới là thủ phạm làm sập `state_hash`!
- **Thuật toán `GiGPO` Tùy biến cho `SafeEnv` (chuyển thể trực tiếp từ `05_GiGPO_verl-agent/gigpo/core_gigpo.py:138-195`):**
  1. **Multi-Turn Group Rollout ($G = 8$ quỹ đạo / mẫu khó):**
     Với mỗi mẫu đầu vào $x$, chạy song song $G = 8$ phiên `SafeEnv` độc lập (tối đa $T_{\max} = 6$ lượt ReAct/phiên, `temperature = 0.85`, bật `vLLM` engine tăng tốc rollout).
  2. **Phần thưởng Cuối phiên Chuẩn Cuộc thi ($R^E_i$):**
     Khi phiên thứ $i$ kết thúc, chấm trực tiếp bằng `src/eval/metrics.py:score_sample`:
     $$R^E_i = 2.0 \cdot \underbrace{\left(\text{F1}_i \cdot \mathbb{I}[h_i = h_{\text{gold}}]\right)}_{\text{Official Final Score}} + 0.35 \cdot \text{F1}_i - 0.04 \cdot \max(0, \text{unmatched\_reads}_i)$$
     *(Thành phần $0.35 \cdot \text{F1}_i$ giữ tín hiệu dẫn đường mượt khi chưa khớp `state_hash`, còn $2.0 \cdot \text{Official Score}$ đặt trọng số áp đảo lên đích cuối cùng, kèm phạt nhẹ lệnh `read` thừa!).*
     Tính lợi thế cấp quỹ đạo (Macro Episode Advantage):
     $$A^E_i = \frac{R^E_i - \text{mean}_{j=1..G}(R^E_j)}{\text{std}_{j=1..G}(R^E_j) + \epsilon}$$
  3. **Gom Nhóm Trạng thái Neo (`AnchorState`) & Lợi thế Cấp Bước ($A^S_{i,t}$):**
     Tại đầu mỗi bước $t$ của quỹ đạo $i$, định nghĩa khóa trạng thái môi trường bất biến:
     $$\text{AnchorState}(s_{i,t}) = \big(\text{sample\_id},\; \text{env}_i.\text{get\_state\_hash}(),\; \text{frozenset}(\text{executed\_read\_signatures}_{i,<t})\big)$$
     Mọi bước $(i, t)$ trong nhóm $G$ quỹ đạo có chung $\text{AnchorState}$ với kích thước nhóm $|G(s)| \ge 2$ được gom lại; định nghĩa giá trị hậu nghiệm của hành động $a_{i,t}$ bằng phần thưởng cuối phiên chiết khấu nhẹ $\tilde{R}_{i,t} = \gamma^{T_i - t} R^E_i$ ($\gamma = 0.95$):
     $$A^S_{i,t} = \begin{cases} \dfrac{\tilde{R}_{i,t} - \text{mean}_{(j,t') \in G(s)} \tilde{R}_{j,t'}}{\text{std}_{(j,t') \in G(s)} \tilde{R}_{j,t'} + \epsilon}, & \text{nếu } |G(s_{i,t})| \ge 2 \text{ và std} > 10^{-6} \\ 0, & \text{ngược lại} \end{cases}$$
  4. **Tổng hợp Lợi thế Hai Tầng & Asymmetric Clip-Higher (`DAPO`/`GiGPO`):**
     $$A_{i,t} = A^E_i + \omega \cdot A^S_{i,t} \quad (\omega = 1.0)$$
     Cập nhật chính sách với tỷ lệ cắt bất đối xứng **`clip_ratio_low = 0.20, clip_ratio_high = 0.28`** (`kl_coef = 0.01` so với Stage-2 DPO reference) để khuyến khích khám phá các chuỗi suy luận đúng hiếm gặp mà không làm trôi phân phối ngôn ngữ!

---

### PHA 6 — Stage 4: Vòng lặp Khép kín `LoopTool` (3 Chu kỳ Săn Điểm Mù) & Hợp nhất 5-Fold Model Soup
**Mục tiêu:** Quét sạch 5–8% lỗi cứng đầu cuối cùng bằng đồng tiến hóa Dữ liệu–Mô hình và giảm phương sai dự đoán bằng hợp nhất trọng số đa Fold.

#### Task 6.1: Ba Vòng lặp `LoopTool` Tự động (`scripts/run_looptool_cycle.py`)
- **Files:**
  - Create: `scripts/run_looptool_cycle.py` (kế thừa kiến trúc `07_LoopTool/dataloop/`)
- **Chu trình Khép kín cho mỗi Vòng $c \in \{1, 2, 3\}$:**
  1. **Capability Probing (`greedy_capability_prob.py`):** Chạy đánh giá toàn bộ Fold huấn luyện bằng cả `Greedy (T=0)` lẫn `Pass@8 (T=0.8)`. Phân loại mọi mẫu có `greedy_score < 1.0` vào 2 rổ:
     - *Rổ A (`greedy < 1.0` nhưng `max_of_8 == 1.0` — Lỗi Xếp hạng Nội tại):* Mô hình đã biết cách giải trong không gian mẫu nhưng chọn sai ở greedy $\implies$ đưa thẳng cặp `(winner_of_8, greedy_loser)` vào tập **On-Policy DPO + Stage 3B GiGPO** của chu kỳ kế tiếp!
     - *Rổ B (`max_of_8 < 1.0` — Điểm Mù Kiến thức/Lập luận):* Phân rã tự động nguyên nhân lỗi (app hiếm, công thức tính giờ, điều kiện rẽ nhánh phức tạp, lỗi đồng âm ASR) $\implies$ gọi module `APIGen-MT` sinh thêm **`5x` kịch bản tương đồng cấu trúc mới hoàn toàn** (kiểm chứng `SafeEnv` `state_hash`) đưa vào tập SFT bổ sung!
  2. **Warm-Restart Micro-Update:** Tinh chỉnh ngắn (`1 epoch` SFT trộn replay `20%` dữ liệu cũ chống quên $\to$ `1 epoch` On-Policy DPO $\to$ `150 steps` GiGPO) và kiểm tra điểm trên `val_fold` tăng đơn điệu.

#### Task 6.2: Hợp nhất Trọng số 5-Fold Model Soup (`scripts/merge_lfm3_soup.py`)
- **Files:**
  - Create: `scripts/merge_lfm3_soup.py`
- **Cơ chế tăng Acc "Miễn phí Tham số & Miễn phí Latency":**
  - Huấn luyện 5 nhánh LoRA/DoRA (hoặc tinh chỉnh từ cùng một base SFT-checkpoint chung để cùng bồn cực tiểu loss — *shared-basin fine-tuning*) trên 5 Fold phân tầng của `train_only` + 1 nhánh trên 100% `train_only`.
  - Hợp nhất trọng số tuyến tính có trọng số tối ưu (Uniform / `TIES-Merging` qua `peft`/`mergekit`) tạo thành **duy nhất 1 checkpoint `LiquidAI/LFM3-1.7B` (`1.73B` tham số, tốc độ suy luận không đổi 1 mili-giây)** nhưng mang hiệu ứng giảm phương sai tương đương Ensemble 5 mô hình (**thường tăng ổn định `+1.2% – +2.0%` `state_match` trên tập mù**)!

---

### PHA 7 — Động cơ Suy luận Siêu tốc KV-Cached + Mô hình Hậu kiểm Ranh giới (`< 0.56B`) & Đóng gói Thi Đấu
**Mục tiêu:** Cắt độ trễ xuống `~2.3s/mẫu` trên NVIDIA A30 và dựng lớp khiên kiểm chứng xác suất thứ hai trước mọi thao tác thay đổi trạng thái `SafeEnv`.

#### Task 7.1: Tích hợp Prefix Caching cho Đa bước ReAct (`src/agent/llm_client.py`)
- **Files:**
  - Modify: `src/agent/llm_client.py`, `configs/e2e_phowhisper_lfm3_17b.yaml`
  - Test: `tests/test_prefix_cache_determinism.py`
- **Thiết kế chi tiết:**
  - Cài đặt song song 2 backend tất định (`temperature=0.0`):
    1. **`VLLMBackend` (Ưu tiên số 1 cho A30 Docker & Remote Batch):** Bật `enable_prefix_caching=True, dtype="bfloat16", gpu_memory_utilization=0.55, max_model_len=16384`. Giữa các bước 1 $\to$ 2 $\to$ 3 của cùng 1 mẫu, `~6,200` token đầu (`system + <tools> + history + user`) được đọc trực tiếp từ Prefix Cache trong `< 15ms`!
    2. **`TransformersCachedBackend` (Fallback tương thích mọi môi trường như T4 `fp16`):** Lưu đối tượng `past_key_values` / trạng thái cache của `LFM3` trong phiên `run_sample` để chỉ forward phần token mới (`assistant_turn + tool_observation`) ở lượt tiếp theo.
  - Kiểm thử bắt buộc: đầu ra token IDs của chế độ bật cache và tắt cache phải khớp tuyệt đối trên chuỗi kiểm thử chuẩn.

#### Task 7.2: Huấn luyện & Tích hợp Cross-Encoder Boundary Verifier (`src/agent/boundary_verifier.py`)
- **Files:**
  - Create: `scripts/train_boundary_verifier.py`, `src/agent/boundary_verifier.py`
  - Modify: `src/agent/react_loop.py`
  - Test: `tests/test_boundary_verifier.py`
- **Khai thác hợp pháp ngân sách tham số còn trống (`1.54B + 1.73B + 0.56B = 3.83B < 6.00B`):**
  - Dùng mô hình ngôn ngữ đa ngữ nhỏ **`BAAI/bge-reranker-v2-m3` (`0.56B`)** (hoặc `microsoft/mdeberta-v3-base` `0.28B`) huấn luyện Out-Of-Fold trên toàn bộ các ca đúng/sai thu thập từ quá trình rollout của `LFM3-1.7B` trên `train_only`.
  - **2 nhiệm vụ chuyên biệt cực gọn của Verifier (chỉ tốn `~50ms` trên GPU):**
    1. **Boundary Gate (`Act` vs `Ask` vs `OOS`):** Khi `LFM3-1.7B` chuẩn bị dừng rỗng `[]` ở bước 1, hoặc chuẩn bị gọi lệnh `write` khi câu nói ngắn/có đại từ mơ hồ, Verifier chấm điểm nhất quán `(Context + Utterance + Observations, Proposed Action)`. Nếu xác suất `should_ask > 0.85` (ví dụ có 2 người trùng tên trong observation vừa đọc mà `LFM3` vẫn lỡ tay chọn 1 người), Verifier chuyển hướng an toàn sang ứng viên thứ 2 từ **Deterministic Beam-2** của `LFM3`!
    2. **Optional-Argument Pruning Verifier:** Khi một lệnh dự đoán chứa các tham số `optional` nhạy cảm (như `note`, `label`, `priority`, `mode`, `repeat`, `location`), Verifier chấm cặp `(Utterance + History + Schema_Param_Desc, Param_Key=Param_Val)`: chỉ giữ tham số optional khi có bằng chứng ngữ nghĩa thực sự trong hội thoại ($p_{\text{grounded}} \ge 0.5$), ngược lại loại bỏ chính xác mà **không bao giờ dùng regex mù gây hỏng nhãn vàng như sanitizer cũ**!

#### Task 7.3: Kiểm định Mù Một Lần Duy nhất trên `dev_v1` (`299` mẫu) & Đóng gói Docker Offline (`scripts/verify_submission.py`)
- **Quy tắc Kỷ luật Thép:**
  - Mọi tinh chỉnh prompt, ngưỡng Verifier, số vòng `LoopTool`, lựa chọn Full-FT vs DoRA đều chỉ được quyết định trên **5-Fold CV của `train_only` (`1,246` mẫu)**.
  - Chỉ khi toàn bộ 8 pha hoàn tất và hệ thống vượt mọi bài test (`pytest tests/ -q` xanh 100%, tổng tham số kiểm đếm tự động `3.83B < 6.00B`, latency p99 `< 12s << 30s`, offline mode `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`), mới chạy đánh giá chốt hạ trên `dev_v1` để xác nhận điểm tổng quát hóa thực tế trước khi xuất container nộp bài!

---

## 6. Bảng Dự phóng Độ chính xác theo Từng Pha & Quản trị Rủi ro

### 6.1. Lộ trình Tăng trưởng Điểm số Kỳ vọng (`Official Final Score` = `F1 * I[state_match]`)

| Giai đoạn | Hạng mục Nâng cấp Cốt lõi | Kỳ vọng `state_match` | Kỳ vọng `Tool-Call F1` | Kỳ vọng **`Official Score`** | Động lực Tăng điểm Chính |
|---|---|---|---|---|---|
| **Baseline hiện tại** | Zero-shot `LFM3-1.7B` + `PhoWhisper` greedy (8-mẫu smoke / nền gốc) | `~68% – 71%` | `~74% – 76%` | **`~66% – 71%`** | Còn đủ 15 lỗi tồn đọng & sanitizer phá nhãn vàng. |
| **Sau Pha 0 + Pha 1** | Xóa 15 lỗi gốc + Zero-Damage Sanitizer + Collection `x-ref` + `PhoWhisper` Domain LoRA (`4,042` wav) + `AURA` 5-beam diff | `~74% – 76%` | `~79% – 81%` | **`~73% – 75%`** (`+4%..+6%`) | Cứu ngay `~5%` mẫu bị hỏng bởi sanitizer & batch ASR context drop; diệt lỗi đồng âm. |
| **Sau Pha 2 + Pha 3** | `+3,500` quỹ đạo `APIGen-MT` (lấp đầy 25 app hiếm) + `Tool-N1` 4-slot `<think>` + Recovery + Stage-1 Curriculum SFT | `~82% – 84%` | `~86% – 88%` | **`~81% – 83%`** (`+8%`) | Mô hình `1.73B` thành thạo 147 tools, tính toán được giờ/tiền trong `<think>` và tự sửa lỗi. |
| **Sau Pha 4 (DPO)** | Stage-2 On-Policy 5-Class `When2Call` + Optional-Arg DPO (`sigmoid+sft`, `w=[1,1]`) | `~85% – 87%` | `~89% – 90%` | **`~84% – 86%`** (`+3%`) | Xóa lỗi đoán mò khi thiếu thông tin, lỗi hỏi bằng text thường & lỗi bịa tham số optional. |
| **Sau Pha 5 (`GiGPO`)** | Stage-3A Dynamic Step GRPO $\to$ Stage-3B Multi-Turn Live `SafeEnv` `GiGPO` (`AnchorState = state_hash`) | `~88% – 90%` | `~91% – 92%` | **`~87% – 89%`** (`+3%`) | Tối ưu hóa trực tiếp vách đá `state_hash` và giảm lệnh `read` thừa trên chuỗi đa ứng dụng. |
| **Sau Pha 6 + Pha 7** | 3 chu kỳ `LoopTool` + 5-Fold Weight Soup + Prefix Cache + Cross-Encoder Boundary Verifier (`0.56B`) | **`~90% – 93%`** | **`~93% – 95%`** | **`~89% – 92%+`** (`+2%..+3%`) | Quét sạch điểm mù đuôi dài, giảm phương sai dự đoán & khóa chốt an toàn trước mọi lệnh `write`. |

### 6.2. Ma trận Quản trị Rủi ro Kỹ thuật

| Rủi ro Tiềm ẩn | Dấu hiệu Nhận biết | Biện pháp Phòng ngừa & Chốt chặn Đã Thiết kế |
|---|---|---|
| **Tràn độ trễ do bật `<think>` ở mẫu 5–6 bước** | Mẫu phức tạp vượt `15s` trên T4/A30 | (1) Ép cấu trúc 4-Slot Micro-`<think>` `<= 85` tokens ngay từ khâu chưng cất + phạt độ dài trong `lfm3_reward.py`; (2) Bật `vLLM` Prefix Caching giảm `6x` thời gian prefill đa bước. |
| **Suy thoái cú pháp XML `<tool_call>` sau DPO/RL** | Tăng tỷ lệ `parse_ok == False` | Dùng `loss_type=["sigmoid", "sft"], loss_weights=[1.0, 1.0]` trong DPO (chuẩn tác giả LiquidAI) + phạt `-1.5` cho lỗi format và giữ `kl_coef = 0.01` trong `GiGPO`. |
| **Nhiễu từ dữ liệu tổng hợp `APIGen-MT`** | Mô hình học thói quen gọi tool khác chuẩn gốc | Mọi quỹ đạo tổng hợp phải vượt qua đồng thời `SafeEnv` execution check + `trajectory_critic` kiểm tra `0` tham số optional dư thừa, và ở nửa cuối Epoch 3 SFT anneal 100% về dữ liệu gốc `train_only`. |
| **Overfit thanh điệu khi fine-tune `PhoWhisper`** | Tăng lỗi chèn từ trên giọng lạ | Dùng `r=8`, SpecAugment mạnh, toàn bộ `4,042` câu audio và đánh giá chéo 5-Fold OOF nghiêm ngặt; nếu CER không giảm trên cả 5 fold thì tự động giữ trọng số gốc + `5-beam` context biasing. |
| **Vi phạm bất kỳ Red Line nào của BTC** | Lỗi chấm tự động = `0` điểm | Bộ test tự động `tests/test_competition_compliance.py` kiểm tra trước mọi lần đóng gói: không import/đọc `_environment`, tổng tham số `< 6.0B`, LLM `< 5.0B`, `context <= 16,384`, `temperature == 0.0`, chạy offline hoàn toàn. |

---

## 7. Thứ tự Khởi chạy Tức thì (Immediate Execution Queue)

Ngay khi phê duyệt kế hoạch này, quy trình triển khai sẽ bắt đầu tuần tự theo kỷ luật TDD & Remote GPU:
1. **Sprint 1 (Local Code & Unit Tests — 100% CPU nhẹ):** Triển khai toàn bộ **Pha 0** (`make_folds.py`, xóa sạch lỗi tự hủy điểm của `sanitize_tool_call` đạt `0/2854` gold corruption, sửa `collection-exact x-ref`, sửa double BOS/EOS, thêm `step_gold_state_hash`, sửa `whisper_engine.py` batch context bug + `AURA` N-best diff) và nâng bộ unit test lên **`100+ passed`**.
2. **Sprint 2 (Remote GPU Batch 1 — Acoustic Layer):** Huấn luyện 5-Fold OOF + Full `PhoWhisper-Large` LoRA trên `4,042` audio của `train_only`, xuất `train_only_oof_nbest.jsonl` và đẩy adapter lên Hugging Face Hub.
3. **Sprint 3 (Remote GPU Batch 2 — Data Factory):** Chạy `distill_think.py` (`Tool-N1` 4-slot `<think>`) + `apigen_mt_synthesizer.py` (`+3,500` quỹ đạo ưu tiên 25 app hiếm, kiểm chứng `SafeEnv`) + `recovery_augmenter.py`, đóng gói tập dữ liệu v3 đẩy lên Hugging Face Hub.
4. **Sprint 4 (Remote GPU Batch 3 — Post-Training `LFM3-1.7B`):** Chạy chuỗi **Stage 1 Curriculum SFT (Full-FT vs Rank-64 DoRA)** $\to$ **Stage 2 On-Policy `When2Call` DPO+SFT** $\to$ **Stage 3A Dynamic Step GRPO** $\to$ **Stage 3B Multi-Turn Live `SafeEnv` `GiGPO`**.
5. **Sprint 5 (Remote GPU Batch 4 — `LoopTool` + Soup + Verifier):** Chạy 3 chu kỳ `LoopTool`, ghép **5-Fold Model Soup**, huấn luyện **Boundary Cross-Encoder Verifier (`0.56B`)**, bật Prefix Caching và thực hiện kiểm định cuối cùng.
