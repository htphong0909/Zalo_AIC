# Kiểm Toán Lỗi Sau DPO V1, Đối Chiếu Nghiên Cứu DPO 2025–2026 & Lộ Trình Tối Ưu MiniCPM5-2B

**Ngày lập:** 2026-11-13  
**Đối tượng phân tích:** `exp_20261112_minicpm5_dpo_apex` (`exp_branchA_dpo_v1`) so với baseline `exp_20261111_minicpm5_sft_v2_g4` (`exp_branchA_sft_v2`)  
**Ràng buộc tuân thủ:** Không truy cập `<split>_environment/` hay `state.json`; không trích luật thủ công từ `configs/splits/dev_v1.json`; toàn bộ số liệu được kiểm chứng trực tiếp từ artifact local, Hugging Face checkpoints và log training (`task-261.log`).

---

## 1. Kiểm Toán Toàn Diện Lỗi Tập Train & Kết Quả Sau DPO V1

### 1.1. Xác Minh Phạm Vi Đánh Giá: Chưa Từng Chạy Eval Post-DPO Trên `train_only`

Qua kiểm tra toàn bộ kết quả đánh giá trong `experiments/`, `data/train/train.jsonl` (`1,545` mẫu) được chia thành **`1,246` mẫu `train_only`** và **`299` mẫu `dev_ids`** (`configs/splits/dev_v1.json`, tất cả đều mang tiền tố ID `train_XXXX`).

- **Chưa có bất kỳ lượt inference/evaluation nào sau DPO (hoặc SFT V2) chạy trên `train_only` (`1,246` mẫu):**
  - Toàn bộ 3 đợt đánh giá MiniCPM5 (`exp_20261110_minicpm5_sft_comparison_l4`, `exp_20261111_minicpm5_sft_v2_g4`, `exp_20261112_minicpm5_dpo_apex`) đều cho `in_dev = 299, in_train_only = 0`.
  - File `experiments/exp_20261111_minicpm5_sft_v2_g4/train_sample_10_preds.jsonl` hiện trống (`0` bytes).
- **Ý nghĩa vận hành:** Hệ thống hiện chưa có ma trận lỗi on-policy trên `1,246` mẫu `train_only` — đồng nghĩa module **LoopTool Greedy Capability Probing (GCP)** bị bỏ trống trong DPO V1 (`looptool_mined = 0`).
- **Hồ sơ cấu trúc chuẩn của `1,246` mẫu `train_only`:**
  - Phân bố nhãn: `act = 982`, `ask = 174`, `oos = 90`.
  - Độ phức tạp: `821` mẫu có `history` (`65.9%`), `859` mẫu multi-step `>1` tool call (`68.9%`).
  - Top 12 app phổ biến: `smart_home (224)`, `contacts (193)`, `bill_payment (175)`, `sys (174)`, `calendar (169)`, `chat_ott (169)`, `banking (163)`, `reminders (109)`, `food_delivery (89)`, `alarm (72)`, `ewallet (70)`, `sms (67)`.

---

### 1.2. Lỗi Trọng Số & Cơ Chế Trainer? — Đã Xác Minh Sạch

Kiểm tra trực tiếp trọng số trên Hugging Face (`htphong0909/zalo-aic-sft-checkpoints`) và `peft==0.21.2` + `transformers==5.22.1`:
- Số tensor keys: `sft_v2 = 588`, `dpo_ref = 588`, `dpo_w = 588`.
- **Reference model khớp tuyệt đối SFT V2:** `max|sft_v2 - dpo_ref| = 0.000000e+00`.
- **LoRA `merge_and_unload()` chuẩn:** `peft==0.21.2` chỉ merge `active_adapters` (`"default"`), không cộng nhầm adapter `"ref"`.
- **Biên độ dịch chuyển trọng số vừa phải:** L2 delta `min=0.00512, median=0.01893, max=0.03804`; relative delta `||dpo - sft|| / ||sft||` đạt `median = 3.23%`, `max = 5.93%`.

$\rightarrow$ **Kết luận kỹ thuật:** Lỗi **không nằm ở code load/merge adapter**, mà xuất phát từ **chất lượng cặp dữ liệu DPO + lệch phân phối template/hệ thống + động lực học tối ưu offline DPO**.

---

### 1.3. Lỗi Trong Tập Dữ Liệu Train DPO (`data/dpo/minicpm5_dpo_pairs.jsonl.gz` — 982 cặp)

Toàn bộ `982` cặp được xây dựng sạch từ `train_only` (`0` rò rỉ dev), phân bố nguồn gồm: `when2call (350)`, `tool_n1_horizon (250)`, `looptool_mutated (232)`, `toolsandbox_anchor (150)` (`looptool_mined = 0`). Tuy nhiên, kiểm toán định lượng phát hiện **hai nhóm lỗi nghiêm trọng**:

#### A. 4 Lỗi Lệch Prompt / Chat Template Toàn Cục So Với Suy Luận Thực Tế
Khi inference (`configs/oracle_minicpm5_2b.yaml` + `adapter/chat_template.jinja` + `src/agent/prompts.py`), model nhận `build_system_prompt(sample)` (chứa `## Ngữ cảnh hiện tại`, `user_profile`, tool catalog), lịch sử hội thoại `sample.history`, quan sát tool bọc trong `<|im_start|>user\n<tool_response>...</tool_response>`, và prompt kết thúc bằng `<|im_start|>assistant\n<think>\n` (`enable_thinking=true`). Ngược lại, trong `data/dpo/minicpm5_dpo_pairs.jsonl.gz` (`scripts/build_minicpm5_dpo_pairs.py`):

| Lỗi toàn cục | Số cặp dính lỗi / 982 | Tác động lên policy |
|---|---:|---|
| **Mất `build_system_prompt(sample)`** (dùng prompt trần `"Bạn là trợ lý ảo LifeMate."`, không tool schema, không `current_time`, không `user_profile`) | **982 / 982 (100.0%)** | Buộc model học preference trên prompt mù ngữ cảnh và mù schema tool. |
| **Cắt bỏ `sample.history`** (mẫu gốc có `history` nhưng prompt DPO có `n_user_turns < 2`) | **663 / 982 (67.5%)** | Phá hủy ngữ cảnh tham chiếu chéo giữa các lượt (`when2call`: 241/350, `tool_n1`: 165/250, `looptool`: 155/232, `anchor`: 102/150). |
| **Sai role tool trong multi-turn** (`<|im_start|>tool\n` thô thay vì `<|im_start|>user\n<tool_response>\n...`) | **400 / 400 multi-turn (100.0%)** | Toàn bộ `250` mẫu `tool_n1` + `150` mẫu `anchor` lệch format tokenizer lúc inference. |
| **Lệch vị trí `<think>` & rò rỉ văn phong đối xứng** | **982 / 982 (100.0%)** | Prompt dừng ở `<|im_start|>assistant\n`, không có `<think>\n` (`ends_with_<think>=0`). Hơn nữa `chosen` và `rejected` dùng hai khuôn tổng hợp tách biệt cứng có header tiếng Anh (`"1. Decision: Gọi đúng tool."` vs `"1. Decision: Quyết định sai lệch."`), khiến loss phân tách bằng từ khóa `<think>` giả tạo ngay Step 10 (`rewards/accuracies = 1.0`). |

*(Thống kê độ dài token bằng tokenizer MiniCPM5: prompt `min=1586, p50=4966, p90=6945, max=8632`, `>4096 = 716`, `>8192 = 17`; `chosen p50=105`; `rejected p50=89`.)*

#### B. Lỗi Đầu Độc Nhãn Theo Từng Nguồn Sinh (`build_minicpm5_dpo_pairs.py`)

1. **`when2call` (`350` cặp):**
   - **Nhánh A (`missing_arg_ask_vs_hallucinated_act`, 175 cặp) — 82.3% nhãn đảo ngược:** **`144 / 175` cặp có `masked_text == orig_text` (`subA_utterance_unchanged_FALSE_ASK`)**, tức câu nói của người dùng còn *đầy đủ thông tin* không hề bị xóa tham số, nhưng `chosen` lại ép gọi `sys.ask_missing_info` còn `rejected` là tool call đúng! Nguyên nhân trực tiếp: `build_minicpm5_dpo_pairs.py` đã bỏ sót check `if masked_text == orig_text: continue` vốn có ở `src/data_gen/when2call_gen.py:188`. Chỉ `30` câu bị rút gọn và `1` câu thay `"chút tiền"`.
   - **Nhánh B (`complete_act_vs_false_ask`, 175 cặp) — 61.7% dạy ảo giác ID:** **`108 / 175` cặp** có `chosen` truyền tham số `*_id` hoàn toàn không tồn tại trong `prompt` (do `history` và các bước Read trước đó bị cắt mất).
2. **`tool_n1_horizon` (`250` cặp):**
   - **`250 / 250 (100%)`** thay kết quả tool thật bằng JSON giả `{"status": "success", "executed": true}`; **`84 / 250`** nén `2–5` tool call trước đó vào một tool message giả duy nhất; **`30 / 250`** có `chosen == sys.ask_missing_info`; **`0` cặp** dạy dừng đúng lúc sau khi hoàn thành (`ACT -> Stop`) và **`0` cặp OOS**.
3. **`looptool_mutated` (`232` cặp):**
   - **`0` mẫu lỗi thực (`looptool_mined = 0`)**. **`220 / 232 (94.8%)`** dùng cùng một rejected cứng `reminders.create(content="việc cần làm", time="2026-11-12T12:00:00")`, quá dễ phân biệt bằng pattern.
   - **`147 / 232 (63.4%)`** có `chosen` gọi lẫn cả hàm `read` và `write` ngay trong Step 1 (ví dụ `music.list_tracks` + `music.play(track_id="trk_07")`) khi chưa hề nhận observation chứa `trk_07`.
4. **`toolsandbox_anchor` (`150` cặp):**
   - **`150 / 150 (100%)`** chèn JSON observation giả `{"status": "success", "results": [{"id": ...}]}` không khớp schema tool thật lên câu nói ngẫu nhiên.
   - **Lỗi nghiêm trọng nhất — `chosen` vi phạm schema catalog:** **`70 / 150 (46.7%)`** cặp có `chosen` **thiếu tham số bắt buộc (`required`)** theo catalog (`bill_payment.pay = 50`, `phone.call = 30`, `calendar.update_event = 26`, `calendar.delete_event = 24`, `sms.send = 20`). Nói cách khác, gần một nửa `chosen` trong nguồn này sẽ fail ngay `SafeEnv.validate()` — trực tiếp huấn luyện model bỏ sót tham số bắt buộc!

---

### 1.4. Lỗi Thượng Nguồn Có Sẵn Trong Dataset SFT V2 (`data/sft/minicpm5_train_sft_v2.jsonl.gz`)

Kiểm tra chéo `4,594` mẫu SFT V2 (`base_train=3,158`, `when2call=500`, `anchor=336`, `apigen=600`) cho thấy chính SFT V2 cũng mang sẵn lỗi từ `apigen_mt_gen.py`, `anchor_binding_gen.py`, và `when2call_gen.py`:
- Toàn bộ **`1,436 / 4,594` (`31.3%`)** mẫu tăng cường (`500` `when2call` + `336` `anchor` + `600` `apigen`) dùng system prompt trần `"Bạn là trợ lý ảo LifeMate."` thiếu `build_system_prompt(sample)`. (`3,158` mẫu `base_train` mới có `"## Ngữ cảnh hiện tại"`).
- **`308 / 600` (`51.3%`)** mẫu `apigen` trộn cả `read` và `write` ngay trong `Step 1` của một trajectory (hoặc nối ghép câu dùng observation giả `{"ok": True, "message": "Thành công"}` tại `src/data_gen/apigen_mt_gen.py:185`).

---

### 1.5. Phân Tích Lỗi Suy Luận Post-DPO vs SFT V2 (Trên 299 Mẫu Đã Đánh Giá)

> **Đính chính số liệu báo cáo cũ:**  
> 1. `experiments/exp_20261112_minicpm5_dpo_apex/REPORT.md` dòng 96 ghi *"258 / 299 samples (86.3%) yielded strictly identical outputs"*. Thực tế `258 / 299` là số mẫu **cùng `score`**, còn số mẫu có chuỗi **`tool_calls` giống hệt nhau chỉ là `173 / 299 (57.9%)`** — có tới **`126 / 299 (42.1%)` mẫu đã bị DPO làm thay đổi `tool_calls`**.  
> 2. `REPORT.md` dòng 77 ghi `52.52% (157/299)` nhưng `summarize_results` trong `src/eval/evaluator.py:41-45` chỉ tính `state_match_rate` trên **`278` mẫu action/ask có `tool_calls`** (`146 / 278 = 52.52%`). Tính trên toàn bộ `299` mẫu, số mẫu `state_match=True` thực tế giảm từ `166` xuống `164`.  
> 3. `reports/LLM_AGENT_EXPERIMENTS_DASHBOARD.md` dòng 14 ghi SFT V2 dùng `lr=1e-5`, thực tế `scripts/scratch/worker_sft_v2_g4.py` chạy `lr=1.5e-4`.

#### Bảng Đối Chiếu Định Lượng (`metrics.json` + `per_sample.jsonl`)

| Chỉ số | SFT V2 Baseline | DPO V1 (`exp_branchA_dpo_v1`) | Chênh lệch / Ý nghĩa |
|---|---:|---:|---|
| **Final Score** | **0.4570** | **0.4511** | **-0.0059** (Suy giảm nhẹ) |
| **State Match Rate** (`n=278` non-OOS) | 52.52% (`146/278`) | 52.52% (`146/278`) | `0.00%` (tổng 299 mẫu: `166 -> 164`) |
| **Tool-calling F1 / P / R** (`n=278`) | `59.91%` / `61.27%` / `60.24%` | `59.58%` / `60.86%` / `59.78%` | `-0.33%` / `-0.41%` / `-0.46%` |
| **Exact `score = 1.0`** | **121** | **116** | **-5 mẫu** |
| **Partial `0 < score < 1`** | 24 | 31 | `+7` |
| **Zero `score = 0.0`** | 154 | 152 | `-2` (nhưng `85` mẫu ở DPO có `f1 > 0` bị gán `0.0` vì `state_match=False`) |
| **OOS Score** (`n=21`) | **0.9048** | **0.8571** | **-0.0477** |
| **History Score** (`n=197`) | **0.4466** | **0.4391** | **-0.0075** |
| **Improved / Regressed / Changed calls** | — | — | `19` cải thiện (`act:16, ask:2, oos:1`) vs `22` tụt (`act:19, ask:1, oos:2`); **`126` mẫu đổi calls** (`more:34, fewer:20, same_len_diff:72`) |

#### Ma Trận Chuyển Dịch Quyết Định (`gold -> pred`) & Nguyên Nhân Lỗi Mức Tham Số

- **Confusion matrix (`SFT V2 -> DPO V1`):**
  - `act -> ask` (hỏi thừa khi đủ ý): **`10 -> 5`** (một cải thiện nhỏ duy nhất, kèm `act -> act: 223 -> 229`).
  - `ask -> act` (cần hỏi lại nhưng đoán bừa/gọi thẳng): không cải thiện (**`23 -> 23`**, `ask -> ask` kẹt ở **`18 / 43`**). Nguyên nhân trực tiếp từ lỗi 82.3% false-ASK đảo cực ở `when2call`.
  - `oos -> act`: tăng **`1 -> 3`** (kéo điểm OOS từ `0.9048` xuống `0.8571` vì `0` mẫu OOS trong DPO).
  - `act_over_calling_extra_tools` (gọi thừa tool ngoài gold): tăng **`13 -> 20`**.
  - Calls vào app chưa cài đặt (`uninstalled apps`): `SFT V2 = 0`, `DPO V1 = 0`. Ghi chú: `BoundaryVerifier` (`src/agent/boundary_verifier.py`) hiện vẫn **chưa từng được nối vào `react_loop.py`**.

- **Bùng nổ lỗi thiếu tham số (tác động trực tiếp từ `70/150` `chosen` thiếu `required` param của `toolsandbox_anchor`):**
  - Khi tên tool đã khớp đúng gold, phân loại sai tham số thay đổi như sau:
    - `exact_val_mismatch`: `80 -> 75`
    - `extra_arg_key`: `16 -> 17`
    - `fuzzy_val_mismatch`: `26 -> 28`
    - **`missing_arg_key`:** **`16 -> 31` (tăng +93.8%!)**
  - Top lỗi tham số trên DPO V1: thừa `contacts.search.query (6)`, sai giá trị `banking.transfer.amount (5)`, sai `contacts.search.query (5)`, thiếu `contacts.search.query (4)`, sai `bill_payment.pay.source (4)`, sai `smart_home.light_set.device_id (4)`, sai `podcast_radio.search.query (4)`, thiếu `calendar.list_events.date_from (3)` & `date_to (3)`.
  - Top lỗi fuzzy (`token_f1 <= 0.6` trong `src/eval/metrics.py:45`): **`sms.send.content (12)`**, `chat_ott.send_message.content (3)`, `email.create_draft.body (2)`, `notes.create.content (2)`, `reminders.create.content (2)`. *(Đối chiếu trên `train_only` cho thấy nội dung chuẩn của `sms.send.content` là tóm tắt siêu ngắn dạng `"mai chạy bộ dời sang 09:00 nhé"`, `"đã đóng tiền mạng rồi"`, nên bất kỳ cách diễn đạt dài dòng hoặc lệch từ khóa nào đều rớt ngưỡng `token_f1 > 0.6` dù `state_match` có thể vẫn `True`).*
  - **8 app có tỷ lệ hoàn hảo (`score=1.0`) thấp nhất dưới DPO V1:** `sms 0.0% (18/18 fail; 8 mẫu đạt state_match nhưng rớt token_f1)`, `ride_hailing 9.1% (10/11 fail)`, `ewallet 9.5% (19/21 fail)`, `bill_payment 12.0% (44/50 call fail; 23 mẫu state_match=False)`, `banking 12.8% (41/47 call fail)`, `maps 14.3% (12/14 fail)`, `calendar 21.2% (26/33 fail)`, `contacts 29.4% (36/51 fail)`.

---

## 2. Đối Chiếu Nghiên Cứu DPO & Tool-Calling 2025–2026 và Đánh Giá DPO V1

### 2.1. Tổng Hợp Nghiên Cứu Mới Nhất (2025–2026)

1. **Động lực học lỗi của Offline DPO tiêu chuẩn:**
   - **Unintentional Unalignment: Likelihood Displacement in DPO** (Razin et al., **ICLR 2025**): Chỉ ra hiện tượng *likelihood displacement* — khi cặp `(y+, y-)` có biểu diễn ẩn chồng lấn hoặc ranh giới ngữ nghĩa nhiễu (đo bằng *Centered Hidden Embedding Similarity — CHES*), DPO làm xác suất của cả câu `y+` lẫn các token liên quan bị đẩy lệch sang hành vi ngược lại thay vì chỉ chuyển xác suất từ `y-` sang `y+`.
   - **Learning Dynamics of LLM Finetuning** (Ren & Sutherland, **ICLR 2025 Oral / Outstanding Paper**): Giải thích **"Squeezing Effect"** — chạy DPO off-policy tạo gradient ép xác suất tập trung sai vào token rìa, làm rớt độ tin cậy trên token đúng và kích thích ảo giác/lặp từ.
   - **Cal-DPO** (Xiao et al., NeurIPS), **WPO** (Zhou et al., ACL 2025), **AlphaDPO** (Wu et al., **ICML 2025**), **SimPO** (Meng et al., NeurIPS 2024): Chỉ ra DPO tiêu chuẩn tối ưu *hiệu* log-ratio không chuẩn hóa độ dài (`logp(y+) - logp(y-)`), dẫn đến bùng nổ biên độ implicit reward không kiểm soát nếu thiếu chuẩn hóa độ dài, target margin hoặc reweight theo xác suất on-policy (`WPO`).
2. **DPO Đa Lượt Cho Agent & Tool Calling (2025–2026):**
   - **DMPO** (Shi et al., EMNLP 2024): DPO đơn lượt hỏng trên hội thoại nhiều bước vì hàm phân hoạch không triệt tiêu giữa các trạng thái trung gian; cần chuẩn hóa theo độ dài và ràng buộc occupancy state-action.
   - **DiaTool-DPO** (Kakao Corp, **SIGDIAL 2025**, `arXiv:2504.02882`, code tại `references/tool_calling_repos/14_DiaTool-DPO_SIGDIAL2025/trl/trainer/dpo_trainer.py:1252-1314`): Chỉ tính loss trên từng assistant turn (`get_masked_labels`), áp dụng chiết khấu hình học theo số lượt $(1 - \gamma^{T-t})/(1-\gamma^T)$ và trừ biên `margin` (`beta=0.5, gamma=0.5, margin=2.0, lr=1e-7`), xây cặp theo 3 quỹ đạo chuẩn (`execute`, `slot-filling / ask`, `reject / oos`). Kết quả trên FunctionChat-Bench: slot-filling tăng từ `0.639 -> 0.917` (+44% tương đối), relevance `0.826 -> 0.913`.
   - **Agentic-DPO** (`arXiv`, 07/2026): Phản biện SFT tĩnh bị *exposure bias*, giải quyết bằng **lấy mẫu on-policy 1 bước từ chính policy hiện tại tại các state chuyên gia**, ghép âm/dương sát biên lỗi thật + *Policy-Preserving Augmentation* — nâng `tau-bench retail` từ `21.7% -> 41.4%` trên model 9B.
   - **Skill-CDPO** (ACL / arXiv 2026) & **DEPO** (`arXiv`, 11/2025): Chỉ tối ưu DPO ở các bước nút thắt (*critical steps*) có độ phân kỳ cao và phạt thừa bước gọi API để giữ hiệu quả chi phí token/lượt.
3. **Đồng Thuận 2025–2026: Verifiable RL & On-Policy Distillation Áp Đảo Offline Synthetic DPO:**
   - **SFT Memorizes, RL Generalizes** (Chu et al., 2025): SFT cần thiết để giữ ổn định format, nhưng RL với verifiable outcome reward mới tạo khái quát hóa ngoài phân phối.
   - **Nemotron-Research-Tool-N1** (**ICLR 2026**): RL nhị phân kiểm chứng cấu trúc + tính đúng của lời gọi hàm vượt xa SFT chưng cất.
   - **ToolRL** (**NeurIPS 2025**, `arXiv:2504.13958`) & **GiGPO** (**NeurIPS 2025**, đã có sẵn code trong `src/agent/lfm3_gigpo.py` và `src/agent/lfm3_reward.py`): Tối ưu trực tiếp quỹ đạo đa bước trong môi trường thực bằng group-relative advantage kép (`macro episode` + `micro anchor_state`) không tốn VRAM critic.
   - **LoopTool** (**ACL 2026**, `arXiv:2511.09148`): Bắt buộc khép kín vòng lặp qua **Greedy Capability Probing (GCP)** tìm mẫu sai thực sự của chính checkpoint đang train và sinh dữ liệu sửa lỗi có kiểm chứng thực thi (*EDDE*).
   - **OffSeeker** (`arXiv:2601.18467`): DPO offline chỉ cạnh tranh được khi có dữ liệu quy mô lớn được kiểm chứng sạch (`33k SFT + 21k DPO`), không có nhãn lỗi/đảo cực.
   - **Công thức chính thức của MiniCPM5-2B** (`references/tool_calling_repos/13_MiniCPM/README.md:179-193, 475-482`): OpenBMB **không dùng offline vanilla DPO**, mà chạy **SFT -> Verifiable RL (`JustRL II` / DAPO-style) -> On-Policy Distillation (OPD)** gộp 16 expert RL models (gồm 5 agentic experts) qua reverse KL trên prompt thật — đạt `↑10.96` điểm reasoning/general và **`↑6.96` điểm agentic** trên MiniCPM5-2B (và `↑16` điểm đồng thời giảm `↓29` điểm % câu quá dài trên bản 1B).

---

### 2.2. Đánh Giá Kết Quả Chạy DPO V1: Đã Thành Công Chưa? Tại Sao?

**Kết luận: CHƯA THÀNH CÔNG.** `exp_branchA_dpo_v1` gây **suy giảm ròng nhẹ (`-0.0059` Final Score: `0.4570 -> 0.4511`, `Exact 1.0: 121 -> 116`, `OOS: 0.9048 -> 0.8571`)** và làm thay đổi hành vi trên **`42.1%` (`126/299`)** mẫu đánh giá theo hướng tăng gần gấp đôi lỗi thiếu tham số (`missing_arg_key: 16 -> 31`) và tăng over-calling (`13 -> 20`). Thành công duy nhất mang tính vận hành là đã xác nhận pipeline checkpointing/auto-resume HF chạy thông suốt và trọng số không bị sập (`state_match_rate` vẫn `52.52%`).

**4 nguyên nhân cốt lõi lý giải sự thất bại (đối chiếu giữa telemetry `task-261.log` và lý thuyết 2025–2026):**

1. **Rác vào — Rác ra (Label Inversion & Invalid Chosen):**
   - `144/175` mẫu `when2call` dạy hỏi ngược trên câu đầy đủ -> triệt tiêu cơ hội cải thiện `ask` (`23` ca `ask -> act` đứng yên).
   - `70/150` mẫu `toolsandbox_anchor` có `chosen` thiếu tham số `required` -> trực tiếp đẩy tỷ lệ `missing_arg_key` từ `16` lên `31`.
   - `147/232` mẫu `looptool_mutated` cho gọi `write(id=...)` trước khi `read` trả về ID, cộng `0` mẫu `looptool_mined` thực và `0` mẫu OOS/Stop chuẩn -> không học được biên thực thi thật mà làm hỏng `oos` (`0.9048 -> 0.8571`).
2. **DPO Bão Hòa Bằng Stylistic Hacking Ngay Từ Step 10 (`accuracies = 1.0`):**
   - Telemetry `task-261.log`: Từ Step 5 (`rewards/accuracies = 0.5875, margin = 0.3236`) nhảy vọt lên **Step 10 (`rewards/accuracies = 1.0, margin = 3.492`)** và dính trần `1.0` tới hết Step 60 (`margin = 12.58`). Model không học sự khác biệt nghiệp vụ tool calling mà chỉ học nhận diện cặp mẫu câu `<think>` tách biệt cứng (`"1. Decision: Gọi đúng tool."` vs `"1. Decision: Quyết định sai lệch."`).
3. **Squeezing Effect & Log-Probability Drift Khi Dùng Prompt Trần (ICLR 2025):**
   - Trong `task-261.log`, `logps/rejected` không những không giảm mà **tăng vọt `+64.6 nats`** (`-250.2` ở Step 5 $\rightarrow$ `-185.6` cuối run), kéo `rewards/rejected` lên dương (`-0.156` $\rightarrow$ `+5.44`, đỉnh `+5.99` ở Step 55) trong khi `rewards/chosen` tăng lên `+18.85` dù có phụ trợ SFT (`loss_weights=[1.0, 0.2]`).
   - Toàn bộ 982 cặp thiếu `build_system_prompt(sample)` và 400 cặp sai thẻ `<|im_start|>tool`: gradient trên phân phối prompt trần đã làm nhiễu không gian biểu diễn của `SFT V2` (đã căn chuẩn `build_system_prompt` + `<tool_response>`), gây dịch chuyển `42.1%` output theo hướng mất cân chỉnh tham số (`WPO` / `CHES` / `Learning Dynamics` ICLR 2025).
4. **Dùng Off-Policy Single-Turn DPO Cho Bài Toán Multi-Turn Stateful:**
   - So với `DiaTool-DPO` (tính logp theo từng lượt có mask + discount $\gamma$ + target margin, `lr=1e-7`) hoặc `Agentic-DPO` (negatives lấy mẫu on-policy từ chính SFT V2), DPO V1 chạy hoàn toàn off-policy trên trajectory tổng hợp giả (`{"executed": true}`), nên không thể nâng `state_match_rate` (`52.52% -> 52.52%`) ở các app cần chuỗi đọc-rồi-ghi chặt chẽ (`banking 12.8%`, `bill_payment 12.0%`, `ewallet 9.5%`, `calendar 21.2%`).

---

## 3. Phương Án Tiếp Tục Tối Ưu Model (Lộ Trình Hướng Tới `Tool F1 >= 80%`, `State Match >= 70%`)

### Nguyên Tắc Kiến Trúc & Ràng Buộc
- Bỏ qua nhánh **DPO trên dữ liệu giả tổng hợp (off-policy synthetic pairs)**.
- Mọi bước sinh dữ liệu/RL chỉ chạy trên `1,246` mẫu `train_only`, kiểm chứng mọi action/observation qua `SafeEnv` (`src/data_gen/execution_verifier.py`), bảo toàn `100%` template `build_system_prompt(sample)` + `<tool_response>` + `<think>\n` khớp `adapter/chat_template.jinja`.
- Toàn bộ training giữ ngân sách `< 5B` LLM (`MiniCPM5-2B`), `<= 16,384` tokens, deterministic (`temperature = 0.0`), lưu checkpoint `<= 10 phút` lên HF + auto-resume.

### Tier 0 — Hành Động Ngay (Rẻ, Kiểm Chứng Tức Thì Trong 1 Phiên GPU Nhỏ)
1. **Chạy Eval Đầy Đủ `SFT V2` Trên `1,246` Mẫu `train_only` (LoopTool Phase 1 GCP):**
   - Khắc phục khoảng trống lớn nhất hiện nay: thu thập ma trận lỗi thật của `SFT V2` trên `1,246` mẫu `train_only` (tuyệt đối không đụng `dev_v1`), phân nhóm lỗi theo app (`banking`, `bill_payment`, `ewallet`, `sms`, `calendar`, `contacts`, `ride_hailing`, `maps`) và theo chế độ (`act`, `ask`, `oos`, `step-k binding`).
2. **Nối & Mở Rộng `BoundaryVerifier` + Kiểm Soát Inference Deterministic:**
   - Kích hoạt `BoundaryVerifier` (`src/agent/boundary_verifier.py`) trong `src/agent/react_loop.py` để chặn điền thừa tham số tùy chọn (`alarm.create.repeat`, `smart_home.light_set.color`, `tasks.add.priority`, `robot_vacuum.start.mode`).
   - Bổ sung kiểm tra `required` params theo catalog trước khi phát `env.call()`, và nếu tên app/nhiều candidate trùng tên không rõ, định tuyến sang `sys.ask_missing_info` thay vì đoán ID.
3. **Thử Nghiệm Nhanh TIES / Linear Soup (`SFT V2` + `DPO V1`, `alpha in {0.7, 0.8}`):**
   - Do trọng số chỉ lệch `median 3.23%` và `DPO V1` giúp giảm một nửa ca `act -> ask` (`10 -> 5`), test 1 lượt eval xem `alpha=0.8` có giữ được lợi ích anti-overask mà triệt tiêu nhiễu `missing_arg_key` không; nếu không vượt `0.4570` thì **loại bỏ hoàn toàn `exp_branchA_dpo_v1`** và chỉ neo tiếp từ `exp_branchA_sft_v2`.

### Tier 1 — Sửa Tận Gốc Data Pipeline & Huấn Luyện `Clean SFT V3` (Hoặc `Agentic/DiaTool Multi-Turn DPO`)
1. **Vá Toàn Bộ 3 Bug Sinh Dữ Liệu SFT/DPO Ngay Trong Source Code:**
   - Bắt buộc hàm render mọi mẫu tổng hợp (`src/data_gen/when2call_gen.py`, `src/data_gen/anchor_binding_gen.py`, `src/data_gen/apigen_mt_gen.py`, `scripts/build_minicpm5_dataset_v2.py`) phải dùng **`build_system_prompt(sample)`** + đầy đủ `sample.history` + `role="tool"` được render qua chuẩn `<tool_response>` của `TRAIN_CHAT_TEMPLATE` và prefix `<think>\n`.
   - Vá `scripts/build_minicpm5_dpo_pairs.py`: áp dụng guard `if masked_text == orig_text: continue`, loại observation giả `{"executed": true}` / `{"ok": True, "message": "Thành công"}` — thay bằng **replay thật qua `SafeEnv`** kết hợp `verify_step_validity` (`src/data_gen/execution_verifier.py`) để bảo đảm `100%` `chosen` pass `env.validate()` và đủ `required` params.
   - Sửa `src/data_gen/apigen_mt_gen.py`: tách tuyệt đối lượt `read` -> observation từ `SafeEnv` -> lượt `write` dùng ID đã quan sát, không trộn `read+write` chung một turn đầu tiên.
   - Bổ sung dữ liệu chuẩn hóa cách viết ngắn gọn cho `sms.send.content` / `chat_ott.send_message.content` dựa trên phân phối tóm tắt trong `train_only` để cứu nhóm fail `token_f1 <= 0.6` (`sms 0/18`, `chat_ott`).
2. **Huấn Luyện `Clean SFT V3` Từ Base `MiniCPM5-2B-SFT`:**
   - Thêm dữ liệu phục hồi lỗi thật (failure recovery + ID binding thực trên `banking`, `bill_payment`, `ewallet`, `contacts`, `calendar`) lấy từ Step 0 `train_only` GCP eval + trajectory đã qua **Dual Execution Gate** (`src/data_gen/trajectory_critic.py`). Mục tiêu nâng nền `SFT V3` vượt mốc `56–60%` State Match trước khi vào RL.
3. *(Tùy chọn nếu vẫn chạy Preference Alignment trước/song song RL)* **Chuyển Sang `On-Policy DiaTool-DPO + WPO / Cal-DPO`:**
   - Lấy mẫu on-policy từ `SFT V3` trên `1,246` mẫu `train_only` (như *Agentic-DPO* & *DiaTool-DPO*), chấm điểm từng lượt bằng `SafeEnv`, giữ cặp sai thật cùng prompt đầy đủ, dùng **DiaTool assistant-only turn-discounted loss** (`gamma=0.5`, `margin=2.0`, `lr=1e-7..5e-7`) hoặc **Cal-DPO / length-normalized SimPO/WPO**, **tuyệt đối bỏ mẫu `<think>` khác nhau giữa `chosen` và `rejected`** (giữ cùng cấu trúc `<think>` hoặc chỉ áp loss lên `<function ...>` / turn sai).

### Tier 2 — Bước Nhảy Chính: Multi-Turn Verifiable RL (`GiGPO` + `Step-GRPO`) Trên `SafeEnv`
1. **Kích Hoạt Stack RL Có Sẵn Trong Repo Cho `MiniCPM5-2B`:**
   - Repo đã cài đặt sẵn `src/agent/lfm3_gigpo.py` (`compute_gigpo_advantages`, `compute_official_episode_reward`, `EpisodeRollout`) và `src/agent/lfm3_reward.py` (`verify_step_in_env`, `compute_step_reward`). Cần port/gắn wrapper này sang **MiniCPM5-2B** (`scripts/train_minicpm5_gigpo.py`).
2. **Thiết Kế Rollout & Reward Đúng Hàm Mục Tiêu Cuộc Thi (`metrics.py`):**
   - Sinh $G = 4\text{–}8$ trajectory on-policy cho mỗi prompt trong `train_only` (tập trung vào nhóm hard apps + `ask` + `oos` + `multi-step`).
   - Tín hiệu reward kết hợp hai tầng (theo `GiGPO` NeurIPS 2025 + `Tool-N1` ICLR 2026 + `DEPO`):
     - **Macro Episode Reward:** `2.0 * official_score + 0.35 * tool_f1 - 0.04 * unmatched_reads` (`score_sample` với exact `state_hash` & `token_f1 > 0.6` cho fuzzy fields như `sms.send.content` — dạy trực tiếp cách tóm tắt ngắn để ăn `token_f1`!).
     - **Micro Step Advantage (`AnchorState`):** Cho điểm trung gian khi gọi đúng `read` hợp lệ (`verify_step_in_env`), trích đúng `*_id` từ observation trước, dừng đúng lúc khi xong subgoals, hỏi `sys.ask_missing_info` đúng khi thiếu slot và trả `[]` ở mẫu OOS (phạt nặng `ask` trên câu đủ ý và `write` thiếu tham số).
   - Dùng `KL beta` nhỏ (`0.02–0.04`) hoặc `DAPO/JustRL II` clip-higher, **chạy với gradient trên `<think>` + `<function>` sinh tự nhiên trên phân phối on-policy** để mở khóa khả năng suy luận đa bước (`SFT Memorizes, RL Generalizes`).

### Tier 3 — Khép Kín `LoopTool` + `On-Policy Distillation (OPD)` & Tối Ưu End-to-End Audio
1. **2–3 Vòng `LoopTool` (ACL 2026) Trên `train_only`:**
   - Sau mỗi vòng `GiGPO` -> chạy GCP trên `1,246` mẫu `train_only` -> cô lập các ca lỗi mới (ví dụ `bill_payment` chuyển nguồn tiền khi ví không đủ, `calendar` trùng lịch, `contacts` trùng tên) -> sinh cặp sửa lỗi qua `SafeEnv` -> chạy vòng tiếp theo.
2. **MiniCPM5-Style `On-Policy Distillation (OPD)` Tổng Hợp Expert:**
   - Nếu xuất hiện đánh đổi giữa các kỹ năng (ví dụ model giỏi `act`/`banking`/`bill_payment` đa bước vs model giỏi `ask`/`oos` không over-call), train các checkpoint LoRA chuyên biệt theo cụm nhiệm vụ rồi dùng **OPD (Reverse KL trên top-k logits ở quỹ đạo on-policy)** để chưng cất về một LoRA `MiniCPM5-2B` duy nhất (cùng phương pháp OpenBMB dùng để tăng `↑6.96` điểm agentic cho MiniCPM5-2B).
3. **Nạp Audio Thật & Kiểm Soát Latency Production (`<= 30s` / `< 6B` params):**
   - Khi `MiniCPM5-2B` trên gold text đạt `Tool F1 >= 80%` / `State Match >= 70%`, kiểm tra lại ASR + E2E `.wav -> transcript -> agent -> state_hash`, tinh chỉnh latency vLLM (`<= 30s` trên 1x A30 24GB) và hoàn thiện Docker offline.
