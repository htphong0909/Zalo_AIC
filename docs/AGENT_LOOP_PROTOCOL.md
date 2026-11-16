# Quy Trình Nghiên Cứu Chuẩn Mực: Agent Loop Tự Trị Cho ASR (Code As Referee)

> **Tuyên ngôn cốt lõi**: Agent (AI) đảm nhiệm vai trò nhà khoa học nghiên cứu độc lập — Code kiểm định (Referee) đảm nhiệm vai trò trọng tài khách quan.  
> **Nguyên tắc tối thượng**: **TUYỆT ĐỐI CẤM LEAK DEV SET**, cấm overfit/snoop vào tập kiểm định, bảo toàn luật Pareto chống suy thoái, tôn trọng giới hạn kích thước mô hình (< 6B params) và độ trễ ($\le$ 30s).

---

## 1. Triết Lý Nghiên Cứu & Đạo Đức Kỹ Thuật (Philosophy & Core Ethics)

1. **Agent là Nhà nghiên cứu - Code là Trọng tài Khách quan (Code as Objective Referee):**
   - Agent (LLM) đảm nhiệm việc chẩn đoán nguyên nhân lỗi âm học/ngữ nghĩa, suy ngẫm bản chất phương ngữ, loanwords, số học và trực tiếp can thiệp mã nguồn (`src/asr/`, prompts, decoders, adapters).
   - Mã kiểm định (Referee) đo lường chính xác, lạnh lùng, tuyệt đối không bị lay chuyển bởi lý lẽ của Agent. Chỉ có số liệu thực tế cải thiện trên tập Dev mới được chấp nhận vào master.
2. **Tuyệt đối cấm Leak Dev Set & Snooping (Zero-Leakage Invariant):**
   - **Tập Dev (`configs/splits/dev_v1.json`, 299 mẫu) là Hộp Đen Niêm Phong (Sealed Blind Box)**.
   - Agent nghiên cứu (`loop-researcher`) **tuyệt đối không được phép xem ID mẫu, file âm thanh, nhãn gốc, hay log chi tiết từng mẫu** của tập Dev.
   - Trọng tài 2 (Sealed Blind Dev Referee) chỉ xuất duy nhất 4 chỉ số vĩ mô: **WER (%), CER (%), Sentence EM (%), và Average Latency (s)**.
   - Mọi hành vi in transcript Dev, hardcode chuỗi ký tự Dev, hoặc can thiệp file split đều bị coi là vi phạm nghiêm trọng và bị Trọng tài kiểm toán (`sop-auditor`) bác bỏ ngay lập tức (`VERDICT: FAIL`).
3. **Phòng thí nghiệm Kính Hiển Vi trên tập Train (Glass-Box Forensic Laboratory):**
   - Toàn bộ việc chẩn đoán pháp y, ma trận nhầm lẫn âm vị, lỗi dấu câu, lỗi số học, lỗi tiếng Anh/loanwords được thực hiện trên tập **Train Sạch** (`train_ids` trong `configs/splits/dev_v1.json`, 1.246+ mẫu).
   - Agent được xem toàn bộ log chi tiết của tập Train để tìm ra quy luật âm học tự nhiên.
4. **Ưu tiên bước nhảy kiến trúc & kỹ thuật thực chất (Architectural Leaps over Cheap Hacks):**
   - Cấm việc viết code bắt chuỗi thủ công (rule-based hardcoding theo từng câu cụ thể) hoặc chỉ sửa prompt/regex hời hợt.
   - Bắt buộc tham chiếu và triển khai theo lộ trình SOTA 2024–2026 đã được kiểm chứng tại ICASSP/Interspeech/ACL: [Nghiên Cứu SOTA 2024–2026: Tinh Chỉnh ASR Đa Phương Ngữ](research/multi_dialect_asr_sota_2024_2026.md).
   - Các hướng nhảy vọt kỹ thuật được ưu tiên:
     - **Dual-Tower LoRA:** Cắm LoRA vào Audio Tower + LLM Thinker (`CLAnSR`, ICASSP 2025; `Xu et al.`, Interspeech 2025).
     - **DialectMoE / MAS-LoRA:** Phân phối các Expert LoRA vùng miền (Bắc/Trung/Nam) qua Dynamic Router (`DialectMoE`, ACL 2024).
     - **DPO-ASR Alignment:** Huấn luyện Direct Preference Optimization phạt trực tiếp các cặp lỗi trượt âm/đồng âm phương ngữ (2025–2026).
     - **Activation Steering:** Can thiệp vector độ dời phương ngữ ở tầng ẩn khi inference (`Sun et al.`, 2026).
5. **Giới hạn cứng của cuộc thi (Hard Constraints):**
   - Tổng dung lượng hệ thống (ASR + LLM) bắt buộc phải **< 6 Tỷ tham số (6B)**. ASR nên duy trì $\le$ 1.6B để dành tài nguyên cho LLM.
   - Độ trễ xử lý âm thanh trung bình phải **$\le$ 2.0s** (đảm bảo cả pipeline $\le$ 30s).
   - Suy luận hoàn toàn offline trong Docker, không gọi API ngoài.
6. **LỆNH CẤM ĐẶC BIỆT: CẤM TÁC VỤ NẶNG TRÊN LOCAL (Local Zero-Compute Constraint):**
   - Máy local cực kỳ yếu (CPU/RAM/Disk rất hạn chế).
   - TUYỆT ĐỐI CẤM: Chạy inference mô hình ASR/LLM lớn, huấn luyện/fine-tune LoRA, tải checkpoint mô hình lớn về local disk, hoặc chạy benchmark/evaluation hàng loạt mẫu trên máy local.
   - MỌI tác vụ inference, benchmark, fine-tune, hoặc tính toán nặng BẮT BUỘC phải chuyển qua Colab GPU (session uzzasr).
   - Máy local CHỈ dùng cho: git, soạn thảo mã nguồn, chạy quick unit tests mock (pytest tests/), và ghi nhận telemetry vòng lặp (scripts/loop_logger.py).

---

## 2. Hệ Thống Mục Tiêu & Chỉ Tiêu Định Lượng (Definition of Done)

Vòng lặp tự trị hướng tới mục tiêu tối thượng: **Đạt chuẩn Definition of Done (DoD) toàn diện cho cả ASR và LLM Tool-Calling Agent trên tập Dev độc lập**:

| Chỉ số Đo Lường | Trạng Thái Baseline (Loop #00) | Hiện Tại (Loop #38 SOTA / LFM3 SFT) | **Mục Tiêu Tối Thượng (Definition of Done)** | Ý Nghĩa Kỹ Thuật |
| :---: | :---: | :---: | :---: | :--- |
| **Pipeline Tool-Call F1** | 0.00% | 42.42% (SFT init) | **$\mathbf{> 80.00\%}$** | **Mục tiêu cốt lõi: $\ge 80\%$ lời gọi công cụ khớp hoàn hảo cả tool name lẫn arguments.** |
| **Pipeline State Match** | 0.00% | 30.94% | **$\ge \mathbf{75.00\%}$** | Đảm bảo tính toán khớp hash trạng thái môi trường ứng dụng mô phỏng. |
| **Dev WER** | 28.79% | 1.50% | **$\le$ 1.50%** | Giảm thiểu tối đa lỗi nhận dạng từ vựng ASR. |
| **Dev CER** | 20.42% | 1.03% | **$\le$ 1.00%** | Triệt tiêu lỗi sai lệch dấu câu và âm vị ASR. |
| **Dev Sentence EM** | 2.01% | 86.96% | **$\ge$ 95.00%** | $\ge$ 95% câu khẩu lệnh ASR khớp chính xác 100%. |
| **End-to-End Latency** | ~5.8s | ~5.5s | **$\le$ 10.0s / mẫu** | ASR $\le 0.5$s + LLM $\le 9.5$s (xa dưới trần 30.0s của BTC). |
| **Total Parameters** | ~4.7B | 3.27B (PhoWhisper + LFM3) | **< 6.0B params** | Tuân thủ giới hạn tối đa toàn hệ thống của cuộc thi. |

---

## 3. Kiến Trúc Hai Trọng Tài Độc Lập (Dual-Referee System)

```
                          ┌──────────────────────────────┐
                          │    MÃ NGUỒN PIPELINE HIỆN TẠI │
                          └──────────────┬───────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
   ┌─────────────────────────────────┐       ┌─────────────────────────────────┐
   │ TRỌNG TÀI 1: TRAIN SẠCH         │       │ TRỌNG TÀI 2: SEALED BLIND DEV   │
   │ (Kính hiển vi chẩn đoán)        │       │ (Hộp đen niêm phong - CẤM LEAK) │
   ├─────────────────────────────────┤       ├─────────────────────────────────┤
   │ - Xuất chi tiết từng audio sai  │       │ - CHỈ XUẤT 4 CHỈ SỐ DUY NHẤT:   │
   │ - Ma trận nhầm lẫn âm vị / từ   │       │   * Dev WER: X.XX%              │
   │ - Lỗi loanwords, số, thực thể   │       │   * Dev CER: Y.YY%              │
   │ ➔ NƠI AGENT SOI LỖI ĐỂ TỐI ƯU   │       │   * Dev EM:  Z.ZZ%              │
   │                                 │       │   * Latency: T.TTs              │
   │                                 │       │ ➔ KHÔNG ID, KHÔNG AUDIO, CẤM LOG│
   └────────────────┬────────────────┘       └────────────────┬────────────────┘
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                          ┌──────────────────────────────┐
                          │   PARETO GATE (Quyết định)   │
                          │   Cải thiện Dev ➔ ACCEPT     │
                          │   Bị Overfit ➔ ROLLBACK      │
                          └──────────────────────────────┘
```

### 1. Trọng tài 1: `src/eval/train_asr_evaluator.py` (Kính hiển vi)
* **Đối tượng:** Tập con đại diện từ `train_ids` (1.246+ mẫu trong `configs/splits/dev_v1.json`).
* **Quyền hạn của Agent:** Được xem toàn bộ log chi tiết:
  * ID mẫu âm thanh, thời lượng, ground truth và transcript dự đoán.
  * Phân loại lỗi: Lỗi số học (ITN), từ mượn tiếng Anh, phương ngữ Bắc/Trung/Nam, ảo giác do nhiễu.
* **Mục đích:** Là nơi Agent điều tra pháp y và thử nghiệm các ý tưởng mới.

### 2. Trọng tài 2: `src/eval/blind_dev_evaluator.py` (Hộp đen niêm phong)
* **Đối tượng:** Toàn bộ 299 mẫu của tập `dev_v1` (`configs/splits/dev_v1.json`).
* **Quyền hạn của Agent:** **BỊ BỊT MẮT HOÀN TOÀN**:
  * Mã nguồn chấm điểm nằm trong sandbox kín, chỉ ghi ra file JSON chứa đúng 4 khóa macro cho phép: `dev_wer`, `dev_cer`, `dev_em`, `dev_latency`.
  * Tuyệt đối không xuất bất kỳ tên mẫu hay nội dung câu nào ra console hoặc log.

---

## 4. Quyết Định Pareto Gate & Quy Chuẩn Đóng Mở Vòng Lặp

Sau khi Hai Trọng Tài hoàn tất chấm điểm, hệ thống tự động đưa ra phán quyết theo **Luật Pareto**:

1. **Điều kiện ACCEPT (Chấp thuận):**
   - $\text{Dev WER}$ giảm ($\Delta \text{WER}_{dev} < 0$), HOẶC $\text{Dev EM}$ tăng mà $\text{Dev WER}$ không tăng.
   - Đồng thời không xảy ra hiện tượng sụp đổ nghiêm trọng trên tập Train ($\Delta \text{WER}_{train} \le +2.0\%$).
   - Thao tác: Git commit code mới, cập nhật baseline tốt nhất vào `configs/agent_loop_state.json`.
2. **Điều kiện REJECT (Bác bỏ):**
   - $\text{Dev WER}$ tăng hoặc $\text{Dev EM}$ giảm (dấu hiệu overfit vào train).
   - Thao tác: Lưu toàn bộ git diff và metrics vào `reports/agent_loops/loop_XXX/`, sau đó tự động chạy `git restore src/` để phục hồi mã nguồn sạch.

---

## 5. Quy Chuẩn Kỹ Thuật Subagents (3 Vai Trò Độc Lập)

Để bảo đảm Agent điều phối chính không bị tràn context window (< 15k tokens qua hàng chục vòng lặp), 3 subagents được phân công rõ ràng:
1. **`loop-researcher`**: Nghiên cứu, chẩn đoán Train Glass-Box, hiện thực hóa mã nguồn trong `src/asr/`, chạy unit test xanh 100%.
2. **`sop-auditor`**: Kiểm toán viên độc lập (chỉ đọc), kiểm tra diff xem có hardcode/snooping dữ liệu Dev không.
3. **`colab-referee`**: Thực thi Hai Trọng Tài trên GPU (Colab hoặc local), ghi nhận kết quả và chạy `loop_logger.py finalize`.
