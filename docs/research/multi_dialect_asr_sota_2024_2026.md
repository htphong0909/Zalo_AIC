# Nghiên Cứu SOTA 2024–2026: Tinh Chỉnh ASR Đa Phương Ngữ & Bản Đồ Phát Triển Kiến Trúc Cho LifeMate Voice Assistant

> **Mục tiêu tối thượng (Definition of Done trên Sealed Dev Set):**  
> **WER $\le$ 3.00%** | **CER $\le$ 3.00%** | **Sentence EM $\ge$ 80.00%** | **Latency $\le$ 0.50s / sample**  
> **Định hướng chiến lược:** Từ chối các giải pháp hời hợt (chỉ sửa system prompt hoặc regex text normalization). Tập trung hoàn toàn vào các bước nhảy vọt kiến trúc (Architectural & Training Leaps) dựa trên các công trình SOTA 2024–2026 từ các hội nghị hàng đầu thế giới (**ICASSP, Interspeech, ACL/CCL, IEEE**).

---

## I. Tổng Quan & Cơ Sở Lý Thuyết Âm Học Đối Sánh

### 1. Sự Tương Đồng Cốt Lõi Giữa Phương Ngữ Tiếng Việt và Tiếng Trung
Tiếng Việt và tiếng Trung có chung bản chất loại hình học: **Đơn lập (Isolating), Đơn âm tiết (Monosyllabic) và Có thanh điệu (Tonal)**.
Sự phân hóa phương ngữ 3 miền (Bắc, Trung, Nam) của Việt Nam tương đồng sâu sắc với hệ thống phương ngữ Trung Quốc (Quan thoại, Tứ Xuyên, Mân Nam, Quảng Đông):

| Hiện tượng âm học | Biểu hiện tại Phương ngữ Tiếng Trung | Biểu hiện tại Phương ngữ Tiếng Việt (Bắc - Trung - Nam) |
| :--- | :--- | :--- |
| **Dịch chuyển Formant ($F_1, F_2$)** | Biến đổi nguyên âm giữa Quan thoại và tiếng Tứ Xuyên/Mân | Miền Trung dịch chuyển nguyên âm: *anh $\to$ eng*, *an $\to$ en*, *ơi $\to$ ưa*. |
| **Nhập / Rụng phụ âm đầu** | Phương ngữ phía Nam không phân biệt *zh/ch/sh* với *z/c/s* | Miền Nam không phân biệt *tr/ch*, *s/x*, gộp *r/d/gi $\to$ /j/*, biến *v $\to$ d/z*. |
| **Biến đổi âm cuối (Coda drift)** | Tiếng Mân biến đổi triệt để các âm mũi cuối $n/ng$ | Miền Nam biến đổi âm cuối: *an $\to$ ang*, *at $\to$ ac*, *en $\to$ eng*, *et $\to$ ec*. |
| **Biến dị thanh điệu (Tone merger)** | Gộp và đảo cao độ thanh điệu ở tiếng Ngô, Quảng Đông | Miền Nam gộp thanh hỏi/ngã (5 thanh); miền Trung nén dải tần, thanh nặng gãy. |

### 2. Tại Sao Tinh Chỉnh Bề Mặt (Prompt / Regex) Thất Bại?
Sự sai lệch phương ngữ xảy ra ngay tại **bước trích xuất dải tần âm thanh (Spectrogram)** và **không gian biểu diễn âm học (Acoustic Latent Space)**. Khi Audio Encoder chuyển giao các vector đặc trưng âm thanh bị méo sang LLM Decoder, LLM bị buộc phải "đoán mò" bằng xác suất ngôn ngữ thay vì giải mã âm học trung thực. Do đó, bước nhảy vọt bắt buộc phải diễn ra ở tầng kiến trúc mạng thần kinh.

---

## II. Metadata Các Công Trình SOTA Hàng Đầu (2024–2026)

Dưới đây là danh mục metadata đầy đủ của các bài báo uy tín nhất được công bố tại **ICASSP, Interspeech, ACL Anthology**:

### 1. CLAnSR: Conformer Low-Rank Adaptation for Accent and Speech Recognition
- **Hội nghị:** **ICASSP 2025 (IEEE International Conference on Acoustics, Speech and Signal Processing)**
- **Tác giả:** Xuyi Zhuang, Xinsheng Wang, Shunshun Yin, **Lei Xie** (*Audio, Speech and Language Processing Lab - ASLP@NPU*).
- **Trọng tâm kỹ thuật:** 
  - Huấn luyện đa nhiệm đồng thời: **Nhận diện phương ngữ (Accent Recognition - AR)** và **Nhận diện tiếng nói (ASR)** trên Conformer backbone.
  - Sử dụng **Accent-aware Multi-channel Embeddings**: Vector phương ngữ từ AR head điều chế trực tiếp ma trận LoRA trong ASR Decoder.
- **Ý nghĩa áp dụng:** Chứng minh rằng cắm LoRA vào Audio Encoder giúp giảm **21.8% CER tương đối** trên các giọng địa phương nặng.

### 2. Leveraging LLM and SSL for Chinese Dialect Speech Recognition
- **Hội nghị:** **Interspeech 2025 (Pages 584–588)**
- **Tác giả:** Tianyi Xu, Hongjie Chen, Qing Wang, Lv Hang, Jian Kang, Jie Li, Zhennan Lin, Yongxiang Li, **Lei Xie** (*ASLP@NPU & Alibaba*).
- **Trọng tâm kỹ thuật:**
  - Khảo sát quy mô lớn trên benchmark **KeSpeech** (1,542 giờ, 8 nhóm phương ngữ lớn).
  - So sánh việc kết hợp Self-Supervised Learning (SSL) Audio Encoder với LLM qua các dạng Projector thích nghi phương ngữ.
- **Ý nghĩa áp dụng:** Khẳng định nếu **đóng băng Audio Tower**, mô hình mất đi hơn **30%** năng lực thích nghi phương ngữ. Bắt buộc phải áp dụng **Dual-Tower LoRA** (tinh chỉnh đồng thời Audio Tower và LLM Decoder).

### 3. PYG-ASR: Pinyin-Guided Chinese Speech Recognition with Large Language Model
- **Hội nghị:** **Interspeech 2025**
- **Tác giả:** Zhengjie Jie, Gaofeng Cheng (*Institute of Acoustics, Chinese Academy of Sciences - CAS*).
- **Trọng tâm kỹ thuật:**
  - Kiến trúc **Dual-Target Mapping**: Ánh xạ đặc trưng âm thanh đồng thời sang chuỗi **Âm vị (Pinyin/Phonemes)** và **Văn bản (Text)**.
  - Chuỗi âm vị đóng vai trò **Mỏ neo âm vị (Phonetic Anchor)**, ngăn chặn LLM Decoder hallucinate khi gặp biến âm phương ngữ.
- **Ý nghĩa áp dụng:** Có thể ánh xạ trực tiếp sang tiếng Việt thông qua **Cấu trúc Âm tiết chuẩn (Syllable-Anchor: Phụ âm đầu + Âm đệm + Âm chính + Âm cuối + Thanh điệu)**.

### 4. PY-GEC: LLM Understanding Pinyin for ASR Error Correction
- **Hội nghị:** **ICASSP 2025**
- **Tác giả:** Yuang Li, Xiaosong Qiao, Xiaofeng Zhao, Huan Zhao, Wei Tang, Min Zhang, Hao Yang.
- **Trọng tâm kỹ thuật:**
  - **Cross-Modal Phonetic Alignment**: Huấn luyện LLM nhận thức không gian khoảng cách âm vị (Phonetic Edit Distance).
- **Ý nghĩa áp dụng:** Các từ biến âm địa phương (ví dụ: *"50 ngàn"* vs *"50 nghìn"*, *"dời"* vs *"giờ"*) có khoảng cách ký tự lớn nhưng khoảng cách âm vị gần bằng 0; cơ chế này triệt tiêu hoàn toàn lỗi nhầm từ đồng âm.

### 5. DialectMoE: Mixture-of-Experts for Low-Resource Multi-Dialect ASR
- **Hội nghị:** **CCL 2024 (ACL Anthology)**
- **Trọng tâm kỹ thuật:**
  - Giải quyết hiện tượng **Curse of Dialect Interference** (xung đột độ dốc giữa các phương ngữ có quy tắc âm học đối lập).
  - Thay ma trận LoRA đơn bằng **Dynamic Gating Router** phân phối tín hiệu vào $E$ Expert LoRAs:
    $$\Delta W(x) = \sum_{k=1}^{E} \text{Softmax}(\text{TopK}(W_g \cdot x))_k \cdot (B_k A_k x)$$
  - **Dialect-Agnostic Inference:** Router tự động phân tích phổ âm và kích hoạt tổ hợp Expert phù hợp mà không cần gán nhãn phương ngữ khi infer.

### 6. Activation Steering for Accent Adaptation in Audio LLMs
- **Hội nghị / Nguồn:** **arXiv 2026 (Sun et al.) / ICASSP 2026 track**
- **Trọng tâm kỹ thuật:**
  - Sử dụng **Mechanistic Interpretability**: Chứng minh biến thiên phương ngữ nằm trong một không gian con tuyến tính (linear subspace) ở tầng ẩn giữa Audio Tower và LLM.
  - Kỹ thuật **Contrastive Activation Addition (CAA)**: Trích xuất vector độ dời phương ngữ $v_{\text{accent}} = \mathbb{E}[h_{\text{standard}}] - \mathbb{E}[h_{\text{dialect}}]$, cộng trực tiếp vào tầng ẩn khi inference $h_l \leftarrow h_l + \alpha \cdot v_{\text{accent}}$.
  - **Hiệu quả:** Parameter-free, nắn biểu diễn âm thanh địa phương về tọa độ chuẩn ngay trong quá trình feed-forward.

### 7. DPO-ASR & GRPO-ASR: Direct Preference Optimization for Speech
- **Nguồn:** **2025–2026 Audio LLM Alignment Literature**
- **Trọng tâm kỹ thuật:**
  - Vượt qua giới hạn của hàm mất mát Cross-Entropy (SFT).
  - Tạo các cặp đối chiếu $(x, y_{\text{chosen}}, y_{\text{rejected}})$: trong đó $y_{\text{chosen}}$ là nhãn chuẩn, còn $y_{\text{rejected}}$ là câu do mô hình tự sinh ra bị trượt âm phương ngữ.
  - Hàm mất mát DPO-ASR ép nới rộng biên độ xác suất giữa câu chuẩn và câu lệch giọng, triệt tiêu triệt để các lỗi bão hòa sau SFT.

---

## III. Bảng Tra Cứu Vũ Khí Kỹ Thuật Theo Triệu Chứng Lỗi (Symptom-to-Technique Mapping)

> **Nguyên tắc Agent Loop:** Không ấn định cứng các vòng lặp tương lai (No waterfall pre-scheduling). Ở mỗi vòng lặp $N$, Agent soi kính hiển vi vào tập Train Glass-Box (`temp_train_res.json`), xác định triệu chứng lỗi chiếm đa số, tra cứu giải pháp kỹ thuật phù hợp từ kho tri thức SOTA này, xây dựng **duy nhất 1 giả thuyết**, chạy thực nghiệm, đánh giá kết quả rồi mới quyết định bước đi tiếp theo.

| Triệu chứng lỗi trên Train Glass-Box | Bản chất âm học / mô hình | Kỹ thuật SOTA phù hợp để kiểm chứng | Tham chiếu Paper |
| :--- | :--- | :--- | :--- |
| **Nuốt âm, méo nguyên âm, lệch thanh điệu phương ngữ** (*anh $\to$ eng, an $\to$ en*) | Audio Tower bị đông cứng, dải tần Formant $F_1, F_2$ lệch không được nắn chỉnh | **Dual-Tower LoRA:** Cắm LoRA vào Audio Tower (`thinker.audio_tower`) + unfreeze projection layer | *CLAnSR (ICASSP 2025)*, *Xu et al. (Interspeech 2025)* |
| **Xung đột sửa giọng giữa các miền** (sửa được miền Nam thì hỏng miền Bắc/Trung) | Hiện tượng Gradient Interference khi 1 adapter duy nhất phải học các quy tắc âm học đối lập | **DialectMoE / MAS-LoRA:** Mạng Gating Router tự động kích hoạt các cụm Expert LoRA chuyên biệt | *DialectMoE (ACL 2024)*, *MAS-LoRA (Interspeech 2024)* |
| **SFT bão hòa, nhầm từ đồng âm/gần âm do biến âm** (*50 ngàn $\leftrightarrow$ 50 nghìn, chú $\leftrightarrow$ chủ*) | Hàm mất mát Cross-Entropy (Next Token Prediction) không phạt đủ mạnh phân phối xác suất từ sai | **DPO-ASR Alignment:** Huấn luyện Direct Preference Optimization trên các cặp âm nhầm lẫn thực tế | *DPO for Audio LLMs (2025)*, *GRPO-ASR (2025)* |
| **Ảo giác từ vựng khi gặp giọng địa phương nặng** | Khoảng cách âm thanh đến văn bản quá xa (Acoustic-to-Text Gap), thiếu mỏ neo âm vị | **Syllable/Phonetic Regularization:** Ràng buộc biểu diễn cấu trúc âm tiết chuẩn | *PYG-ASR (Interspeech 2025)*, *PY-GEC (ICASSP 2025)* |
| **Suy giảm độ chính xác trên giọng chuẩn khi học giọng địa phương** | Catastrophic Forgetting trên phân phối giọng phổ thông | **On-Policy Self-Distillation (OPSD):** Dùng phân phối của checkpoint gốc làm mất mát chưng cất | *WeNet & ASLP@NPU (2024-2025)* |
| **Độ trôi biểu diễn phương ngữ ở tầng ẩn khi inference** | Vector kích hoạt (hidden activation) bị lệch khỏi tọa độ chuẩn | **Activation Steering:** Cộng vector độ dời phương ngữ $v_{\text{accent}}$ tại tầng ẩn khi suy luận | *Sun et al. (03/2026)* |

---

## IV. Cam Kết Kỷ Luật Thực Nghiệm (Experiment Discipline)
1. **Tuyệt đối không xâm phạm Dev Set:** Toàn bộ quá trình nghiên cứu, trích xuất lỗi, phân tích âm học và tạo cặp DPO chỉ được thực hiện trên tập Train (`train_ids` trong `configs/splits/dev_v1.json`). Dev set (299 mẫu) là Sealed Blind Box vĩnh viễn.
2. **Tuân thủ ngưỡng tài nguyên:** Toàn bộ hệ thống ASR + Agent LLM luôn được bảo đảm $< 6$ tỷ tham số (ASR $\le 2.04$B) và latency $\le 0.5$ giây/mẫu.
3. **Lưu trữ bất biến:** Mọi thí nghiệm thuộc các làn sóng trên đều được ghi nhận vào `reports/agent_loops/` và dashboard [reports/AGENT_LOOP_DASHBOARD.md](file:///C:/WORKSPACE/ZALO_AIC/reports/AGENT_LOOP_DASHBOARD.md).
