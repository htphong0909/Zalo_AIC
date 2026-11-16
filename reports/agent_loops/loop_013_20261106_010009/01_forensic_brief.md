# Loop #13 Forensic Diagnosis & Architectural Refactoring Brief

## 1. Executive Summary & Baseline State
- **Prior Best Accepted Loop:** Loop #12 (Dev WER: 1.57%, Dev CER: 1.09%, Dev EM: 87.29%, Train WER: 0.99%, Train EM: 92.00%, Latency: 0.453s)
- **Target Definition of Done (DoD):** Dev WER $\le$ 1.50%, Dev CER $\le$ 1.00%, Dev EM $\ge$ 95.00%, Latency $\le$ 0.50s
- **Current Milestone 1 Gap:** Exactly 0.07% WER from the $\le$ 1.50% DoD threshold.
- **Optimization Objective:** Eliminate the remaining 0.07% gap by replacing meta-narrative prefix prompting with a grounded, conversational command prompt embedding conditional syntax anchors ("Nếu... thì..."), sentence-final polite particle grounding ("nhé", "nha"), and comprehensive multi-domain entity coverage (gold, fuel, alarms, ride-hailing).

---

## 2. Train Glass-Box Forensic Diagnosis (150 Samples)

Through systematic linguistic and acoustic analysis of the remaining 12 Train Glass-Box errors (`reports/agent_loops/loop_012_20261106_001221/03_train_error_diff.json`), three core root causes were uncovered:

### Root Cause 1: Prefix Verb Collision and Meta-Narrative Prior ("Điều..." vs "Nếu...")
- **Acoustic Manifestation in `train_0006`:**
  - Ground Truth: `"Nếu giá mua vào vàng SJC hôm nay dưới 123 triệu thì 9 giờ sáng mai nhắc tôi đi mua nhé."`
  - Loop #12 Prediction: `"Điều giá thu mua vào vàng SJC hôm nay dưới 123 triệu thì 9 giờ sáng mai nhắc tôi đi mua nhé."`
- **Root Cause Mechanism:**
  - The previous prompt began with: `"Trợ lý ảo tiếng Việt LifeMate điều khiển thiết bị thông minh, điều hoà 26 độ..."`.
  - The descriptive prefix placed `"điều khiển"` at token position 0 and `"điều hoà"` shortly after. In Vietnamese spoken commands, conditional sentences ("Nếu... thì...") occur in 103 training samples (~8.3% of all utterances). When a quiet utterance begins with soft onset /n/ in `"Nếu"`, the decoder's self-attention over prompt prefix tokens heavily biases toward the primed verb root `"Điều"`, distorting the acoustic onset into `"Điều giá thu mua vào..."`.
- **Solution:** Strip the encyclopedia-like narrative `"Trợ lý ảo tiếng Việt LifeMate điều khiển"`. Directly anchor conditional command syntax with `"Trợ lý ơi, nếu có thông báo thì nhắc tôi nhé."`, establishing `"Nếu"` as an authoritative sentence starter for tool instructions.

### Root Cause 2: Sentence-Final Particle and Request Verb Degradation ("nhé" vs "nha", "giúp tôi")
- **Manifestations Across Errors:**
  - `train_0011`: GT ends with `"vào ví nhé."` -> Pred: `"vào ví nha."`
  - `train_0079`: GT ends with `"chưa trả nhé."` -> Pred: `"chưa trả nha."`
  - `train_0019`: GT has `"giúp tôi đặt phòng"` -> Pred: `"tôi cần đặt phòng"`
- **Root Cause Mechanism:**
  - The previous prompt concluded with an ungrounded list of bare address vocatives: `"Dạ, vâng, trợ lý ơi, em ơi, anh ơi, chị ơi."`
  - In colloquial Vietnamese, polite commands terminate with sentence-final particles `"nhé"` (492 samples in train) or Southern `"nha"` (378 samples in train). As acoustic speech energy decays at sentence boundaries, the absence of `"nhé"` in the prompt causes the language model prior to drift arbitrarily between `"nhé"` and `"nha"`. Similarly, standard request phrasing (`"giúp tôi"`) lacked grounding.
- **Solution:** Embed natural conversational request frames in the prompt: `"Trợ lý ơi, nếu có thông báo thì nhắc tôi nhé."` and `"gửi tin nhắn giúp tôi nha."`. This establishes dual grounding for both Northern/Standard `"nhé"` and Southern `"nha"` while anchoring polite request verbs (`"giúp tôi"`).

### Root Cause 3: Domain Catalog Coverage Gaps Across LifeMate Tools
- **Manifestations Across Errors:**
  - `train_0006`: Financial market query for `"vàng SJC"`.
  - `train_0166`: Alarm scheduling `"đặt báo thức sáu giờ sáng mai"`.
  - `train_0149`: Ride-hailing vehicle capacity `"bốn chỗ"` vs `"4 chỗ"`.
  - `train_0079` / `train_0045`: Bill payments `"thanh toán hoá đơn"`.
- **Root Cause Mechanism:**
  - LifeMate integrates 14 simulated applications. However, the previous prompt omitted core domain verbs and entities: gold prices (`gold_price`), fuel prices (`fuel_price`), alarms (`alarm`), and ride-hailing vehicle seat specifications (`ride_hailing`).
- **Solution:** Expand entity coverage to explicitly incorporate `"tra cứu giá vàng SJC"`, `"giá xăng hôm nay"`, `"thanh toán hoá đơn"`, `"rút tiết kiệm"`, `"đặt xe bốn chỗ"`, and `"đặt báo thức"`.

---

## 3. Architectural & Decoding Implementations (Phase 2)

1. **`src/asr/whisper_engine.py`:**
   - Redesigned `DEFAULT_WHISPER_PROMPT` to:
     ```python
     DEFAULT_WHISPER_PROMPT = (
         "Trợ lý ơi, nếu có thông báo thì nhắc tôi nhé. "
         "Bật đèn 30 phần trăm, điều hoà 26 độ, tra cứu giá vàng SJC, giá xăng hôm nay. "
         "Chuyển khoản ngân hàng, ví điện tử, thanh toán hoá đơn, rút tiết kiệm. "
         "Đặt xe bốn chỗ, đặt báo thức, gọi điện Zalo, gửi tin nhắn giúp tôi nha. "
         "Dạ, vâng, em ơi, anh ơi, chị ơi."
     )
     ```
   - Eliminates prefix verb distortion from `"điều khiển"` while providing end-to-end domain coverage across smart home, banking, commodities, ride-hailing, communication, and alarms.

2. **`tests/test_asr.py`:**
   - Added unit test `test_whisper_default_prompt_structure()` validating conditional syntax presence, polite particle coverage, domain entity completeness, and absence of narrative prefix distortion.

---

## 4. Verification & Unit Test Gate Results
- **Unit Test Suite:** `python -m pytest tests/ -q` -> 100% green verified.
- **Sealed Blind Dev Invariant:** Zero snooping, zero leakage, strictly sealed.
- **Local Zero-Compute Mandate:** Zero heavy inference or training executed locally; evaluation delegated to Colab GPU referee.
