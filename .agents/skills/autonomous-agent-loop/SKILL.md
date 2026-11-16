---
name: autonomous-agent-loop
description: Use when running, resuming, or supervising the autonomous optimization loop under docs/AGENT_LOOP_PROTOCOL.md to optimize the End-to-End Voice Agent pipeline toward Dev WER <= 1.5%, CER <= 1.0%, Sentence EM >= 95%, and Pipeline Dev Tool-Calling F1 > 80% without leaking the Dev set.
---

# Autonomous ASR Agent Loop Orchestrator

## Core Principle: Single Model, Multi-Agent Context Isolation + Permanent Disk Archive

You are the **Main Orchestrator**. Keep your own context window under **15k tokens across 25+ loops** by enforcing **"Write-Deep on Disk, Read-Shallow in Context"**:
- **Never** read raw evaluation transcripts, long Colab outputs, or full model files in the main session.
- **Always** spawn fresh ephemeral subagents (`Model: "inherit"` — using the user's active model) that discard their heavy working context upon returning a compact structured brief.
- **Always** persist 100% of every run (BOTH `ACCEPTED` and `REJECTED`) via [`scripts/loop_logger.py`](file:///C:/WORKSPACE/ZALO_AIC/scripts/loop_logger.py) BEFORE any `git restore src/` rollback.

---

## Non-Negotiable Domain & Protocol Invariants

Read [`docs/AGENT_LOOP_PROTOCOL.md`](file:///C:/WORKSPACE/ZALO_AIC/docs/AGENT_LOOP_PROTOCOL.md) if unfamiliar, and enforce these 7 hard gates on every subagent:
1. **TUYỆT ĐỐI CẤM LEAK DEV SET (Sealed Blind Box Invariant):**
   - The 299 samples in `configs/splits/dev_v1.json` are completely sealed.
   - Zero sample IDs, zero audio transcripts, and zero per-sample error logs from `dev_v1` may ever be viewed, inspected, printed, or leaked to the research agent.
   - Trọng tài 2 (Sealed Blind Dev Referee) strictly outputs only 4 macro metrics: `wer`, `cer`, `sentence_em`, `latency`.
2. **Glass-Box Train Forensic Diagnosis:**
   - All phoneme confusion analysis, error clustering (numbers, loanwords, dialects), and exploratory probes must be conducted on the Train set (`train_ids` in `configs/splits/dev_v1.json`).
3. **Architectural & Methodological Leaps Only:**
   - Ban cheap regex keyword hacks or sample-specific hardcoding. Every loop must test a principled hypothesis (e.g. `initial_prompt` domain biasing, Silero VAD trimming, LoRA/full-finetune on train audio, CTranslate2 INT8/FP16 quantization, acoustic augmentation).
4. **Model Size & Latency Budget (AGENTS.md):**
   - Total system size (ASR + LLM) must be strictly **< 6 Billion parameters**. ASR should stay $\le$ 2B (e.g., PhoWhisper 1.54B, Whisper Turbo 809M, Conformer 146M).
   - Sample latency must stay $\le$ 2.0s (whole pipeline $\le$ 30s).
5. **100% Unit Test Discipline:**
   - `python -m pytest tests/ -q` MUST pass 100% before Phase 3 Referee execution.
6. **LỆNH CẤM ĐẶC BIỆT: CẤM TÁC VỤ NẶNG TRÊN LOCAL (Local Machine Zero-Compute Constraint):**
   - Máy local cực kỳ yếu (CPU/RAM/Disk rất hạn chế).
   - TUYỆT ĐỐI CẤM: Chạy inference mô hình ASR/LLM lớn, huấn luyện/fine-tune LoRA, tải checkpoint/weights lớn về local disk, hoặc chạy benchmark/evaluation hàng loạt mẫu trên máy local.
   - MỌI tác vụ inference, benchmark, fine-tune, hoặc tính toán nặng BẮT BUỘC phải chuyển qua Colab GPU (session `buzzasr`).
   - Máy local CHỈ được phép: git, soạn thảo mã nguồn, chạy quick unit tests mock (`pytest tests/ -q`), và ghi nhận telemetry vòng lặp (`scripts/loop_logger.py`).
7. **LỆNH BẮT BUỘC: CHECKPOINT BỀN VỮNG LÊN HUGGING FACE (Durable Checkpoint Invariant):**
   - Session Colab/Kaggle có thể sập bất kỳ lúc nào; thứ gì chỉ nằm trên ổ đĩa của VM coi như đã mất.
   - MỌI lệnh train/fine-tune/eval/datagen chạy > ~5 phút trên GPU từ xa BẮT BUỘC: (a) chạy preflight HF TRƯỚC khi tốn GPU, (b) đẩy checkpoint/kết quả lên HF **mỗi ≤ 10 phút** và khi kết thúc (kể cả khi lỗi), (c) đẩy output của từng stage trước khi sang stage kế tiếp, (d) có marker hoàn tất để retry bỏ qua phần đã xong.
   - Đọc và tuân thủ skill [`checkpoint-sync-hf`](file:///C:/WORKSPACE/ZALO_AIC/.agents/skills/checkpoint-sync-hf/SKILL.md) (công cụ: `scripts/hf_sync_run.py`, `src/core/hf_sync.py`, `scripts/train_minicpm5_sft.py`). Thiếu `HF_TOKEN`/`HF_SYNC_REPO` hợp lệ ⇒ DỪNG và hỏi user, không được chạy tiếp.
   - `sop-auditor` phải `VERDICT: FAIL` nếu thấy lệnh GPU dài không có cơ chế sync này.

---

## Definition of Done (DoD on Sealed Dev)

- **Pipeline Dev Tool-Call F1:** $\mathbf{> 80.00\%}$ (Mục tiêu cốt lõi của Trợ lý Voice-to-Action LifeMate)
- **Pipeline Dev State Match Rate:** $\ge \mathbf{75.00\%}$ (Độ chính xác trạng thái môi trường)
- **Dev WER (ASR):** $\le$ **1.50%**
- **Dev CER (ASR):** $\le$ **1.00%**
- **Dev Sentence EM (ASR):** $\ge$ **95.00%**
- **End-to-End Latency:** $\le$ **10.0s / sample** (ASR $\le 0.5$s, LLM ReAct $\le 9.5$s, quy định BTC $\le 30.0$s)
- **Total Model Parameters:** $\mathbf{< 6.0B}$ params (PhoWhisper + LLM Agent)
- **Status:** Evaluated and verified purely on Sealed Blind Dev without any data contamination.

---

## Step-by-Step Orchestration Loop

### Step 0: One-Time Subagent Registration (Start of Session)
Read [`references/subagent_contracts.md`](file:///C:/WORKSPACE/ZALO_AIC/.agents/skills/autonomous-agent-loop/references/subagent_contracts.md) and call `define_subagent` once to register the 3 specialized roles (all invoked with `Model="inherit"`):
1. `loop-researcher` (`EnableWriteTools=True`, `EnableMCPTools=False`)
2. `sop-auditor` (`EnableWriteTools=False`, `EnableMCPTools=False`)
3. `colab-referee` (`EnableWriteTools=True`, `EnableMCPTools=True`)

*(Note: If the current campaign has not yet anchored an initial baseline, dispatch `colab-referee` once with `--baseline` first to anchor Loop 00).*

### Step 1: Anchor Context Shallowly (< 500 Tokens)
Run:
```powershell
python scripts/loop_logger.py brief --max-recent 5
```
Check Definition of Done:
- **ASR Gate:** `Dev WER <= 1.5%`, `Dev CER <= 1.0%`, `Dev Sentence EM >= 95.0%`.
- **Pipeline Tool-Calling Gate:** `Dev Tool-Call F1 > 80.0%`, `State Match >= 75.0%`.
If not yet fulfilled, proceed to Loop `N`.

### Step 2: Phase 1 & 2 — Dispatch Fresh `loop-researcher`
1. Initialize the permanent archive folder for Loop `N`:
   ```powershell
   python scripts/loop_logger.py start --loop <N> --hypothesis "Pending Phase 1 diagnosis" --agent loop-researcher
   ```
2. Invoke a **fresh** `loop-researcher` (`Model="inherit"`), passing ONLY the `brief` output and `loop_id = N`.
3. `loop-researcher` diagnoses Train Glass-Box errors, saves `01_forensic_brief.md` via `loop_logger.py forensic`, implements ONE architectural or decoding leap in `src/asr/`, runs `python -m pytest tests/ -q` until 100% green, and returns a **<= 20-line summary** (`Hypothesis`, `Modified Files`, `Optimization Summary`, `Pytest Count`).

### Step 3: Phase 2b — Dispatch Fresh `sop-auditor` (Independent Bias-Free Gate)
1. Update live monitor: `python scripts/loop_logger.py phase --loop <N> --phase 2_audit --agent sop-auditor`
2. Invoke a **fresh** read-only `sop-auditor` (`Model="inherit"`). It inspects `git diff src/ configs/` cold against the invariants above (especially **Anti-Snoop / Zero Dev Leakage**).
3. If `sop-auditor` returns `VERDICT: FAIL`, message `loop-researcher` to fix before touching GPU.
4. Once `VERDICT: PASS`, capture the immutable patch snapshot **before** Referee execution:
   ```powershell
   python scripts/loop_logger.py snapshot --loop <N> --optimization "<1-line summary>" --pytest-passed --pytest-summary "<X/X passed>" --audit-passed --audit-notes "PASS"
   ```

### Step 4: Phase 3 & 4 — Dispatch Fresh `colab-referee` & Finalize Archive
1. Read [`references/telemetry_and_colab_runbook.md`](file:///C:/WORKSPACE/ZALO_AIC/.agents/skills/autonomous-agent-loop/references/telemetry_and_colab_runbook.md). Invoke a **fresh** `colab-referee` (`Model="inherit"`) for Loop `N` with the exact `Hypothesis`.
2. `colab-referee` executes Referee 1 (`src/eval/train_asr_evaluator.py`) and Referee 2 (`src/eval/blind_dev_evaluator.py`) on Colab GPU (session `buzzasr`), saves JSON outputs to `scripts/scratch/`, runs `python scripts/loop_logger.py finalize --loop <N> ...`, and returns **only the Pareto Gate output**.
3. Verify the permanent folder `reports/agent_loops/loop_NNN_.../` exists, along with updated [`reports/AGENT_LOOP_DASHBOARD.md`](file:///C:/WORKSPACE/ZALO_AIC/reports/AGENT_LOOP_DASHBOARD.md) and [`reports/agent_loops/live_monitor.html`](file:///C:/WORKSPACE/ZALO_AIC/reports/agent_loops/live_monitor.html).
   If the loop trained or fine-tuned anything, also verify (list the HF repo files) that the final adapter/checkpoint is on Hugging Face **before** the Colab session is stopped (Gate 7).
4. Loop straight to **Step 1** for Loop `N + 1` until Definition of Done is met.
