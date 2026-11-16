# Track B Implementation Plan — Clean Fold-0-Gated Data Engine, 4-Slot Reasoning SFT V3, Real ASR Adaptation & Step-DPO Calibration

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the **43 Oracle + 7 E2E Fold 0 failure cases** using verified insights from `LoopTool`, `Hammer`, `When2Call`, `APIGen-MT`, `ToolSandbox`, `HammerBench`, `Tool-N1`, `AURA`, `MiniCPM`, and `Step-DPO`—raising **unseen held-out E2E Voice accuracy** from `0.7503` (Track A Dev 299 baseline) toward **`≥ 0.83–0.85+` on unseen Fold 0 (`N=250`)** before final `1,246`-sample consolidation / Track C RL.

**Architecture:** Replace the no-thinking, synthetic-CTC, static `qwen35_sft_v2` pipeline with a clean, leak-free 2-stage Track B pipeline trained strictly on `folds 1..4` (`996` samples) and gated strictly on unseen `fold_0` (`250` samples):
1. **Runtime & Grammar Fixes (`B1`):** Allow closing `</think>` in GBNF guided decoding, add `diverse_nbest` beam groups to `WhisperASREngine` (without altering greedy 1-best `1.50%` Loop #38 WER), and add `loyalty.get_points({})` / prompt hardening.
2. **Real Champion ASR Cache (`B2`, L4):** Transcribe `train_only` (`1,246` samples + history audio) with Loop #38 PhoWhisper (`nbest=3`) so SFT/DPO train on genuine PhoWhisper errors + `[Gợi ý đồng âm ASR]` diffs (`~30%` of trajectories) instead of synthetic CTC corruption.
3. **4-Slot Reasoning + `LoopTool`/`Hammer`/`HammerBench` Data Engine (`B3`):** Build `scripts/build_qwen35_sft_v3_data.py` + `src/data_gen/edde_augmenter.py` with `enable_thinking: true`, arithmetic/weekday/condition scratchpads, mandatory fuzzy entity checklists, `4-16` turn history coreference replay, self-correction (`IS/SO`), and Hammer default/func/param masking—100% validated via `execution_verifier.verify_step_validity` and `SafeEnv`.
4. **Clean G4 SFT V3 Training (`B4`, G4 Train → L4 E2E Fold 0 Gate):** Fine-tune `Qwen/Qwen3.5-4B` (`r=32, alpha=64`, `lr=4e-5`, `2.0 epochs`, `max_length=16384`) on `folds 1..4`, auto-resuming from HF Hub `checkpoints/latest` every `40` steps / `8` min, prune vision/MTP weights via `prune_qwen35_text_only.py`, and gate on unseen `fold_0` (L4 E2E + Oracle).
5. **LoopTool Hard-Error Probing + Prefix-Matched Step-DPO (`B5`, G4 Train → L4 E2E Fold 0 Gate):** Mine hard `folds 1..4` failures (`T=0.7`, `N=8` sampling via `run_looptool_gcp.py`) + targeted counterfactuals into prefix-matched Step-DPO pairs (`sigmoid + 0.35 * sft`), train on G4 (`lr=1.5e-5`, `beta=0.1`), sweep & soup checkpoints (`merge_lfm3_soup.py`) on unseen `fold_0`, then execute consolidation vs Track C decision (`B6`).

**Tech Stack:** Python 3.12, PyTorch + PEFT LoRA + TRL (`SFTTrainer`, `DPOTrainer`) + Liger Kernel (`use_liger_kernel: true`), vLLM (L4 inference & Fold 0 sweep), `kzap201/phowhisper-large-zaloai` (`1.54B`, Loop #38), `Qwen/Qwen3.5-4B` pure-text (`4.21B`), Hugging Face Hub sync (`src/core/hf_sync.py`), Google Colab (**G4 = training only; L4 = E2E & vLLM evaluation**).

---

## 0. Forensic Basis & Paper-to-Fix Mapping (Fold 0, `N = 250`)

Measured Track A baseline on `fold_0` (`N = 250`): **Oracle `0.8823`** (`SM 91.77%`, `F1 90.93%`; `207` perfect, `23` `SM=True & F1<1.0`, `20` `SM=False`) → **E2E `0.8596`** (`SM 89.61%`, `F1 89.42%`, ASR WER `0.84%`; `7` E2E drops vs Oracle, `1` gain). Track A generalization gap on unseen Dev 299: **`0.7788` Oracle / `0.7503` E2E** (`11.26s/sample` L4 E2E vs `30s` limit → **`>18s` headroom for `<think>`**).

| Fold 0 Error Bucket | Observed Cases (Sample IDs) | Root Cause in V2 Pipeline | Paper & Code Mechanism in Track B | Target Module(s) |
| :--- | :---: | :--- | :--- | :--- |
| **1. Time, Weekday, Count Math & Conditionals** | **9** (`0224, 0279, 0777, 0802, 1055, 1115, 1262, 1326, 1433`) | `enable_thinking: false` (`0/4954` SFT rows had `<think>`); no `checking` vs `savings` or branch calc trace | **`Tool-N1` + `LoopTool` EDDE**: turn ON `<think>` + Slot 2 arithmetic/weekday/balance calc + `edde_time_arithmetic` | `src/data_gen/edde_augmenter.py`, `configs/qwen35_sft_v3_clean.yaml` |
| **2. Ignored `history` & Reflexive `ask`** | **7** (`0708, 0902, 0970, 1252, 1419, 1629, 1657`) | Multi-turn coref drowned by long history (`hist=4..20`); `loyalty.get_points(alias)` failed → gave up with `ask` | **`HammerBench` External + `ToolSandbox`**: `edde_long_history_coref` + Slot 2 citation `[Lịch sử lượt k]` + `loyalty.get_points({})` fallback | `src/agent/react_loop.py`, `src/agent/prompts.py`, `src/data_gen/edde_augmenter.py` |
| **3. Fuzzy `token_f1 <= 0.6` & Entity/Address Drops** | **10** (`0470, 0697, 0814, 0830, 0878, 1266, 1289, 1322, 1419, 1542`) | `metrics.calls_match` requires `token_f1 > 0.6` (no punct strip); SFT dropped IDs (`tx_01`), street addresses, numbers or padded filler | **`HammerBench` ROUGE/Fidelity + `Step-DPO`**: Slot 3 `Từ khóa bắt buộc` extraction + fuzzy chosen/rejected DPO contrast | `scripts/build_qwen35_sft_v3_data.py`, `scripts/build_qwen35_dpo_v3.py` |
| **4. Optional Args `set(p)!=set(g)` & `ask`/`oos`/Self-Correction** | **14+** (`0748, 0751, 0801, 0894, 1093, 1098, 1118, 1188, 1237, 1308, 1340, 1347, 1534, 1555, 1593, 1704`) | Optional keys (`location`, `end`, `contacts.search.query`, `phone_number` vs `contact_id`); self-correction trap `"lộn"`; OOS confused with `ask` | **`Hammer` (name+default mask) + `When2Call` Tri-Branch + `HammerBench` `IS/SO` + `LoopTool` `GCP->JGLV->EDDE`**: explicit schema grounding & Step-DPO optional-key/OOS margin | `src/data_gen/edde_augmenter.py`, `scripts/run_looptool_gcp.py`, `scripts/build_qwen35_dpo_v3.py` |
| **5. E2E ASR Acoustic Drops & Out-of-Dist `[Gợi ý đồng âm]`** | **7** (`0079, 0234, 0488, 0830, 0898, 1125, 1289`) | V2 trained on synthetic CTC (`0.15`), `0` real Loop #38 transcripts, `0` `[Gợi ý đồng âm ASR]` rows; `num_beams=3` collapsed to 3 identical strings! | **`AURA` Diverse N-Best + Real Loop #38 SFT/DPO Mix**: `30%` real Whisper transcripts + `[Gợi ý đồng âm]` in SFT V3; `diverse_nbest` in `WhisperASREngine` | `src/asr/whisper_engine.py`, `scripts/run_asr.py`, `scripts/build_qwen35_sft_v3_data.py` |

> [!CAUTION]
> **Hard Safety & Anti-Leakage Rules (`AGENTS.md` §2 & CV Integrity):**
> 1. **NEVER** open, read, or mine `<split>_environment/` or `state.json`. Only use `zaic_tools.ToolEnv` / `src/env/safe_env.py` (`validate`, `call`, `get_tool_calls`, `get_state_hash`).
> 2. **NEVER** include any ID from `configs/splits/dev_v1.json` (`299`) OR `configs/splits/train_only_5folds.json -> folds['fold_0']` (`250`) in ANY Track B data build, GCP mining, SFT, or Step-DPO dataset until B1+B2 exit gates are cleared.
> 3. **Hardware split:** **Colab G4** is strictly for Training (`SFT V3`, `Step-DPO`). **Colab L4** is strictly for Inference/E2E evaluation (`run_asr.py`, `run_agent.py`, `run_eval.py`).
> 4. **HF Durable Sync:** Every Colab job runs `HFSyncer.preflight()` first, auto-resumes from `<prefix>/checkpoints/latest`, and persists every `checkpoint-<step>` + final pure-text `Qwen3_5ForCausalLM` weights (`< 4.3B`, total `< 6B`).

---

## Task B1: Runtime, Grammar & ASR Diversity Prerequisites (Local, Zero GPU)

Fix three verified code bugs discovered during forensic inspection so `<think>` training, diverse ASR N-best hints, and `loyalty.get_points` recovery work reliably at inference time.

**Files:**
- Modify: `src/agent/llm_client.py` (update `build_qwen35_tool_grammar`, `VLLMBackend._build_sampling_params`)
- Modify: `src/asr/whisper_engine.py` (`_decode_nbest_hypotheses`, `transcribe`)
- Modify: `scripts/run_asr.py` (pass `diverse_nbest` flag when `--nbest-beams > 1`)
- Modify: `src/agent/prompts.py` (refine `build_system_prompt` rules for `loyalty.get_points`, street address copying, math verification, and `[Gợi ý đồng âm ASR]`)
- Modify: `src/agent/react_loop.py` (`loyalty.get_points` empty/error auto-retry with `{}`)
- Create: `tests/test_trackb_runtime_fixes.py`

### Steps

- [ ] **Step B1.1: Write failing unit tests in `tests/test_trackb_runtime_fixes.py`**

Create `tests/test_trackb_runtime_fixes.py` testing:
1. `build_qwen35_tool_grammar(tools, allow_thinking=True)` produces GBNF allowing optional reasoning text + `</think>` before `<tool_call>` while `allow_thinking=False` keeps strict `<tool_call>` root.
2. `WhisperASREngine._decode_nbest_hypotheses` passes `num_beam_groups=num_beams_nbest` and `diversity_penalty=0.5` when `diverse_nbest=True` (and gracefully retries without `diversity_penalty` if the underlying `generate` rejects group beam kwargs), without changing 1-best greedy `gen_kwargs`.
3. `build_system_prompt` includes rules for `loyalty.get_points({})`, verbatim street address/ID copying, and `[Gợi ý đồng âm ASR]` disambiguation.
4. `run_react_loop` automatically retries `loyalty.get_points({})` once (or feeds clear empty-program recovery tool feedback without aborting) when `loyalty.get_points({"program": "Sen Air"})` returns empty `programs: []` / `ok: False`.

Run test to verify failure:
```powershell
$env:PYTHONPATH="."; python -m pytest -q tests/test_trackb_runtime_fixes.py
```

- [ ] **Step B1.2: Implement grammar fix in `src/agent/llm_client.py`**

Modify `build_qwen35_tool_grammar(tools: list[dict[str, Any]], *, allow_thinking: bool = False) -> str`:
```python
    call_expr = " | ".join(fn_refs)
    if allow_thinking:
        root_rule = 'root ::= ( think-block )? call ( "\\n" call )* ws\nthink-block ::= [^<]* "</think>" ws'
    else:
        root_rule = 'root ::= call ( "\\n" call )* ws'
    lines = [
        root_rule,
        'call ::= "<tool_call>\\n" (' + call_expr + ') "\\n</tool_call>"',
        'value ::= [^<]+',
        'ws ::= [ \\n\\t]*',
        *fn_rules,
    ]
```
In `VLLMBackend._build_sampling_params`, pass `allow_thinking=self.enable_thinking` into `build_qwen35_tool_grammar(tools, allow_thinking=self.enable_thinking)`. Keep `max_new_tokens` backward compatible.

- [ ] **Step B1.3: Implement Diverse N-Best Option in `src/asr/whisper_engine.py` & `scripts/run_asr.py`**

Update `WhisperASREngine._decode_nbest_hypotheses(..., diverse_nbest: bool = False)` and `transcribe(..., diverse_nbest: bool = True)`:
- Keep `self.model.generate(input_features, **gen_kwargs)` (the 1-best greedy pass) **100% untouched** so Loop #38 `1.50%` WER cannot regress.
- Only for the secondary `num_beams_nbest > 1` N-best pass: when `diverse_nbest=True` and `num_beams_nbest >= 2`, try `num_beam_groups=num_beams_nbest, diversity_penalty=0.5` inside a `try/except (TypeError, ValueError)` that falls back cleanly to standard `num_beams=num_beams_nbest`. Wait: also normalize both raw and ITN forms or keep `_sanitize_hyp` so genuine phonetic alternatives (e.g., `"500 nghìn"` vs `"500 ngày"`, `"qua tử"` vs `"bao tử"`) surface in `nbest` & `extract_phonetic_diff`.
- Note: check `scripts/run_asr.py` lines 201-205 & 214-229 so when `engine.transcribe(..., num_beams_nbest=args.nbest_beams)` runs, diverse candidates aren't filtered away unless `top_score - score > args.nbest_max_delta`.

- [ ] **Step B1.4: Refine `src/agent/prompts.py` and `src/agent/react_loop.py`**

In `src/agent/prompts.py` (`build_system_prompt`):
- Add under Rule 1: `Nếu tra cứu điểm thành viên (loyalty.get_points) mà chưa biết chính xác mã/tên chương trình trong hệ thống, hãy gọi loyalty.get_points({}) không tham số trước để lấy danh sách đầy đủ.`
- Add under Rule 4: `Khi tạo ghi chú, nhắc việc, email, tin nhắn hoặc tìm đường (maps.get_route), sao chép chính xác địa chỉ đầy đủ, mã giao dịch, số tiền và từ khóa cốt lõi từ lời nói/kết quả tool; tính kỹ phép cộng trừ phút và thứ trong tuần từ current_time.`
- Add Rule 8: `8. **Gợi ý đồng âm ASR:** Nếu lời nói kèm \`[Gợi ý đồng âm ASR: "..."]\`, kiểm tra cả hai phương án theo ngữ cảnh ứng dụng (hóa đơn/số tiền, danh bạ, lịch sử) trước khi kết luận thiếu thông tin hoặc ngoài phạm vi.`

In `src/agent/react_loop.py` inside `run_react_loop` after executing a read tool (around lines 875-882):
- Check if `c.name == "loyalty.get_points"` and `c.arguments` was non-empty AND `(res.get("ok") is False or not res.get("programs"))`: append a concise system/tool hint `"Không tìm thấy chương trình theo tên vừa nhập; hãy gọi loyalty.get_points({}) không tham số để xem toàn bộ chương trình."` (no fabricated state access, purely handling tool return payload cleanly!).

- [ ] **Step B1.5: Verify tests pass**

```powershell
$env:PYTHONPATH="."; python -m pytest -q tests/test_trackb_runtime_fixes.py tests/test_llm_client.py tests/test_prompts.py tests/test_react_loop.py tests/test_whisper_context_biasing.py tests/test_aura_nbest_diff.py
```
Expected: All PASS (`0` failed).

---

## Task B2: Build `train_only` Real PhoWhisper Loop #38 + Diverse N-Best ASR Cache (Colab L4)

Eliminate the distribution mismatch between SFT training (`0` real PhoWhisper transcripts, `0` `[Gợi ý đồng âm ASR]` examples) and E2E inference.

**Files:**
- Generated artifact: `data/asr/train_only_phowhisper_loop38_nbest.jsonl` (backed up to HF `htphong0909/zalo-aic-sft-checkpoints/shared_assets/asr/`)
- Output metrics: `experiments/exp_qwen35_trackB_asr/train_only_asr_metrics.json`

### Steps

- [ ] **Step B2.1: Package clean codebase & sync `train.jsonl.gz` + `train_audio` assets if missing on Colab L4**

Verify HF preflight locally first:
```powershell
$env:PYTHONPATH="."; python -c "from src.core.hf_sync import HFSyncer; s=HFSyncer('htphong0909/zalo-aic-sft-checkpoints'); print(s.preflight())"
```

- [ ] **Step B2.2: Provision Colab L4 session and run `scripts/run_asr.py` on `train_only`**

Run **strictly on L4** with `--ids configs/splits/train_only.json` (`1,246` samples; ZERO `dev_v1` IDs!):
```bash
PYTHONPATH=. python scripts/run_asr.py \
  --split train \
  --root data \
  --ids configs/splits/train_only.json \
  --engine whisper \
  --model-id kzap201/phowhisper-large-zaloai \
  --dtype float16 \
  --nbest-beams 3 \
  --resume \
  --out data/asr/train_only_phowhisper_loop38_nbest.jsonl \
  --metrics-out experiments/exp_qwen35_trackB_asr/train_only_asr_metrics.json
```
*(Estimated runtime on Colab L4: ~12–16 min including multi-turn history wavs.)*

- [ ] **Step B2.3: Upload ASR cache to HF Hub & download locally, then check cache stats**

Push `train_only_phowhisper_loop38_nbest.jsonl` to HF (`shared_assets/asr/train_only_phowhisper_loop38_nbest.jsonl`) via `HFSyncer.push`, download locally to `data/asr/train_only_phowhisper_loop38_nbest.jsonl`, and verify:
- Total rows == `1,246` (`len(set(ids) & dev_ids) == 0`).
- Count rows where `audio_text != oracle audio_text` AND rows where `asr_phonetic_diff is not None`.
- Do NOT inspect or touch `dev_v1`. Inspect summary stats on `folds 1..4` (`996` rows) only.

---

## Task B3: Implement `edde_augmenter.py` + Upgraded 4-Slot `<think>` + `scripts/build_qwen35_sft_v3_data.py` (Local TDD)

Before modifying/integrating with `src/data_gen/distill_think.py` and `src/agent/qwen35_sft.py`, inspect those two files directly during implementation to preserve all existing public signatures & passing tests (`test_distill_think.py`, `test_qwen35_sft.py`).

**Files:**
- Create: `src/data_gen/edde_augmenter.py`
- Modify: `src/data_gen/distill_think.py` (extend `synthesize_deterministic_think` with arithmetic/weekday verification, history citation, fuzzy keyword extraction, and `[Gợi ý đồng âm ASR]` awareness inside 4 concise slots)
- Modify: `src/agent/qwen35_sft.py` (ensure `step_to_qwen35_example(..., enable_thinking=True)` puts valid `<think>...</think>` BEFORE `<tool_call>...</tool_call>` when present)
- Create: `scripts/build_qwen35_sft_v3_data.py`
- Create: `tests/test_trackb_data_pipeline.py`

### Exact Architecture of `src/data_gen/edde_augmenter.py` (`folds 1..4` ONLY)

Every generator function in `src/data_gen/edde_augmenter.py` accepts `samples: list[Sample], catalog: dict[str, Any], seed: int = 42, exclude_ids: set[str] | None = None`, asserts `sample.id not in exclude_ids`, executes deterministic offline transformations, synthesizes upgraded 4-slot `<think>`, and filters 100% through `verify_step_validity(step, catalog)`:

1. **`generate_time_and_math_edde(samples, catalog, count=350, seed=42)` (`Fix Bucket 1`):**
   - Picks `folds 1..4` samples calling `reminders.create`, `alarm.create`, `calendar.create_event`, `calendar.update_event`, `appliances.start_cycle`, `fitness.get_activity`, `banking.transfer`, `ewallet.pay_qr`, `weather.get_forecast`.
   - Applies clean deterministic perturbations to offsets (`delay_minutes` ∈ `{45, 75, 90, 120, 150, 180}`, minute subtraction `-12, -18, -25, -27, -35` before event time, day-of-week relative shifts `+1..+4` days matching `current_time` weekday, or slot shifts past busy interval `[start_A, end_A]` → `[end_A, end_A + dur]`), updating BOTH user utterance/history text AND gold ISO datetime/numeric arguments identically!
   - Also adds **Banking Checking-vs-Savings & Threshold Branch** variations (`> 131 triệu` vs `< 131 triệu`, `checking` sufficient vs insufficient) grounded in observed tool output balances.

2. **`generate_long_history_coref_edde(samples, catalog, count=300, seed=42)` (`Fix Bucket 2`):**
   - Takes `folds 1..4` multi-turn/coreference samples (`gold_calls` referencing an entity/value/scene_id/event/device established in `sample.history` OR needing a lookup tool like `notes.list` / `smart_home.get_status` / `loyalty.get_points({})`).
   - Prepends `3–7` unrelated distractor QA turn pairs sampled from other `folds 1..4` histories (respecting token budget `max_prompt_tokens=11000`), verifying the gold answer & `step_0` target remain bit-identical and the 4-slot `<think>` explicitly extracts `[Lịch sử lượt k: ...]` in Slot 2 (`Ranh giới: Act — KHÔNG ask_missing_info vì lịch sử đã có đủ`).

3. **`generate_fuzzy_fidelity_edde(samples, catalog, count=250, seed=42)` (`Fix Bucket 3`):**
   - Selects `folds 1..4` trajectories with fuzzy arguments (`notes.create`, `reminders.create`, `tasks.add`, `email.create_draft`, `chat_ott.send_message`, `sms.send`, `maps.get_route`, `maps.save_place`) containing numbers, codes (`tx_`, `mail_`), street addresses, or proper names.
   - Extracts mandatory anchor tokens from gold fuzzy values + tool observations and embeds `Từ khóa bắt buộc: [...]` inside `<think>` Slot 3 so the policy plans exact entity retention before generating XML arguments.

4. **`generate_self_correction_and_oos_edde(samples, catalog, count=250, seed=42)` (`Fix Bucket 4`: `HammerBench IS/SO` + `When2Call`):**
   - Synthesizes Vietnamese self-correction templates over `folds 1..4` pairs of sibling tools (`tasks.add` → `"lộn, vô shopping_list"` `shopping_list.add`; `smart_home.scene_run` `"buổi sáng, à thôi, tiếp khách"`; `alarm` time revision `"6 giờ, à khoan, 7 rưỡi"`), labeled with the post-correction tool & Slot 1 explicitly noting `Đổi ý/tự sửa lỗi -> bỏ phương án cũ, chọn ...`.
   - Generates **Out-of-Scope Device/App Refusal** contrasts (`appliances` vs `smart_home`, missing app in `installed_apps`) labeled `[]` (`oos`) so unsupported device requests do NOT hallucinate `sys.ask_missing_info`.

5. **`generate_hammer_masked_edde(base_steps, catalog, count=500, seed=42)` (`03_Hammer` full 3-mask):**
   - Port `replace_function_names_new`, `replace_param_names_new`, AND **`replace_param_default_values_news`** (from `03_Hammer_ICLR2025/train/data_processing.py`) cleanly into pure functions inside `edde_augmenter.py` (`mask_tools_and_params_hammer`), preserving valid tool schemas, history names, and XML call alignment while masking schema surface strings (`p=0.67` on synthetic masked slice) so the LLM learns true parameter semantics & optional-arg minimalism (`only pass optional args present in utterance/rule`).

### Dataset Mixture Target (`scripts/build_qwen35_sft_v3_data.py` — Clean `folds 1..4`)

Run `scripts/build_qwen35_sft_v3_data.py --include-folds fold_1 fold_2 fold_3 fold_4 --enable-thinking --asr-transcripts data/asr/train_only_phowhisper_loop38_nbest.jsonl --asr-ratio 0.30 --out data/sft/qwen35_train_sft_v3_clean.jsonl.gz`:
- **Base `folds 1..4` unrolled trajectories (`~2,380` steps):**
  - `70%` Oracle clean text (`enable_thinking: true`, 4-slot `<think>`)
  - `30%` Real Loop #38 PhoWhisper transcripts (curated from `train_only_phowhisper_loop38_nbest.jsonl`, including `history_texts` AND `[Gợi ý đồng âm ASR: ...]` whenever generated → target remains verified gold!)
- **`edde_time_and_math`**: `~350` steps (`folds 1..4`)
- **`edde_long_history_coref`**: `~300` steps (`folds 1..4`)
- **`edde_fuzzy_fidelity`**: `~250` steps (`folds 1..4`)
- **`edde_self_correction_and_oos`**: `~250` steps (`folds 1..4`)
- **`hammer_masked_v3` (name + param + default-value mask)**: `~480` steps
- **`when2call_tribranch` (missing required / post-lookup ask / OOS contrast)**: `~450` steps
- **`apigen_mt` (`include_step_1=True`, `folds 1..4`)**: `~600` steps
- **`anchor_binding` (`folds 1..4`)**: `~400` steps
- **`recovery` (`empty_search` + `xref_repair`, `folds 1..4`)**: `~260` steps
- **Hard-bucket weighted replay (`folds 1..4` samples matching conditional/arithmetic/loyalty/weather/multi-step rare apps)**: `~280` steps
- **Total:** `~6,000` verified steps (`100%` passing `verify_step_validity`, **`0` `fold_0` IDs, `0` `dev_v1` IDs**).

### Steps

- [ ] **Step B3.1: Write failing tests in `tests/test_trackb_data_pipeline.py`**

Verify:
1. All `generate_*_edde` helpers raise/refuse if `fold_0` or `dev_v1` IDs leak when `--exclude-ids` / fold guard is active, and 100% of emitted rows pass `verify_step_validity(row, catalog)`.
2. Every emitted assistant completion has valid balanced `<think>...</think>` followed by `<tool_call>...` (or short Vietnamese OOS/done reply when `step_calls == []`), with non-empty 4-slot headers (`1. Ý định:`, `2. Dữ liệu:`, `3. Kế hoạch:`, `4. Ranh giới:`).
3. `mask_tools_and_params_hammer` preserves consistency between masked `<tools>` JSON definitions, history calls, and target completion XML tags/param names.
4. `build_qwen35_sft_v3_data.py` CLI produces a `.jsonl.gz` on tiny mock fixture with `~30%` ASR-noisy prompts including `[Gợi ý đồng âm ASR: ...]` formatting.

Run failing test:
```powershell
$env:PYTHONPATH="."; python -m pytest -q tests/test_trackb_data_pipeline.py
```

- [ ] **Step B3.2: Read `src/data_gen/distill_think.py` & `src/agent/qwen35_sft.py` in full to preserve contracts, then implement `src/data_gen/edde_augmenter.py`, upgrade `distill_think.py` + `qwen35_sft.py`, and create `scripts/build_qwen35_sft_v3_data.py`**

- [ ] **Step B3.3: Run tests & generate `data/sft/qwen35_train_sft_v3_clean.jsonl.gz` locally**

```powershell
$env:PYTHONPATH="."; python -m pytest -q tests/test_trackb_data_pipeline.py tests/test_distill_think.py tests/test_qwen35_sft.py
$env:PYTHONPATH="."; python scripts/build_qwen35_sft_v3_data.py `
  --folds-file configs/splits/train_only_5folds.json `
  --include-folds fold_1 fold_2 fold_3 fold_4 `
  --asr-transcripts data/asr/train_only_phowhisper_loop38_nbest.jsonl `
  --asr-ratio 0.30 `
  --enable-thinking `
  --out data/sft/qwen35_train_sft_v3_clean.jsonl.gz
```
Inspect token distribution (`mean`, `p95`, `max <= 16384`), verify `0` `fold_0` & `0` `dev_v1` sample IDs, and verify 100% XML parseability with `src/agent/tool_caller.parse_tool_calls`.

---

## Task B4: Stage B1 — Clean SFT V3 Training (G4) & Held-Out `Fold 0` Gate (L4)

**Files:**
- Create: `configs/qwen35_sft_v3_clean.yaml`
- Create: `configs/oracle_qwen35_4b_v3.yaml`
- Create: `configs/e2e_phowhisper_qwen35_4b_v3.yaml`
- Reuse: `scripts/train_qwen38_sft.py`, `scripts/colab_qwen35_sft.py`, `scripts/prune_qwen35_text_only.py`
- Update (if needed for custom dataset filename arg cleanly): `scripts/colab_qwen35_sft.py`

### Exact Configs

`configs/qwen35_sft_v3_clean.yaml`:
```yaml
name: qwen35_sft_v3_clean
base_model: Qwen/Qwen3.5-4B
train_file: data/sft/qwen35_train_sft_v3_clean.jsonl.gz
output_dir: experiments/exp_qwen35_sft_v3_clean
seed: 42
enable_thinking: true
prompt_style: qwen35

lora:
  r: 32
  alpha: 64
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - in_proj_qkv
    - in_proj_z
    - out_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  num_train_epochs: 2.0
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 4.0e-5
  weight_decay: 0.01
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  max_length: 16384
  use_kernels: false
  use_liger_kernel: true
  logging_steps: 5
  save_steps: 40
  hf_interval_min: 8.0
```
*(Why `lr: 4.0e-5` & `epochs: 2.0` vs V2 `5e-5, 1.5 ep`: V3 introduces structured 4-slot `<think>` generation ~120 tokens/turn alongside `apigen_mt`/`edde`/`hammer`, matching `xLAM` `5e-5` / `Tool-N1` `2e-5..4e-5` stable regime without overfitting `996` underlying environments.)*

`configs/oracle_qwen35_4b_v3.yaml` and `configs/e2e_phowhisper_qwen35_4b_v3.yaml`:
- Identical to V2 configs EXCEPT: `enable_thinking: true`, `max_new_tokens: 640`, `repetition_penalty: 1.05`, `guided_decoding: true`, `prompt_style: qwen35`, `max_steps: 10`, `max_repairs: 3`, `timeout_s: 25.0`, `max_prompt_tokens: 15000`.

### Steps

- [ ] **Step B4.1: Stage `qwen35_train_sft_v3_clean.jsonl.gz` + updated code bundle to HF Hub (`exp_qwen35_sft_v3_clean/assets`)**

- [ ] **Step B4.2: Train `qwen35_sft_v3_clean` on Colab G4 ONLY with mandatory HF preflight, `checkpoint-<step>` preservation & auto-resume**

```bash
PYTHONPATH=. python scripts/train_qwen38_sft.py \
  --config configs/qwen35_sft_v3_clean.yaml \
  --hf-repo htphong0909/zalo-aic-sft-checkpoints \
  --hf-prefix exp_qwen35_sft_v3_clean
```
*(Estimated wall time on G4: `~6,000 * 2 / 16 ≈ 750` optimizer steps, ~55–85 min depending on packed length; every 40 steps pushed to `exp_qwen35_sft_v3_clean/checkpoints/checkpoint-<step>` & mirrored to `latest/`. Stop G4 cleanly once `merged/` + `training_summary.json` are verified on HF.)*

- [ ] **Step B4.3: Export pure text-only merged weights & checkpoint sweep on CLEAN UNSEEN `Fold 0` on Colab L4**

On Colab L4 (`fold0_ids` generated from `configs/splits/train_only_5folds.json -> folds['fold_0']`):
1. Run `run_agent.py` + `run_eval.py` across candidate checkpoints (`checkpoint-480`, `checkpoint-600`, `final`) on **Oracle Fold 0** to pick best generalizing checkpoint (or merge 2 best via `scripts/merge_lfm3_soup.py` only if beneficial).
2. Run **E2E Fold 0** using **diverse N-best PhoWhisper Loop #38** on L4 (`run_asr.py --ids fold0_ids --engine whisper --nbest-beams 3` + `run_agent.py --config configs/e2e_phowhisper_qwen35_4b_v3.yaml` + `run_eval.py`).
3. **Exit Gate B1 (Clean Unseen `Fold 0`, `N=250`):**
   - Compare directly vs unseen Dev299 Track A baseline (`0.7788` Oracle / `0.7503` E2E):
   - Target **`Oracle Fold 0 (unseen) ≥ 0.825`**, **`E2E Fold 0 (unseen) ≥ 0.805`**, **`avg_latency ≤ 14.0s/sample`** (`≪ 30s`), **`schema_abort == 0`**.
   - Stop L4 session (`colab stop`) before moving to B5 training or immediately run Task B5 Stage 1.

---

## Task B5: Stage B2 — `LoopTool` Hard-Error GCP Probing (`folds 1..4`) + Prefix-Matched `Step-DPO` Calibration (G4 Train → L4 E2E Gate)

SFT V3 learns `<think>` format & patterns, yet cross-entropy cannot directly optimize `score_sample` cliffs (`set(p)==set(g)`, `token_f1 > 0.6`, checking-account preference, optional-arg minimalism, OOS boundary). Stage B2 closes that exact cliff using **real sampled failures from Stage B1 on `folds 1..4`** paired with verified Step-DPO counterfactuals.

**Files:**
- Create: `scripts/run_looptool_gcp.py` (samples `N=8` rollouts at `T=0.7` in vLLM + executes in `SafeEnv` strictly over `folds 1..4`, outputting `failures_folds14.jsonl` with real failure trajectories + perplexity/logprob difficulty stats, cf. `07_LoopTool_ACL2026/dataloop/greedy_capability_prob.py`)
- Modify/Reuse: `scripts/colab_looptool_probe.py` (parameterize `COLAB_GPU="L4"` and `--ids folds_1_4` chunking to orchestrate `run_looptool_gcp.py` without code duplication)
- Create: `scripts/build_qwen35_dpo_v3.py` (builds prefix-matched Step-DPO pairs `{'prompt', 'chosen', 'rejected', 'pair_type', 'sample_id'}` reusing `configs/splits/train_only_5folds.json` exclusions)
- Create: `scripts/train_qwen35_dpo.py` (Qwen3.5 ChatML TRL `DPOTrainer` with `loss_type: ["sigmoid", "sft"]`, `loss_weights: [1.0, 0.35]`, `HFSyncer` checkpoint-<step> sync & auto-resume, saving pure-text merged `Qwen3_5ForCausalLM`)
- Create: `configs/qwen35_dpo_v3_clean.yaml`
- Create: `tests/test_qwen35_dpo.py`

### Exact Step-DPO Pair Taxonomy (`scripts/build_qwen35_dpo_v3.py`, `~1,800–2,200` pairs, `folds 1..4` ONLY)

Using **Step-DPO prefix matching** (`chosen` and `rejected` share identical `<think>` Slot 1 `1. Ý định:` + Slot 2 data lines up to the diverging decision line when synthetic, OR natural failing rollout vs gold step from `run_looptool_gcp.py`!):

1. **On-Policy `GCP` Mined Pairs (`~600–900` pairs from `run_looptool_gcp.py` on `folds 1..4`):**
   - For any `folds 1..4` sample where `SFT V3 Clean` at `T=0.7` (`N=8`) produced a failing rollout (`score_sample < 1.0`, wrong `state_hash`, or `SM=True & F1 < 1.0`):
     - Pair `(chosen = verified gold step with 4-slot <think>, rejected = failing SFT V3 step)`.
     - Automatically targets the **exact brittle weights of SFT V3 Clean** without inventing mismatches!
2. **Optional Argument Minimalism & Key-Set Exactness (`~350` pairs):**
   - `weather.get_*`: omitting/including `location` properly (`Huế / Ninh Bình` → include vs default local → omit).
   - `calendar.create_event`: matching gold `end` rule vs omitting unrequested `priority`/`repeat`/`color`/`mode`/`query`.
   - `phone.call`: `phone_number` vs `contact_id` depending on whether phone digits vs contact search resolved it.
   - Anti-extra-read-call: `chosen = minimal gold calls`, `rejected = extra unrequested chat_ott.list_chats / robot_vacuum.get_status / contacts.search`.
3. **Fuzzy Slot Fidelity vs Verbose/Entity-Dropped Counterfactuals (`~300` pairs):**
   - `chosen = exact concise gold text retaining all codes (tx_xx), numbers (45.000 cây, 75.000 đồng), full street addresses` vs `rejected = truncated/paraphrased string dropping entity (`tok_f1 <= 0.6`) OR venue name instead of address OR verbose greeting prefix/suffix`.
4. **History Coreference & Anti-Reflexive `ask` vs Under-Asked `ask` vs `oos` (`~350` pairs):**
   - When `history` contains the answer: `chosen = Act step citing [Lịch sử]`, `rejected = premature sys.ask_missing_info`.
   - Ambiguous contact/missing slot: `chosen = sys.ask_missing_info`, `rejected = hallucinated entity ID/dropoff`.
   - Unsupported device / cancelled intent: `chosen = [] (OOS)`, `rejected = hallucinated sys.ask_missing_info / write call`.
5. **Real Loop #38 ASR Phonetic Recovery Pairs (`~300` pairs):**
   - Prompt uses real noisy PhoWhisper transcript (`+ [Gợi ý đồng âm ASR]`): `chosen = gold call resolving homophone correctly`, `rejected = literal mistranscription failure` (or `[]` false-OOS).

### Exact `configs/qwen35_dpo_v3_clean.yaml` Hyperparameters
```yaml
name: qwen35_dpo_v3_clean
base_model: <SFT_V3_CLEAN_MERGED_HF_OR_LOCAL>
train_file: data/sft/qwen35_train_dpo_v3_clean.jsonl.gz
output_dir: experiments/exp_qwen35_dpo_v3_clean
seed: 42
enable_thinking: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - in_proj_qkv
    - in_proj_z
    - out_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  num_train_epochs: 1.5
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 1.5e-5
  beta: 0.1
  loss_type: ["sigmoid", "sft"]
  loss_weights: [1.0, 0.35]
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  max_length: 12288
  max_prompt_length: 10240
  logging_steps: 5
  save_steps: 30
  hf_interval_min: 8.0
```

### Steps

- [ ] **Step B5.1: Write failing tests in `tests/test_qwen35_dpo.py` verifying pair format, Step-DPO prefix sharing, zero `fold_0`/`dev_v1` leakage, and `DPOConfig` builder**

```powershell
$env:PYTHONPATH="."; python -m pytest -q tests/test_qwen35_dpo.py
```

- [ ] **Step B5.2: Implement `scripts/run_looptool_gcp.py`, `scripts/build_qwen35_dpo_v3.py`, and `scripts/train_qwen35_dpo.py` (with `HFSyncer` checkpoint-<step> sync & auto-resume + pure-text merged export)**

- [ ] **Step B5.3: Run GCP probing on `folds 1..4` (L4), build `qwen35_train_dpo_v3_clean.jsonl.gz` locally, verify tests pass**

```powershell
$env:PYTHONPATH="."; python -m pytest -q tests/test_qwen35_dpo.py
# After GCP output downloaded from L4:
$env:PYTHONPATH="."; python scripts/build_qwen35_dpo_v3.py `
  --folds-file configs/splits/train_only_5folds.json `
  --include-folds fold_1 fold_2 fold_3 fold_4 `
  --gcp-failures experiments/exp_qwen35_sft_v3_clean/gcp_folds14_failures.jsonl `
  --asr-transcripts data/asr/train_only_phowhisper_loop38_nbest.jsonl `
  --out data/sft/qwen35_train_dpo_v3_clean.jsonl.gz
```

- [ ] **Step B5.4: Train `qwen35_dpo_v3_clean` on Colab G4 ONLY, then run full Fold 0 checkpoint sweep & E2E Voice eval on Colab L4**

Evaluate checkpoints (`step 30, 60, 90, final`) + test 2-checkpoint uniform soup (`scripts/merge_lfm3_soup.py`) strictly on clean unseen **`Fold 0` (`250` samples)** on L4.

- **Exit Gate B2 (Clean Unseen `Fold 0`, `N=250`):**
  - Target **`Oracle Fold 0 (unseen) ≥ 0.845–0.855`**, **`E2E Fold 0 (unseen) ≥ 0.825–0.840`**, **`State Match ≥ 89.0%`**, **`SM=True & F1<1.0` error count cut by ≥ 50% vs unseen baseline (`~25+` → `≤ 12`)**, **`latency ≤ 15.0s/sample`**, **`schema_abort == 0`**.

---

## Task B6: Decision Gate, Consolidation / Track C Bridge & Competition Verification (`AGENTS.md` §2 & §8)

- [ ] **Step B6.1: Evaluate Track B Gate Outcome on Unseen `Fold 0`**

Choose the highest-ROI path strictly from measured **Clean `Fold 0` E2E Score**:
- **Case 1 (`E2E Fold 0 ≥ 0.845`):** Scaling from `996` (`folds 1..4`) → all `1,246` (`train_only`) with identical frozen V3 SFT + Step-DPO recipes adds ~25% more in-domain entity/app coverage. Consolidate Full `1,246` on G4 (`exp_qwen35_trackB_full`) and run single-shot **Sealed `Dev 299` E2E on L4** targeting **`≥ 85.0%+`**.
- **Case 2 (`E2E Fold 0` in `0.810 – 0.844`):** Do NOT stop early—hand off the cleaned Track B (`SFT V3 + Step-DPO`) checkpoint directly as the initialization for **Track C (`Multi-Turn SafeEnv GSPO/GiGPO RL + SDPO/OPD` on `folds 1..4`)** to push unseen `Fold 0` past `0.855–0.870+` BEFORE full `1,246` consolidation & Dev 299 evaluation!

- [ ] **Step B6.2: Final Verification & Experiment Logging (Whenever Sealed Dev 299 Runs)**

1. Check total parameters via `scripts/count_params.py` (`LLM < 5.0B`, `PhoWhisper + LLM < 6.0B`).
2. Run 2x deterministic repeatability check (`temperature=0.0`, identical `tool_calls` & `state_hash` across two passes).
3. Verify ALL HF checkpoints + pure-text merged artifacts are present on `htphong0909/zalo-aic-sft-checkpoints` BEFORE stopping Colab.
4. Run full local test suite: `python -m pytest -q`.
5. Write complete `experiments/<exp_id>/REPORT.md` (with verified HF links, F1/SM/act/ask/oos/history breakdown, per-sample diagnostics on Fold 0) and update `reports/LLM_AGENT_EXPERIMENTS_DASHBOARD.md`.

---

## Plan Self-Review Checklist
- [x] Grounded in 100% of Fold 0 (`N=250`) imperfect samples (`23` `SM=True, F1<1.0` + `20` `SM=False` + `7` E2E drops) and exact `metrics.calls_match` / `token_f1 > 0.6` rules.
- [x] Every technique cites and adapts verified local code + paper repo implementations (`07_LoopTool_ACL2026`, `03_Hammer_ICLR2025`, `06_xLAM_APIGen-MT_NeurIPS2025`, `08_When2Call_NAACL2025`, `09_ToolSandbox_NAACL2025`, `10_HammerBench_ACL2025`, `11_TinyAgent_EMNLP2024`, `12_AURA_ASRU2025`, `13_MiniCPM`, `01_Tool-N1_ICLR2026`, `16_Step-DPO_2024_2025`).
- [x] Zero `dev_v1` (`299`) and zero `<split>_environment/` access anywhere in data synthesis, GCP mining, or validation.
- [x] Strict hardware discipline enforced: **G4 = Training only**, **L4 = ASR & E2E vLLM evaluation only**.
- [x] HF durable checkpoint sync (`checkpoint-<step>` + `checkpoints/latest` auto-resume + final pre-stop Hub verification) included in every remote job.
