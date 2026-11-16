# Qwen3.5-4B (Pure-Text) + PhoWhisper 85%+ Accuracy Master Plan

> **For agentic workers:** Execute track-by-track in strict sequence. Do NOT skip gates or train/tune on sealed `dev_v1` (`299` samples). Keep `AGENTS.md` §2 & §4.3 invariants active at every commit.

**Goal:** Push **Oracle Final Composite Score** on Sealed Dev (`299` samples) from **`0.7788` → `≥ 0.87`** (`State Match ≥ 91%`, `Tool F1 ≥ 89%`) AND push **End-to-End Voice Pipeline** (`.wav → PhoWhisper → Qwen3.5-4B → SafeEnv`) to **`≥ 85.0%+` Final Score**, strictly under `AGENTS.md` limits (`LLM < 5B`, `ASR + LLM < 6B`, `max_model_len ≤ 16384`, `latency ≤ 30s/sample` on A30 24GB, offline Docker, deterministic `temperature=0.0`).

**Architecture:** Fix two structural flaws first — (1) train/val leakage in local benchmarking (`fold_0` was inside `qwen35_train_sft_v2.jsonl`, hiding an `-8.59 pt` generalization gap between `0.8647` Fold 0 and `0.7788` Dev299) and (2) zero post-SFT alignment / ungrounded `enable_thinking: false` — then execute a 4-track stack synthesized from `LoopTool`, `Tool-N1`, `ToolRL`, `GiGPO`, `GSPO`, `ARPO`, `SDPO`, `OPD`, and `AURA`:
- **Track A (Deterministic Runtime & ASR-Diff Wiring):** Schema/XGrammar-guarded decoding, `BoundaryVerifier.should_ask_clarification` integration, read-before-write/ask consistency guards, `asr_phonetic_diff` wiring (`run_asr.py` → `run_agent.py`), `max_repairs=3`, and fuzzy-field/optional-arg discipline.
- **Track B (Clean CV + LoopTool Data Engine V3 + Hybrid `<think>` SFT V3):** Strict held-out `fold_0` (`250`) vs `folds 1..4` (`996`) clean gate; `GCP` hard-sample mining + `JGLV` target auditing + `EDDE` targeted expansion; selective concise 4-slot `<think>` supervision in `src/data_gen/distill_think.py` + `src/agent/qwen35_sft.py` + real Champion PhoWhisper Loop #38 transcripts (`0.82%` train WER) & N-best phonetic diffs.
- **Track C (Multi-Turn Verifiable RL + Self/On-Policy Distillation):** Replace single-step TRL `GRPOTrainer` with multi-turn `SafeEnv` rollout optimization directly targeting exact competition score `f1 if state_match else 0.0` (`src/eval/metrics.py:score_sample`) + `ToolRL` bipartite shaping + `GiGPO` AnchorState step advantages + `GSPO` sequence-level importance ratios & `DAPO Clip-Higher (0.28)` + `ARPO` post-tool entropy-aware branching on non-trivial `LoopTool` `pass@G ∈ (0, 1)` samples, followed by **`SDPO`/`OPD` Reverse-KL On-Policy Distillation** (`MiniCPM` / Hübotter et al. 2026) using environment error tracebacks / teacher logits to recover precision on the `51%` `state_match=True, F1<1.0` failure mass.
- **Track D (E2E Voice Calibration, Final 1,246 Retrain & Verification):** Joint AURA N-best beam calibration (`num_beams=3` for `asr_phonetic_diff`), full `train_only` (`1,246`) recipe consolidation once clean `fold_0` clears gate, A30/G4 latency/determinism/size audit, and full local + HF verification.

**Tech Stack:** `Qwen/Qwen3.5-4B` (pure text `language_model_only=True` ~`4.21B`), `kzap201/phowhisper-large-zaloai` (~`1.54B`, Loop #38 frontend, combined ~`5.75B < 6B`), `vLLM` (with `XGrammar` / structured logits backend + `LoRARequest`), `PyTorch 2.6+`, `PEFT`, `TRL`, `huggingface_hub` (`HFSyncer`), `zaic_tools.ToolEnv`, Colab `G4` (`L4` fallback) via `scripts/colab_runner.py`.

---

## 1. Empirical & Literature Diagnosis (Why Dev299 is `0.7788` and How to Reach `0.85+`)

### Verified Current State (No Dev Per-Sample Leakage)
- **Oracle SFT V2 (`experiments/exp_qwen35_sft_v2/eval_dev299/metrics.json`):** `Final Score = 0.7788` (`n=299`), `State Match Rate = 83.45%`, `Tool-Call F1 = 82.65%` (`P=82.42%`, `R=83.70%`), `OOS = 100.00%` (`n=21`), `Act = 75.87%` (`n=235`), `Ask = 78.04%` (`n=43`), `History = 77.73%` (`n=197`), `No-History = 78.16%` (`n=102`). Latency on Colab G4: `328.73s` (`1.10s/sample` vs `30.0s` budget).
- **Critical Train/Val Hygiene Discovery:** `data/sft/qwen35_train_sft_v2.jsonl` (`4,954` rows) spans all `1,246` `train_only` samples (`250/250` overlap with `fold_0`, `0/299` overlap with `dev_v1`). Consequently, `Fold 0` (`0.8647` Final, `90.48%` SM, `89.91%` F1) reflects **in-sample recall**, whereas `Dev 299` (`0.7788`) reflects **out-of-sample generalization** (`-8.59 pts` gap). We MUST evaluate development ablations on a **clean held-out `fold_0` (`250` samples) strictly excluded from training (`folds 1..4 = 996` samples)** before running the final `1,246` full-`train_only` pipeline onto `Dev 299`.
- **Root Failure Breakdown (Dev299 Summary + Fold 0 `49/250` Non-Perfect Inspection):**
  1. **`State Match = True` but `F1 < 1.0` (~5.57 pts lost on Dev299, `25/49 = 51%` of Fold 0 errors, mean F1 `0.6069`):** `src/eval/metrics.py:calls_match` strictly requires **`set(pred_args) == set(gold_args)`** and **`token_f1 > 0.6`** on `x-match: "fuzzy"` fields (`32` tools). Pure SFT frequently adds unrequested optional filters (`lottery.get_results(date=today)`, `contacts.search(query=...)` when `{}` was gold), drops requested optional fields (`weather.*(location=...)`, `ewallet.transfer(note=...)`, `calendar.create_event(end=...)`), adds extra redundant read calls (`chat_ott.list_chats`, `robot_vacuum.get_status`), or trims/rewords fuzzy strings (`reminders.create.content`, `notes.create.{title,content}`, `sms.send.content`) past `0.60` `token_f1`. SFT cross-entropy has zero training gradient for `set(p) == set(g)` or `token_f1 > 0.6` thresholds!
  2. **Pure No-Think Errors (`enable_thinking: false`) — Arithmetic, Unit Conversion, Slot Extraction & Conditional Branches:** With `max_new_tokens=384` and no `<think>`, the model jumps directly into XML tool args, causing off-by-one dates, wrong subtraction/percentages (`train_0814`, `1115`), misread tool return numbers (`train_1433` `3.2` vs `3.4`), or wrong account/ID selection (`train_0279`) even while runtime latency sits at only `1.10s/sample` (`>25s/sample` unused compute budget).
  3. **Premature Ask-Without-Lookup & Over-Ask / Alias Fabrication (`7` `over_ask` + `2` `under_ask` + ask-without-read on Fold 0; `ask = 78.04%` on Dev299):** Calls `sys.ask_missing_info` for `note_id` / `scene_id` without first calling `notes.list` / `smart_home.get_status`, OR guesses `phone.call(contact_id=...)` / misses missing-param boundaries because `BoundaryVerifier.should_ask_clarification` in `src/agent/boundary_verifier.py` is never called by `src/agent/react_loop.py`.
  4. **Syntax / Schema Aborts & Repair Budget:** Even on seen `Fold 0`, `11` repair triggers occurred (`10` in failed samples, `3` ending in `schema_abort` at `max_repairs=2`) — fixable deterministically via grammar/schema-guided decoding and informative validation feedback.
  5. **E2E Voice Pipeline Gap:** Champion ASR Loop #38 (`kzap201/phowhisper-large-zaloai`, `1.50%` Dev WER, `0.36s/sample`) is completely decoupled from LLM fine-tuning: `scripts/run_asr.py` emits only 1-best transcripts without N-best beams (`aura_nbest.py`), `scripts/run_agent.py` never passes `asr_phonetic_diff` to `run_react_loop()`, and `qwen35_train_sft_v2` was built with synthetic CTC character jitter rather than real PhoWhisper transcripts & N-best confusion sets.

### What Cloned Paper Repos & 2025–2026 Literature Prescribe
1. **`LoopTool` (`references/tool_calling_repos/07_LoopTool_ACL2026`):** Closed **GCP** (greedy inference + token logprob difficulty) → **JGLV** (rule validator + label verification) → **EDDE** (targeted variation expansion on verified failure clusters) data loop; `DAPO Clip-Higher` (`clip_ratio_low=0.2, clip_ratio_high=0.28`), `use_kl_loss=False` in pure verifiable RL, `entropy_coeff=0`, `lr=1e-6`, `pass@k` filtering (`0 < pass@G < 1`) so gradient updates focus on learnable boundary/arg errors.
2. **`Tool-N1` (`01_Tool-N1_ICLR2026`) & `ToolRL` (`04_ToolRL_NeurIPS2025`):** Binary outcome correctness alone is sparse for long chains; combine **strict outcome verification** (our exact `score_sample`) with **`ToolRL` bipartite arg/name partial shaping** only below the exact-match threshold (strictly capped `< 0.55` when `state_match=False` and `< 0.90` when `calls_match` fails) so the policy never hacks partial credit over `score=1.0`.
3. **`GiGPO` (`05_GiGPO_verl-agent_NeurIPS2025`) & `ARPO` (ICLR 2026):** Multi-turn tool chains need **per-step credit assignment** (`GiGPO`: episode outcome + `AnchorState` step-group advantage) and **branching rollouts right after high-entropy tool observations** (`ARPO`) instead of uniform open-loop generation.
4. **`GSPO` (`arXiv:2507.18071`):** Sequence-level length-normalized importance ratio $s_i(\theta) = (\pi_\theta(y_i|x)/\pi_{\theta_{old}}(y_i|x))^{1/|y_i|}$ eliminates high-variance token-ratio clipping spikes on XML/JSON tool syntax and mixed-length `<think>` rollouts.
5. **`JustRL` + `OPD` (`13_MiniCPM/README.md`, `arXiv:2604.13016`) + `SDPO` (Hübotter et al. 2026):** After SFT + RL, **On-Policy Distillation (reverse KL on student rollouts)** and **Self-Distillation with Rich Environment Feedback** (conditioning a frozen/EMA teacher on `SafeEnv` validation tracebacks + gold schema diffs in-context to distill token-level log-probs into the student trajectory) recover SFT capabilities (fuzzy natural phrasing, `ask` calibration) while preserving RL state-match gains.
6. **`DiaTool-DPO` (`14_DiaTool-DPO_SIGDIAL2025`) & `AURA` (`12_AURA_ASRU2025`):** Turn-discounted trajectory optimization (`psi_c` normalization when pred and gold have unequal step counts, e.g. missing/extra `contacts.search` or `notes.list`) + N-best ASR phonetic-diff prompting (`src/asr/aura_nbest.py`) so homophone errors in Vietnamese names/numbers are disambiguated inside LLM context using top-`K` Whisper beams.

---

## 2. Global Execution Rules & Anti-Leakage Discipline

1. **Strict Split Hygiene (Zero Dev Leakage):**
   - `configs/splits/dev_v1.json` (`299` IDs) is **STRICTLY READ-ONLY / SEALED FOR AGGREGATE BENCHMARKING ONLY**. Never mine errors, synthesize examples, inspect `per_sample.jsonl`, or tune thresholds on `dev_v1`.
   - Create `configs/splits/train_folds_1_to_4.json` (`996` IDs = `train_only \ fold_0`) and hold out **`fold_0` (`250` IDs)** as the **clean local generalization gate** across Track A/B/C/D development.
   - Only after the full recipe beats baseline on clean held-out `fold_0`, re-run the exact scripted pipeline on all `1,246` `train_only` IDs (`SFT V3 Full -> RL Full -> OPD Full`) and evaluate once per milestone on `dev_v1`.
2. **Colab G4 & Durable HF Checkpoints (`AGENTS.md` §4.3 & `.agents/skills/checkpoint-sync-hf`):**
   - Before every GPU run: verify `HF_TOKEN` & `HF_SYNC_REPO` via `HFSyncer(...).preflight()`.
   - Every training step saves to `<prefix>/checkpoints/checkpoint-<step>/` (never overwriting prior steps, keeping `latest/` updated every `≤ 10` minutes), with automatic download + resume from `<prefix>/checkpoints/latest/`.
   - Keep blocking `colab exec` calls under `45` minutes (use detached `nohup` workers + `check`/`ensure` orchestrator loops); always verify HF repo contents before `colab stop`.
3. **Hard Inference Contract:**
   - All inference modules must preserve **deterministic `temperature = 0.0`**, **no access to `<split>_environment/` or `state.json` outside `SafeEnv` (`validate`, `call`, `get_tool_calls`, `get_state_hash`)**, **`max_model_len ≤ 16384`**, **`total_params < 6B`**, **`avg_latency ≤ 30s/sample`** (target budget: `≤ 4.0s/sample` total E2E!).

---

## 3. Track A — Deterministic Runtime & ASR-Diff Hardening (Fast Win, Zero Retrain Required)

**Expected Impact:** `+2.0 to +3.5` pts on unseen Oracle Dev299 (closing syntax aborts, ask-without-lookup misses, ambiguity guesses, and ASR phonetic wiring before any new GPU training).
**Latency Budget:** `< +0.15s/sample` (`1.10s → ~1.25s/sample`).

### Task A1: Wire `BoundaryVerifier.should_ask_clarification` & Read-Before-Ask Consistency into `react_loop.py`
**Files:**
- Modify: `src/agent/boundary_verifier.py`
- Modify: `src/agent/react_loop.py`
- Modify: `tests/test_boundary_verifier.py` & `tests/test_react_loop.py`

**Step 1: Write failing unit tests first**
- Add tests to `tests/test_boundary_verifier.py` covering:
  1. Duplicate contact/device/note names retrieved in tool output → `should_ask_clarification(...)` returns `True` when caller attempts a write call (`phone.call`, `sms.send`, `ewallet.transfer`, `notes.append`, `smart_home.scene_run`) without disambiguating attribute.
  2. Read-before-Ask prerequisite check: when user references a stateful target (`notes.append` → `notes.list`, `smart_home.scene_run` → `smart_home.get_status`, `alarm.update`/`delete` → `alarm.list`) and that read tool is in `tools` and NOT in `seen_read_calls` or `history_reads`, verify helper flags the missing lookup step cleanly.
- Run `python -m pytest -q tests/test_boundary_verifier.py tests/test_react_loop.py` → confirm new tests FAIL.

**Step 2: Implement deterministic runtime improvements**
- In `src/agent/boundary_verifier.py`:
  - Expand `BoundaryVerifier` (`__init__(self, catalog=None)`) with `find_missing_prereq_read(call: ToolCall, seen_reads: set[tuple[str, str]], available_tools: set[str]) -> ToolCall | None` for well-defined read-before-write/ask prerequisites derived strictly from tool schemas (`notes.append`/`update`/`delete` → `notes.list`, `smart_home.scene_run`/`light_set`/`ac_set`/`switch_set` needing `device_id`/`scene_id` → `smart_home.get_status`, `calendar.update_event`/`cancel_event` → `calendar.list_events`, `alarm.update`/`delete` → `alarm.list`, `tasks.complete`/`delete` → `tasks.list`).
  - Strengthen `verify_optional_args(...)` using schema defaults (`default` / `bỏ trống = ...` in catalog descriptions): e.g., when user gives NO explicit date string and `lottery.get_results` receives `date == current_date`, only strip `date` if utterance contains `"hôm nay"` / `"chiều nay"` / no past date token (never strip explicit past dates like `"hôm qua"` / `"ngày 5"`). Keep every check purely utterance/profile grounded per docstring.
- In `src/agent/react_loop.py`:
  - Cache decoded JSON lists of retrieved entities from `contacts.search`, `notes.list`, `smart_home.get_status`, `calendar.list_events`, `chat_ott.list_chats` inside `run_react_loop`.
  - Wire `boundary_verifier.should_ask_clarification(...)` inside the preflight/repair gate: if a mutating/sending call picks an entity from `>= 2` ambiguous retrieved matches with identical names (`len(set(names)) < len(names)`) and no distinguishing ID/phone/email/tag in user text, convert or repair into `sys.ask_missing_info` with exact `[tool_name, required_id_param]`.
  - Wire read-before-ask / repair feedback: if the model emits `sys.ask_missing_info` for an ID param (`note_id`, `scene_id`, `device_id`, `event_id`, `alarm_id`) while the corresponding prerequisite `read` tool has NOT yet been executed in this turn or `history`, inject the prerequisite `read` step automatically (or trigger 1 repair turn if `repairs < max_repairs`) so `SafeEnv` tool results inspect actual state first!
  - Bump default `max_repairs` in runtime configs (`oracle_*.yaml`, `e2e_*.yaml`) from `2 → 3` and ensure `schema_repair` message lists the exact allowed tool parameter names & required fields from `catalog[c.name]` so 3/3 `schema_abort`s convert into valid tool calls.

**Step 3: Verify tests**
- Run `python -m pytest -q tests/test_boundary_verifier.py tests/test_react_loop.py tests/test_collection_xref_guard.py` → expect ALL PASS.

---

### Task A2: Schema-Constrained / XGrammar Guided Decoding in `VLLMBackend`
**Files:**
- Modify: `src/agent/llm_client.py`
- Modify: `src/core/config.py` (add `guided_decoding: bool = False` option on `LLMConfig`, enabled selectively on retries or strict tool schema mode)
- Modify: `tests/test_llm_client.py`

**Step 1: Write failing unit tests first**
- Add unit tests in `tests/test_llm_client.py` verifying:
  1. `build_tool_call_json_schema(tools)` generates a valid Draft-7 JSON schema covering `<tool_call>` grammar / parameter types & `required` fields from `tools`.
  2. When `VLLMBackend` detects vLLM's `GuidedDecodingParams` / `StructuredOutputsParams` (mocked cleanly in unit test), retry generation on `malformed_repair` or `schema_repair` switches seamlessly to constrained decoding (guaranteeing zero malformed JSON/XML syntax on repair steps while leaving `<think>...</think>` unconstrained on step 0 when free text/OOS is allowed).

**Step 2: Implement compatibility-safe guided fallback in `src/agent/llm_client.py`**
- Add `generate(messages, tools, *, force_tool_schema: bool = False)` optional flag on `VLLMBackend` (default `False` so step-0 `<think>` & free OOS `[]` remain unrestricted), called by `react_loop.py` strictly on `malformed_repair` / `schema_repair` turns when `has_malformed_tool_call` fired. Use `xgrammar` backend when supported by installed vLLM (`StructuredOutputsParams` / `GuidedDecodingParams`), falling back cleanly if unavailable.

**Step 3: Verify tests**
- Run `python -m pytest -q tests/test_llm_client.py tests/test_config.py` → expect ALL PASS.

---

### Task A3: Enable Top-K N-Best Beam Extraction & Wire `asr_phonetic_diff` End-to-End
**Files:**
- Modify: `src/asr/base.py` (add `nbest: list[str] = field(default_factory=list)` to `ASRResult`)
- Modify: `src/asr/whisper_engine.py` (`WhisperASREngine.transcribe(..., num_beams_nbest: int = 1)` & `transcribe_batch`)
- Modify: `scripts/run_asr.py` (add `--nbest-beams 3`, emit `"nbest"` + `"asr_phonetic_diff"` per sample using `src.asr.aura_nbest.extract_phonetic_diff`)
- Modify: `scripts/run_agent.py` (read `"asr_phonetic_diff"` from transcript JSONL rows and pass `asr_phonetic_diff=...` into `run_react_loop(...)`)
- Modify: `tests/test_asr.py`, `tests/test_aura_nbest_diff.py`, `tests/test_run_agent.py`

**Step 1: Write failing unit tests first**
- Test that `WhisperASREngine` (mock model) returns sanitized N-best candidate strings (`ASRResult.nbest`) when `num_beams > 1`, preserves identical greedy 1-best (`ASRResult.text`) when `num_beams == 1`, and `scripts/run_agent.py` forwards `asr_phonetic_diff` from transcript rows to `run_react_loop()`.
- Run `python -m pytest -q tests/test_asr.py tests/test_aura_nbest_diff.py tests/test_run_agent.py` → confirm FAIL before edits.

**Step 2: Implement confidence-gated / fast N-best wiring**
- Note: ASR Loop #38 (`AGENTS` dashboard) showed greedy `num_beams=1` had better 1-best WER (`1.50%`) than 2-beam search (`1.61%`) and took `0.36s/sample`. Therefore:
  - Always keep **1-best `text` strictly greedy (`do_sample=False, num_beams=1`)** to preserve Loop #38's `1.50% WER`.
  - Add an optional secondary fast 3-beam pass **only when mean sequence token log-probability `< threshold` (or toggled via `--nbest-beams 3` batched)** inside `run_asr.py`, calling `extract_phonetic_diff([greedy_text] + nbest_texts)` (`src/asr/aura_nbest.py`) to compute `asr_phonetic_diff` strictly without touching greedy 1-best text! Total ASR latency remains `< 0.7s/sample`.
- Update `scripts/run_agent.py:168-189` to extract `asr_phonetic_diff = t_entry.get("asr_phonetic_diff")` and pass `asr_phonetic_diff=asr_phonetic_diff` to `run_react_loop(...)`.

**Step 3: Verify tests**
- Run `python -m pytest -q tests/test_asr.py tests/test_aura_nbest_diff.py tests/test_run_agent.py` → ALL PASS.
- Benchmark runtime-only **Track A** on `exp_qwen35_sft_v2` adapter (`Fold 0` & `Dev 299` via `run_agent.py` + `run_eval.py`) to isolate pure Track A delta before starting training.
- **Exit Gate A:** Zero `schema_abort` on Fold 0; zero `malformed` unrecovered errors; `run_agent.py` passes `asr_phonetic_diff` cleanly; Dev299 improves over `0.7788` baseline at `latency ≤ 2.0s/sample`.

---

## 4. Track B — Clean CV Split, LoopTool Data Engine V3 & Hybrid `<think>` SFT V3

**Expected Impact:** `+3.5 to +5.5` pts on unseen general accuracy by fixing (1) arithmetic/slot/conditional mistakes via compact `<think>`, (2) systematic failure clusters mined *strictly* on `train_only`, and (3) real Champion PhoWhisper ASR transcripts + `asr_phonetic_diff` exposure during SFT.
**Latency Budget:** Compact 4-slot `<think>` (`40–95` tokens/step) adds ~`1.0–1.5s/sample` on vLLM G4 (`total ~2.5s/sample ≪ 30s`).

### Task B1: Create Clean Held-Out Development Protocol (`folds_1_to_4` Train / `fold_0` Clean Val)
**Files:**
- Create: `configs/splits/train_folds_1_to_4.json` & `configs/splits/fold_0.json` (derived cleanly from existing `configs/splits/train_only_5folds.json`)
- Create/Update: `scripts/make_folds.py` (optional flag `--export-eval-subsets`)
- Add test: `tests/test_folds_no_leak.py` (assert `dev_v1 ∩ fold_0 == ∅`, `fold_0 ∩ folds_1_to_4 == ∅`, `len(fold_0) == 250`, `len(folds_1_to_4) == 996`)

**Step 1:** Export `train_folds_1_to_4.json` (`996` IDs) and `fold_0.json` (`250` IDs) and run `python -m pytest -q tests/test_folds_no_leak.py`. **Never mine errors from `dev_v1` at any point.**

---

### Task B2: Upgrade `src/data_gen/distill_think.py` + Selective Hybrid `<think>` in `src/agent/qwen35_sft.py`
**Why:** `qwen35_sft_v2.yaml` set `enable_thinking: false` globally, causing numerical/slot/conditional misses (`train_0224, 0279, 0777, 0802, 0814, 0921, 1055, 1115, 1326, 1433`). Meanwhile, `qwen35_sft.py` & `distill_think.py` already implement 4-slot micro-`<think>` (`< 95` tokens)!
**Improvement needed in `distill_think.py`:** Currently `synthesize_deterministic_think` copies scalar facts like `balance`, `amount`, `price` but does **not**:
1. Extract numeric values / travel times / percentages / item quantities from user & tool text to explicitly record **arithmetic verification** (e.g., `08:00 - 12p = 07:48`, `425k - 350k = 75k`, route `distance=3.4 km`).
2. Explicitly check **optional argument minimalism vs requirement** (e.g., *"Người dùng không nêu ngày cụ thể -> dùng mặc định, KHÔNG truyền `date` thừa"* / *"Thiếu tên đường/ID -> giữ nguyên chuỗi `address` từ kết quả tool"*).
3. Explicitly check **fuzzy content completeness** (`x-match: "fuzzy"` keywords preserved from user request + tool result: recipient title, amount, reason/topic, transaction code).
4. **Selective / Hybrid `<think>` Strategy (`LoopTool` / `Qwen3.5` Best Practice):** Instead of burning 80 think tokens on trivial OOS/final_stop turns, train the model cleanly with `enable_thinking: true` using:
   - Compact **4-slot micro-`<think>`** (`45–90` tokens) on tool-calling / conditional / arithmetic / recovery steps, AND
   - Ultra-short `<think>\nRanh giới: ...</think>` (`10–20` tokens) on OOS / final confirmation turns, keeping inference p95 total tokens fast (`≤ 2.5s/sample`).

**Files:**
- Modify: `src/data_gen/distill_think.py` (add numeric/time/comparison grounding, optional-arg minimalism rule, fuzzy keyword preservation check in slots 2–3)
- Modify: `src/agent/qwen35_sft.py` (support `compact_stop_think: bool = True` + clean handling of `asr_phonetic_diff` hints in user/system prompt)
- Modify: `tests/test_distill_think.py` & `tests/test_qwen35_sft.py`

**Step 1:** Write tests in `tests/test_distill_think.py` ensuring generated `<think>` stays `<= 120` words, captures tool observations (`balance`, `amount`, `distance_km`, list length/names, ambiguity), preserves required keywords, and never leaks future/post-step info.
**Step 2:** Run `python -m pytest -q tests/test_distill_think.py tests/test_qwen35_sft.py` → PASS.

---

### Task B3: Implement `LoopTool` Closed Data Engine (`GCP` + `JGLV` + `EDDE`) + Real Champion ASR Transcripts
**Files:**
- Create: `scripts/run_looptool_data_engine.py` (implements LoopTool's 3 stages adapted cleanly to offline `SafeEnv` without requiring external GPT APIs!)
  1. **Stage 1 — `GCP` (`Greedy Capability Probing`, cf. `07_LoopTool_ACL2026/dataloop/greedy_capability_prob.py`):** Run greedy inference of `exp_qwen35_sft_v2` (and a clean `folds 1..4` SFT checkpoint for Fold-0-clean CV) on **`train_only` / `folds 1..4`** ONLY, capturing `pred_calls`, `state_hash`, exact `score_sample`, `f1`, `state_match`, and sequence mean log-probability (`cum_logprob / num_tokens`). Partition failures into:
     - `H1`: `state_match == True, f1 < 1.0` (fuzzy/optional-arg/redundant-read errors)
     - `H2`: `state_match == False` arithmetic/slot/sequence errors
     - `H3`: `ask` vs `act` boundary errors & lookup-before-ask errors
  2. **Stage 2 — `JGLV` (`Verification`, cf. `judgement_label_verification.py`):** Verify ALL synthetic candidates (`apigen_mt`, `anchor_binding`, `when2call`, `recovery`) against `SafeEnv` (`validate()` + `call()` + `get_state_hash()`) + `replay_gold` schema checks; discard any synthetic example with inconsistent state/schema or ambiguous fuzzy target.
  3. **Stage 3 — `EDDE` (`Error-Driven Data Expansion`, cf. `error_data_expansion.py`):** Using deterministic verified offline augmentation + `apigen_mt_gen.py` / `when2call_gen.py` / `anchor_binding_gen.py` / `recovery_augmenter.py`, generate targeted variations ONLY from failed `folds 1..4` templates (`SCENARIO_VARIATIONS`: time-offset arithmetic, multi-item lists, fuzzy content keyword preservation, lookup-before-ask `notes.list` / `smart_home.get_status` / `alarm.list`, empty `{}` vs filtered `contacts.search`, default `date`/`location` omission). Upweight hardest verified failures (`2x–3x`) cleanly.
  4. **Stage 4 — Real Champion ASR + N-Best Injection:** Run ASR Loop #38 (`scripts/run_asr.py --nbest-beams 3`) on `train_only` audio clips (including `history` turn audios) to produce `data/asr/train_only_phowhisper_loop38_nbest.jsonl`. Update `scripts/build_qwen38_sft_data.py` to ingest BOTH `--asr-transcripts` AND `--asr-nbest` (`asr_phonetic_diff` hints) for `~30%` of SFT trajectories (preserving oracle clean text on `~70%`), so SFT learns both Oracle precision and real PhoWhisper acoustic disambiguation.

- Modify: `scripts/build_qwen38_sft_data.py` (cleanly add `--include-ids`, `--exclude-ids`, `--enable-thinking`, `--asr-transcripts` N-best wiring, `--edde-jsonl`, default output `qwen35_train_sft_v3*.jsonl.gz`)
- Create: `configs/qwen35_sft_v3_clean.yaml` (`folds 1..4`, `enable_thinking: true`, `lr: 4e-5`, `epochs: 2.0`, `max_length: 16384`, `lora r=32, alpha=64`, `save_steps: 40`, `hf_interval_min: 8.0`) & `configs/qwen35_sft_v3_full.yaml` (all `1,246` `train_only` once clean gate passes).
- Create/Update Test: `tests/test_looptool_data_engine.py` verifying zero overlap with `dev_v1` / `fold_0`, valid `SafeEnv` replay on every EDDE row, and `<think>` format integrity.

**Step 1:** Run local tests: `python -m pytest -q tests/test_looptool_data_engine.py tests/test_qwen35_sft.py`.
**Step 2:** Execute `run_asr.py` (Loop #38 + `--nbest-beams 3`) on `train_only` (`1,246` samples, ~8 min on Colab G4) and sync `data/asr/train_only_phowhisper_loop38_nbest.jsonl` to HF (`HFSyncer`).
**Step 3:** Build `qwen35_sft_v3_clean` (`folds 1..4` only, `fold_0` strictly excluded) + run SFT V3 Clean on Colab G4 with detached runner & `checkpoint-<step>` / `latest` HF sync (`scripts/colab_qwen35_sft.py`).
- **Exit Gate B (Measured on CLEAN unseen `Fold 0` = `250` samples, zero leakage):**
  - `SFT V3 Clean (folds 1..4)` on unseen `Fold 0` MUST achieve **`Final Score ≥ 0.810`** (vs inferred unseen baseline `~0.779`) and **`State Match ≥ 86.5%`**, with zero `schema_abort`s and `avg_latency ≤ 3.5s/sample`.

---

## 5. Track C — Multi-Turn Verifiable RL (`Tool-N1` + `ToolRL` + `GSPO`/`GiGPO`/`ARPO`) & `SDPO`/`OPD` Distillation

**Why Track C is the Decisive Jump to `85+`:**
SFT V3 establishes accurate reasoning traces (`<think>`) and fixes missing failure patterns, **yet SFT cross-entropy still treats all tokens uniformly**—it cannot optimize `score_sample` non-differentiable cliffs (`set(pred_args) == set(gold_args)`, `token_f1 > 0.6`, exact `state_hash`, non-redundant read calls). Paper evidence from `Tool-N1` (ICLR 2026), `LoopTool` (ACL 2026), `GiGPO` (NeurIPS 2025), `ARPO` (ICLR 2026), `GSPO` (`arXiv:2507.18071`), and `MiniCPM + OPD` / `SDPO` proves **Verifiable Multi-Turn RL + On-Policy Distillation** adds `+5 to +10` points over strong SFT!

**Expected Impact:** `+4.0 to +6.5` pts on unseen general accuracy (`~0.815–0.83` SFT V3 → `0.86–0.88+` Final Score).

### Task C1: Build Exact BTC-Aligned Trajectory + Step Verifier (`src/agent/rl_reward_v3.py`)
**Files:**
- Create: `src/agent/rl_reward_v3.py`
- Create: `tests/test_rl_reward_v3.py`

**Exact Reward Design (Grounded in `src/eval/metrics.py` + `01_Tool-N1` + `04_ToolRL` + `05_GiGPO` + `07_LoopTool`):**
1. **Primary Trajectory Reward (`R_traj ∈ [0.0, 1.0]`):** Reuses **`src/eval/metrics.score_sample(pred_calls, pred_hash, gold_calls, gold_hash, catalog)`** directly on completed multi-turn `SafeEnv` rollouts!
   - If `score_sample == 1.0` AND format `<think>...</think>` valid AND zero redundant invalid repairs: **`R = 1.0`** (`Tool-N1` strict positive).
   - If `state_match == True` AND `score_sample < 1.0` (`H1` error mass!): reward equals **`0.45 + 0.45 * f1` (strictly `< 0.90`, `≥ 0.45`)**, explicitly penalizing optional-arg mismatch (`set(p) != set(g)`), extra read calls (`-0.08/extra_read`), and `token_f1 <= 0.6` (`+0.15 * min_fuzzy_f1` dense shaping only below threshold). **This directly eliminates the `25/49` `sm_true_f1_low` error class!**
   - If `state_match == False`: compute `ToolRL` bipartite arg-name/value shaping `0.15 * name_f1 + 0.25 * arg_f1` strictly capped at **`≤ 0.40`** so wrong-state trajectories NEVER overtake correct-state ones (`0.40 < 0.45 < 1.0`).
   - If OOS (`gold_calls == []`): `1.0` iff `pred_calls == []`, else `-0.2` (discourages `oos_fp`).
   - If malformed XML / exceeded `max_new_tokens` without closing tag / hallucinated tool: **`0.0`**.
2. **Per-Step Anchor Advantage (`GiGPO`, `src/agent/lfm3_gigpo.py` reuse):**
   - For multi-turn steps sharing identical canonicalized `AnchorState` (user query + prior observation signature), add normalized step advantage `A_step(s_t)` weighted by `α = 0.5` to `A_episode` so turn-level errors (e.g. skipping `notes.list` on step 1 or wrong args on step 2) get localized gradient credit!

**Step 1:** Write unit tests in `tests/test_rl_reward_v3.py` verifying strict strict monotonic ordering: `OOS false-positive (-0.2) < Wrong-State partial (≤ 0.40) < State-Match F1<1.0 (0.45–0.89) < Exact Gold Match (1.0)`, plus fuzzy-field shaping monotonicity and `GiGPO` step grouping. Run `pytest` → PASS.

---

### Task C2: Multi-Turn `SafeEnv` Rollout Engine + `LoopTool` Difficulty Filtering + `ARPO`/`GSPO` Trainer
**Why not reuse `src/agent/lfm3_grpo.py` directly?** `lfm3_grpo.py` uses TRL `GRPOTrainer` on single-step static teacher-forced prompts (`prefix_calls` fixed) and token-level GRPO importance weights, NOT closed-loop multi-turn `SafeEnv` rollouts where step-1 mistakes change step-2 observations!
**Architecture (`src/agent/mt_rl_trainer.py` + `scripts/train_qwen35_rl.py`):**
1. **`LoopTool` Offline/Online `pass@G` Filtering (`filter_rl_prompts`):** Sample `G = 8` rollouts (`temperature=0.7` during exploration only!) from SFT V3 across `train_only` / `folds 1..4`. Retain non-trivial prompts where **`0 < pass@G < 1`** (plus `15%` hard `0/8` prompts paired with `prefix_calls` curriculum hints and `5%` anchor `8/8` stability prompts) — eliminating zero-variance compute waste cleanly.
2. **Closed-Loop Multi-Turn Rollouts in `SafeEnv` (`rollout_group` with vLLM `LoRA` fast weight reload or PyTorch batched KV rollout):**
   - Execute real `SafeEnv` tool calls (`env.validate` + `env.call`) up to `max_steps=8`, storing exact `(token_ids, action_mask, old_logprobs, env_tracebacks, step_anchors, R_traj)` per trajectory. (Action mask = `1` strictly on assistant generated tokens including `<think>` + XML `<tool_call>`; `0` on user/system/tool-observation tokens!).
   - **Optional `ARPO`-Style Post-Tool Entropy Branching:** Track token entropy right after tool observation injection on `state_match=False` trajectories; fork extra branch completions (`+2` rollouts) at highest-entropy decision step within same budget (`max_group=10`) when variance is low.
3. **`GSPO` Sequence-Level Objective + `DAPO` Clip-Higher + Zero KL Loss (`LoopTool` setting):**
   - Compute sequence-level length-normalized importance ratio over masked assistant tokens:
     $$s_i(\theta) = \exp\!\left(\frac{1}{|a_i|} \sum_{t \in a_i} \big(\log \pi_\theta(y_{i,t}|x_i, y_{i,<t}) - \log \pi_{old}(y_{i,t}|x_i, y_{i,<t})\big)\right)$$
   - Apply **asymmetric `DAPO Clip-Higher` (`ε_low = 0.20, ε_high = 0.28`)**, **`entropy_coeff = 0.0`**, **`kl_loss_coef = 0.0`** (or ultra-light `0.001` only if length drift occurs), `lr = 1e-6` cosine, `LoRA r=32, alpha=64` initialized directly from SFT V3 merged/adapter weights!
   - Optional curriculum (`DiaTool-DPO` / trajectory pair reserve): save all `(y_win=1.0, y_lose<1.0)` on-policy rollout pairs to disk (`rl_pairs.jsonl`) as free fuel for Task C3!

**Files:**
- Create: `src/agent/mt_rl_trainer.py` (clean batched multi-turn `SafeEnv` rollout collector, `GSPO` + `GiGPO` + `DAPO` loss computation, HF checkpoint sync hook every `save_steps=20`)
- Create: `scripts/train_qwen35_rl.py` (auto-resume from HF `<prefix>/checkpoints/latest/`, deterministic seed, saves every `checkpoint-<step>`)
- Create: `configs/qwen35_rl_clean.yaml` & `configs/qwen35_rl_full.yaml`
- Create: `tests/test_mt_rl_trainer.py` (unit tests for action masking `tool` tokens = 0, sequence-level ratio math `exp(mean_log_diff)`, DAPO asymmetric clipping bounds `(0.80, 1.28)`, and mock `SafeEnv` multi-turn rollout reward collection).

---

### Task C3: On-Policy Self-Distillation (`SDPO` + `MiniCPM5 OPD`) Recovery & Polish
**Why (`13_MiniCPM/README.md`, `arXiv:2604.13016`, Hübotter et al. 2026 `SDPO`):**
RL improves exact state-match & strict arg structure (`+5–6 pts`), yet pure outcome RL can sharpen probability mass away from rich natural Vietnamese phrasing (hurting `fuzzy token_f1 > 0.6` on emails/notes/reminders) or slightly drift tone/ask style. **OPD + SDPO** fixes this in `< 30` min on Colab G4 with **zero external teacher API**!
**Implementation (`src/agent/on_policy_distill.py` + `scripts/train_qwen35_opd.py`):**
1. Generate on-policy greedy/low-temp rollouts $y \sim \pi_{\theta}$ from the Stage C2 RL checkpoint on challenging `folds 1..4` / `train_only` prompts where RL got `score < 1.0` OR where `SFT V3` had higher fuzzy F1.
2. Construct **Self-Teacher Feedback Context (`SDPO`)** for the teacher forward pass (`SFT V3` or EMA teacher `π_teacher` frozen in bfloat16/4bit — uses same model family so logits share exact `Qwen3.5` tokenizer!): append `[Self-Verification Feedback: Gold schema constraints, SafeEnv error traceback, exact fuzzy keywords/state outcome]` inside a privileged teacher-only system prompt wrapper, and strip that privileged hint from the student prompt!
3. Compute **Top-K / Full-Vocab Reverse KL Divergence** $D_{\mathrm{KL}}(\pi_\theta(\cdot|x, y_{<t}) \,\|\, \pi_{teacher}(\cdot|x_{feedback}, y_{<t}))$ strictly over assistant tokens (`mask=1`), plus small `0.05 * SFT_NLL` anchor on verified `score=1.0` fuzzy demonstrations (`DiaTool`-length-normalized `psi_c` optional tie-breaker).
4. Evaluate checkpoint progression across RL steps and OPD steps on clean **`Fold 0`**; test uniform checkpoint souping (`scripts/merge_lfm3_soup.py` already in repo!) if adjacent checkpoints trade off `Act` vs `Ask` slightly.

**Files:**
- Create: `src/agent/on_policy_distill.py`
- Create: `scripts/train_qwen35_opd.py`
- Create: `configs/qwen35_opd_clean.yaml` & `configs/qwen35_opd_full.yaml`
- Create: `tests/test_on_policy_distill.py` (unit test verifying exact Reverse KL invariance when `student_logits == teacher_logits` → loss `0.0`, mask excludes prompt/tool tokens, and finite gradient flow).

- **Exit Gate C (Measured on CLEAN unseen `Fold 0` = `250` samples, Oracle text):**
  - `Final Score ≥ 0.855`, `State Match Rate ≥ 89.5%`, `Tool-Calling F1 ≥ 88.5%`, `sm_true_f1_low` error count cut by **`≥ 50%`**, `avg_latency ≤ 3.5s/sample`.

---

## 6. Track D — End-to-End Voice Calibration, Full `1,246` Pipeline & Final Competition Gate

**Goal:** Ensure zero accuracy cliff from clean text (`Oracle ≥ 0.87`) to spoken `.wav` audio (**E2E Voice Final Score `≥ 85.0%+`**) using Champion ASR Loop #38 (`1.50% WER`) + AURA N-best phonetic diff + `Qwen3.5-4B` RL+OPD.

### Task D1: End-to-End Voice Robustness Audit on Clean `Fold 0` Audio
**Files:**
- Modify/Verify: `configs/e2e_phowhisper_qwen35_4b.yaml` (point `adapter_path` / merged model to Track C full-pipeline output, set `enable_thinking: true`, `max_new_tokens: 512`, `max_repairs: 3`, Loop #38 whisper settings + `nbest_beams: 3`)
- Create: `scripts/eval_e2e_pipeline.py` (single command running `scripts/run_asr.py` → `scripts/run_agent.py` → `scripts/run_eval.py` cleanly + checking latency breakdown `asr_latency + agent_latency ≤ 30s/sample`)

**Step 1:** Run E2E evaluation on **`Fold 0` (`250` `.wav` files)** comparing:
- Clean Oracle Fold 0 vs PhoWhisper 1-best vs PhoWhisper + AURA `asr_phonetic_diff` (`nbest_beams=3`).
**Step 2:** Inspect any ASR-induced tool errors **strictly on `Fold 0`** (never Dev299!). If homophones / entity spellings (e.g., proper names, amounts, abbreviations) cause `> 2.0 pts` drop between Oracle and E2E Voice on `Fold 0`:
- Add **Voice-Specific Preference/OPD Pairs (`folds 1..4` real Whisper transcripts)** pairing `[Whisper noisy transcript + asr_phonetic_diff → gold calls]` with `Teacher = Oracle clean transcript logits` inside `on_policy_distill.py`! Notice how clean that `AURA + OPD` synergy is: distilling `π(y | clean_text)` into `π(y | asr_text + phonetic_diff)` trains the LLM directly on real Whisper failure patterns without corrupting clean text accuracy!

---

### Task D2: Full `1,246` `train_only` Consolidation Run & Sealed Dev299 Verification
Once **ALL** Track A/B/C/D Exit Gates pass on clean held-out `Fold 0`:
1. Execute the **identical frozen hyperparameters & scripts** on all `1,246` `train_only` samples (`dev_v1` strictly excluded):
   - Build `data/sft/qwen35_train_sft_v3_full.jsonl.gz` (with `train_only` LoopTool EDDE + `train_only` real PhoWhisper N-best ASR + 4-slot `<think>`)
   - Train `exp_qwen35_sft_v3_full` → `exp_qwen35_rl_v3_full` → `exp_qwen35_opd_v3_full` (all durable HF `checkpoint-<step>` synced!)
2. Merge final adapter into pure-text `Qwen/Qwen3.5-4B` (`scripts/prune_qwen35_text_only.py` pattern / HF merged weights, `< 4.3B` LLM) and push final merged artifact to HF Hub.
3. Run offline deterministic benchmarks on **Sealed `Dev 299`**:
   - **Oracle Text Benchmark** via `scripts/run_agent.py` + `scripts/run_eval.py`: target **`Final Score ≥ 0.87`**, **`State Match ≥ 91%`**, **`Tool F1 ≥ 89%`**.
   - **End-to-End Voice Benchmark (`.wav` → PhoWhisper Loop #38 + N-best → Qwen3.5-4B → SafeEnv)**: target **`Final Score ≥ 85.0%`** (e.g., `~85.5%–86.5%`), **`Latency ≤ 4.0s/sample`** (`≪ 30s` limit).
4. **Final Verification Gate (`AGENTS.md` §2 & §8):**
   - Verify exact parameter counts (`LLM < 5B`, `ASR + LLM < 6B`) via `scripts/count_params.py` (local offline / Colab check).
   - Verify 2x deterministic run hash equality (`temperature=0.0` produces bit-identical `dev_preds.jsonl` and `state_hash`).
   - Run full test suite (`python -m pytest -q`) confirming zero failures.
   - Write complete `experiments/<exp_id>/REPORT.md` and update `reports/LLM_AGENT_EXPERIMENTS_DASHBOARD.md` with HF links and deltas vs `exp_qwen35_sft_v2` (`0.7788`).

---

## 7. Summary of Target Milestones & Expected Trajectory

| Milestone | Dataset Scope | Expected Oracle Final Score | Expected E2E Voice Score | Total Time / Risk |
| :--- | :--- | :---: | :---: | :--- |
| **Baseline (`exp_qwen35_sft_v2`)** | Train=`1,246`, Dev=`299` (unseen) | **`0.7788`** (verified) | N/A (~`0.73–0.75` est.) | Done (`1.10s/sample`) |
| **Track A (Runtime + Schema + N-Best)** | Zero retrain (fixes runtime/guards) | **`0.800 – 0.815`** | **`0.765 – 0.785`** | ~0.5d local + eval |
| **Track B (Clean CV + LoopTool + `<think>` SFT V3)** | Clean `folds 1..4` → unseen `Fold 0` | **`0.815 – 0.835`** (clean) | **`0.790 – 0.815`** | ~2.5h Colab G4 |
| **Track C (Multi-Turn `GSPO/GiGPO/ARPO` + `SDPO/OPD`)** | Clean `folds 1..4` → unseen `Fold 0` | **`0.855 – 0.875`** (clean) | **`0.830 – 0.850`** | ~3.5h Colab G4 |
| **Track D (Full `1,246` + Voice OPD + Final Dev299)** | Train=`1,246`, **Sealed Dev=`299`** | **`≥ 0.875`** | **`≥ 85.0%+`** (`< 4.0s/s`) | ~4.0h Colab G4 |

> [!IMPORTANT]
> **Critical Risk Mitigation:** Never skip Clean Held-Out `Fold 0` (`996 -> 250`)! Evaluating SFT V3, RL, and OPD on unseen `Fold 0` guarantees every technique genuinely generalizes (+X pts) rather than memorizing training turns before spending Colab GPU budget on the final `1,246` run.
