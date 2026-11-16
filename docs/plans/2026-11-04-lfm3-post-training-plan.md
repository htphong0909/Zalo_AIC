# LFM3 Post-Training (SFT & GRPO) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a leak-free, multi-step ReAct Post-Training pipeline (Stage 1: SFT Warmup with `merge_and_unload`, Stage 2: GRPO Policy Optimization with Step-wise Verifiable Environment Reward) for `LiquidAI/LFM3-1.7B` strictly inheriting the golden code in `src/lfm3-post-training/`, targeting $\ge$ 85–90% Tool-Call F1 and $\ge$ 80–85% State Match on `dev_v1`.

**Architecture:** A two-stage reinforcement learning from environment feedback pipeline tailored for multi-step voice tool-calling:
1. **Stage 0 (Multi-Step ReAct Unrolling & ASR Augmentation):** Decompose each leak-free multi-step trajectory from `build_sft_trajectory()` into step-wise `(prompt, completion)` pairs preserving `sample.history`, `installed_apps` tool filtering, and `role="tool"` observations, augmented with `ChunkFormer-CTC` noisy transcripts.
2. **Stage 1 (SFT Warmup):** Train `LiquidAI/LFM3-1.7B` with TRL `SFTTrainer` and Tri-Hybrid LoRA (`r=16, alpha=32, dropout=0.05` targeting `q/k/v/o_proj, in/out_proj, w1/w2/w3`), then `merge_and_unload()` so the SFT-merged model serves as the zero-memory-overhead KL reference for Stage 2.
3. **Stage 2 (GRPO Alignment):** Train on top of the SFT-merged checkpoint with TRL `GRPOTrainer` (`r=32, alpha=64, dropout=0.0`, `num_generations=8`, `beta=0.04`), oversampling `ask` and `oos` steps 2x, guided by a step-wise verifiable reward combining Liquid AI's golden parameter scorer, BTC's `token_f1` metric, and `SafeEnv.validate()`.

**Tech Stack:** Python 3.12+, PyTorch, Hugging Face Transformers 5.20+, TRL 1.16+, PEFT 0.21+, SafeEnv facade, pytest.

**Spec:** [`docs/specs/2026-11-04-lfm3-post-training-spec.md`](file:///C:/WORKSPACE/ZALO_AIC/docs/specs/2026-11-04-lfm3-post-training-spec.md)

## Global Constraints
- Maximum system size strictly < 6 Billion parameters (`LiquidAI/LFM3-1.7B` 1.73B + `ChunkFormer-CTC-Large-Vie` 146M = 1.88B $\ll$ 6B).
- Maximum inference context length $\le$ 16,384 tokens.
- Per-sample inference latency $\le$ 30 seconds (target $\le$ 8 seconds on L4 GPU).
- Zero direct access to `<split>_environment/` or `state.json` (`SafeEnv` facade only).
- Strict train/dev separation: exclude all 299 IDs in `configs/splits/dev_v1.json` from training data.
- Offline reproducible evaluation with deterministic seeds (`seed=42`, `do_sample=False`, `repetition_penalty=1.1`).
- Clean repository root: no scratch files in root; all experiment artifacts in `experiments/`.

---

### Task 1: Multi-Step ReAct Unrolling & Dataset Builder for LFM3 SFT & GRPO

**Files:**
- Modify: `src/agent/sft_dataset.py`
- Create: `scripts/build_lfm3_data.py`
- Test: `tests/test_lfm3_data.py`

**Interfaces:**
- Consumes: `build_sft_trajectory(sample, catalog, *, current_text=None, history_texts=None)` from `src/agent/sft_dataset.py`, `load_catalog("train", "data")`, `load_split("train", "data")`, `configs/splits/dev_v1.json`, `experiments/exp_20261104_asr_chunkformer_large/transcripts.jsonl`.
- Produces:
  - `unroll_trajectory_steps(trajectory: dict[str, Any]) -> list[dict[str, Any]]` in `src/agent/sft_dataset.py`
  - `format_lfm3_completion_xml(completion_msg: dict[str, Any]) -> str` in `scripts/build_lfm3_data.py`
  - `build_sft_step_record(step: dict[str, Any], tokenizer: Any | None = None) -> dict[str, Any]` in `scripts/build_lfm3_data.py`
  - `build_grpo_step_record(step: dict[str, Any], tokenizer: Any | None = None) -> dict[str, Any]` in `scripts/build_lfm3_data.py`
  - CLI `scripts/build_lfm3_data.py` generating `data/sft/lfm3_train_sft.jsonl` and `data/sft/lfm3_train_grpo.jsonl` (with 2x oversampling for `step_category in ("ask", "oos")`).

- [x] **Step 1: Write the failing unit tests for multi-step unrolling and dataset building**

```python
# tests/test_lfm3_data.py
import json
import pytest
from src.core.schemas import Label, Sample, ToolCall, Turn
from src.agent.sft_dataset import build_sft_trajectory, unroll_trajectory_steps
from scripts.build_lfm3_data import (
    build_grpo_step_record,
    build_sft_step_record,
    format_lfm3_completion_xml,
    oversample_grpo_records,
)


def _mock_catalog() -> dict[str, dict]:
    return {
        "contacts.search": {
            "name": "contacts.search",
            "kind": "read",
            "description": "Tìm liên hệ",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
        "sms.send": {
            "name": "sms.send",
            "kind": "write",
            "description": "Gửi SMS",
            "parameters": {
                "type": "object",
                "properties": {"contact_id": {"type": "string"}, "content": {"type": "string"}},
                "required": ["content"],
            },
        },
        "sys.ask_missing_info": {
            "name": "sys.ask_missing_info",
            "kind": "meta",
            "description": "Hỏi thông tin thiếu",
            "parameters": {
                "type": "object",
                "properties": {"tool_name": {"type": "string"}, "params": {"type": "array"}},
                "required": ["tool_name", "params"],
            },
        },
    }


def test_unroll_multi_step_trajectory_preserves_history_and_tool_results():
    catalog = _mock_catalog()
    sample = Sample(
        id="train_9001",
        audio="audio/train_9001.wav",
        audio_text="Nhắn cho Bố là tối nay con về muộn",
        context={
            "current_time": "2026-11-20T17:00:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
            "installed_apps": {"contacts": {"tools": ["contacts.search"]}, "sms": {"tools": ["sms.send"]}},
        },
        history=[
            Turn(role="user", audio_text="Chào bạn"),
            Turn(role="assistant", text="Chào bạn, tôi có thể giúp gì?"),
        ],
        label=Label(
            tool_calls=[
                ToolCall(name="contacts.search", arguments={"query": "Bố"}),
                ToolCall(name="sms.send", arguments={"contact_id": "c_01", "content": "tối nay con về muộn"}),
            ],
            tool_results=[
                {"ok": True, "contacts": [{"id": "c_01", "name": "Bố", "phone": "0912345678"}]},
                {"ok": True, "message": {"id": "m_01"}},
            ],
        ),
    )

    traj = build_sft_trajectory(sample, catalog, current_text="Nhắn cho ba là tối nay con về muộn")
    steps = unroll_trajectory_steps(traj)

    # Step 0: read (contacts.search), Step 1: write (sms.send), Step 2: final_stop text
    assert len(steps) == 3
    assert steps[0]["step_index"] == 0
    assert steps[0]["step_category"] == "act"
    assert steps[0]["step_calls"] == [{"name": "contacts.search", "arguments": {"query": "Bố"}}]
    assert steps[0]["prefix_calls"] == []
    assert steps[0]["prompt_messages"][-1]["content"] == "Nhắn cho ba là tối nay con về muộn"

    # Step 1 prompt must contain Step 0 assistant call AND Step 0 tool result containing c_01
    assert steps[1]["step_index"] == 1
    assert steps[1]["step_category"] == "act"
    assert steps[1]["prefix_calls"] == [{"name": "contacts.search", "arguments": {"query": "Bố"}}]
    assert steps[1]["step_calls"] == [
        {"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "tối nay con về muộn"}}
    ]
    assert steps[1]["prompt_messages"][-1]["role"] == "tool"
    assert "c_01" in steps[1]["prompt_messages"][-1]["content"]

    # Step 2 is final stop confirmation (no tool calls)
    assert steps[2]["step_index"] == 2
    assert steps[2]["step_category"] == "final_stop"
    assert steps[2]["step_calls"] == []


def test_build_sft_and_grpo_records_and_oversampling():
    catalog = _mock_catalog()
    sample = Sample(
        id="train_9002",
        audio="audio/train_9002.wav",
        audio_text="Gửi tin nhắn giúp mình",
        context={
            "current_time": "2026-11-20T17:00:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
            "installed_apps": {"sms": {"tools": ["sms.send"]}},
        },
        label=Label(
            tool_calls=[
                ToolCall(
                    name="sys.ask_missing_info",
                    arguments={"tool_name": "sms.send", "params": ["contact_id", "content"]},
                )
            ],
            tool_results=[{"ok": True}],
        ),
    )
    traj = build_sft_trajectory(sample, catalog)
    steps = unroll_trajectory_steps(traj)
    assert len(steps) == 1
    assert steps[0]["step_category"] == "ask"

    sft_rec = build_sft_step_record(steps[0], tokenizer=None)
    assert "prompt" in sft_rec and "completion" in sft_rec
    assert "sys.ask_missing_info" in sft_rec["completion"]

    grpo_rec = build_grpo_step_record(steps[0], tokenizer=None)
    assert grpo_rec["sample_id"] == "train_9002"
    assert grpo_rec["step_category"] == "ask"
    assert json.loads(grpo_rec["calls"])[0]["name"] == "sys.ask_missing_info"

    balanced = oversample_grpo_records([grpo_rec], seed=42)
    assert len(balanced) == 2  # 2x oversampled for 'ask'
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lfm3_data.py -v`
Expected: FAIL with `ImportError` / `ModuleNotFoundError`

- [x] **Step 3: Update `src/agent/sft_dataset.py` and implement `scripts/build_lfm3_data.py`**

1. In `src/agent/sft_dataset.py`:
   - Update `build_sft_trajectory` to accept optional `current_text: str | None = None` and `history_texts: dict[int, str] | None = None` and pass them to `build_messages`.
   - Implement `unroll_trajectory_steps(trajectory: dict[str, Any]) -> list[dict[str, Any]]` which locates the last `role == "user"` turn and slices every subsequent `role == "assistant"` turn into a step dictionary with keys: `id`, `step_index`, `step_category` (`"act"`, `"ask"`, `"oos"`, `"final_stop"`), `prompt_messages`, `completion_message`, `step_calls`, `prefix_calls`, `tools`.
2. In `scripts/build_lfm3_data.py`:
   - Implement `format_lfm3_completion_xml(completion_msg)` and `render_step_prompt_completion(step, tokenizer=None)` using `tokenizer.apply_chat_template(..., tools=step["tools"], enable_thinking=False)` when `tokenizer` is provided, or deterministic XML formatting when `tokenizer is None`.
   - Implement `build_sft_step_record`, `build_grpo_step_record`, `oversample_grpo_records`.
   - Implement CLI `--split train`, `--root data`, `--exclude-ids configs/splits/dev_v1.json`, `--asr-transcripts experiments/exp_20261104_asr_chunkformer_large/transcripts.jsonl`, `--model-id LiquidAI/LFM3-1.7B`, `--out-dir data/sft` (combining clean + differing ASR trajectories and strictly excluding `dev_v1.json`).

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lfm3_data.py tests/test_sft_dataset.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/agent/sft_dataset.py scripts/build_lfm3_data.py tests/test_lfm3_data.py
git commit -m "feat(lfm3): add multi-step ReAct unrolling and ASR-augmented dataset builder"
```

---

### Task 2: Parameter Scorer & Tool Matcher (Inheriting Liquid AI Golden Code + BTC Metric)

**Files:**
- Create: `src/agent/lfm3_scorer.py`
- Test: `tests/test_lfm3_scorer.py`

**Interfaces:**
- Consumes: `catalog: dict[str, dict[str, Any]]` from `load_catalog`, `gold_calls: list[dict]`, `pred_calls: list[dict]`.
- Produces:
  - `norm(s: Any) -> str`
  - `infer_param_kind(prop_name: str, prop_spec: dict[str, Any]) -> str`
  - `same_value(kind: str, gold: Any, pred: Any) -> bool`
  - `canonical(name: str, arguments: dict[str, Any] | None, catalog: dict[str, dict]) -> dict[str, Any]`
  - `same_call(gold: dict, pred: dict, catalog: dict[str, dict]) -> bool`
  - `calls_match(gold: list[dict], pred: list[dict], catalog: dict[str, dict]) -> bool`
  - `tool_name_f1(gold: list[dict], pred: list[dict]) -> float`
  - `arg_overlap_f1(gold: list[dict], pred: list[dict], catalog: dict[str, dict]) -> float`

- [x] **Step 1: Write the failing unit tests for `src/agent/lfm3_scorer.py`**

```python
# tests/test_lfm3_scorer.py
import pytest
from src.agent.lfm3_scorer import (
    arg_overlap_f1,
    calls_match,
    norm,
    same_value,
    tool_name_f1,
)


def _sample_catalog() -> dict[str, dict]:
    return {
        "banking.transfer": {
            "name": "banking.transfer",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_account_id": {"type": "string", "x-ref": "banking.accounts"},
                    "beneficiary_id": {"type": "string", "x-ref": "banking.beneficiaries"},
                    "amount": {"type": "integer"},
                    "note": {"type": "string", "x-match": "fuzzy"},
                },
                "required": ["from_account_id", "beneficiary_id", "amount"],
            },
        },
        "ewallet.transfer": {
            "name": "ewallet.transfer",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_id": {"type": "string"},
                    "amount": {"type": "integer"},
                },
                "required": ["amount"],
            },
        },
        "alarm.create": {
            "name": "alarm.create",
            "parameters": {
                "type": "object",
                "properties": {"time": {"type": "string", "format": "date-time"}},
                "required": ["time"],
            },
        },
    }


def test_norm_vietnamese_and_accents():
    assert norm("Phạm Hùng") == norm("pham hung")
    assert norm("  Hà   Nội! ") == "ha noi"


def test_same_value_kinds():
    assert same_value("number", 2800000, 2800000.0) is True
    assert same_value("bool", True, True) is True
    assert same_value("bool", True, "true") is False
    assert same_value("fuzzy", "chuyển tiền nhà tháng này", "chuyển tiền nhà tháng này nhé") is True
    assert same_value("datetime", "2026-11-20T06:00:00+07:00", "2026-11-20T06:00:00+07:00") is True


def test_calls_match_order_invariant_and_fuzzy():
    cat = _sample_catalog()
    gold = [
        {
            "name": "banking.transfer",
            "arguments": {
                "from_account_id": "acc_01",
                "beneficiary_id": "ben_03",
                "amount": 2800000,
                "note": "tiền nhà tháng 11",
            },
        },
        {"name": "ewallet.transfer", "arguments": {"amount": 200000, "contact_id": "c_04"}},
    ]
    pred_same = [
        {"name": "ewallet.transfer", "arguments": {"amount": 200000, "contact_id": "c_04"}},
        {
            "name": "banking.transfer",
            "arguments": {
                "from_account_id": "acc_01",
                "beneficiary_id": "ben_03",
                "amount": 2800000,
                "note": "tiền nhà tháng 11 nhé",
            },
        },
    ]
    pred_wrong_amount = [
        {
            "name": "banking.transfer",
            "arguments": {"from_account_id": "acc_01", "beneficiary_id": "ben_03", "amount": 3800000},
        }
    ]
    assert calls_match(gold, pred_same, cat) is True
    assert calls_match(gold, pred_wrong_amount, cat) is False


def test_tool_name_and_arg_overlap_f1():
    cat = _sample_catalog()
    gold = [{"name": "alarm.create", "arguments": {"time": "2026-11-21T06:00:00+07:00"}}]
    pred_right_tool_wrong_arg = [{"name": "alarm.create", "arguments": {"time": "2026-11-21T09:00:00+07:00"}}]
    assert tool_name_f1(gold, pred_right_tool_wrong_arg) == 1.0
    assert arg_overlap_f1(gold, pred_right_tool_wrong_arg, cat) == 0.0
    assert arg_overlap_f1(gold, gold, cat) == 1.0
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lfm3_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.lfm3_scorer'`

- [x] **Step 3: Implement `src/agent/lfm3_scorer.py`**

Port the golden scoring functions from `src/lfm3-post-training/lfm3_grpo_with_trl.ipynb` (`norm`, `contained`, `name_equal`, `text_similar`, `parse_when`, `calculate`, `same_value`, `canonical`, `tool_name_f1`) and combine with BTC's `token_f1` and Kuhn's maximum matching from `src/eval/metrics.py` (`calls_match`, `arg_overlap_f1`).

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lfm3_scorer.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/agent/lfm3_scorer.py tests/test_lfm3_scorer.py
git commit -m "feat(lfm3): add golden parameter scorer and bipartite tool matcher"
```

---

### Task 3: Step-wise Verifiable Environment Reward Module

**Files:**
- Create: `src/agent/lfm3_reward.py`
- Test: `tests/test_lfm3_reward.py`

**Interfaces:**
- Consumes: `src.agent.lfm3_scorer`, `src.agent.tool_caller`, `src.env.safe_env.SafeEnv`, `tokenizer` (with optional `.parse_response()`).
- Produces:
  - `extract_predicted_calls(completion_ids: Any, prompt: str, tools: list[dict], tokenizer: Any | None = None, catalog: dict[str, dict] | None = None) -> tuple[list[dict], bool]`
  - `compute_step_reward(gold_calls: list[dict], pred_calls: list[dict], catalog: dict[str, dict], *, is_valid_format: bool = True, all_env_valid: bool = True) -> float`
  - `validate_step_in_env(sample_id: str, split: str, root: str, prefix_calls: list[dict], pred_calls: list[dict]) -> bool`
  - `create_lfm3_reward_func(catalog: dict[str, dict], tokenizer: Any | None = None, split: str = "train", root: str = "data", verify_env: bool = True)` returning a reward callable `tool_call_reward(prompts, completion_ids, calls, **kwargs) -> list[float]` adhering to `lfm3_grpo_with_trl.ipynb`.

- [x] **Step 1: Write failing unit tests for `src/agent/lfm3_reward.py`**

```python
# tests/test_lfm3_reward.py
import json
import pytest
from src.agent.lfm3_reward import (
    compute_step_reward,
    create_lfm3_reward_func,
    extract_predicted_calls,
)


def _catalog() -> dict[str, dict]:
    return {
        "weather.get_forecast": {
            "name": "weather.get_forecast",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string", "format": "date"}},
                "required": ["date"],
            },
        }
    }


def test_compute_step_reward_tiers():
    cat = _catalog()
    gold = [{"name": "weather.get_forecast", "arguments": {"date": "2026-11-20"}}]
    pred_exact = [{"name": "weather.get_forecast", "arguments": {"date": "2026-11-20"}}]
    pred_wrong_date = [{"name": "weather.get_forecast", "arguments": {"date": "2026-11-25"}}]

    # Tier 1: Exact match + valid env -> 1.0
    assert compute_step_reward(gold, pred_exact, cat, is_valid_format=True, all_env_valid=True) == 1.0

    # Tier 2: Right tool name, wrong arg, valid env -> 0.2 * 1.0 + 0.3 * 0.0 + 0.1 = 0.3
    r_partial = compute_step_reward(gold, pred_wrong_date, cat, is_valid_format=True, all_env_valid=True)
    assert pytest.approx(r_partial, 0.01) == 0.3

    # Tier 3: OOS / Final Stop (gold == [])
    assert compute_step_reward([], [], cat, is_valid_format=True, all_env_valid=True) == 1.0
    assert compute_step_reward([], pred_exact, cat, is_valid_format=True, all_env_valid=True) == 0.0

    # Tier 4: Malformed tool call -> 0.0
    assert compute_step_reward(gold, [], cat, is_valid_format=False, all_env_valid=False) == 0.0


def test_extract_predicted_calls_fallback_xml():
    cat = _catalog()
    raw_xml = "<tool_call>\n<function=weather.get_forecast>\n<parameter=date>\n2026-11-20\n</parameter>\n</function>\n</tool_call>"
    calls, valid_fmt = extract_predicted_calls(
        completion_ids=raw_xml,
        prompt="",
        tools=[],
        tokenizer=None,
        catalog=cat,
    )
    assert valid_fmt is True
    assert calls == [{"name": "weather.get_forecast", "arguments": {"date": "2026-11-20"}}]


def test_reward_callable_interface():
    cat = _catalog()
    reward_fn = create_lfm3_reward_func(cat, tokenizer=None, verify_env=False)
    prompts = ["prompt1", "prompt2"]
    completions = [
        "<tool_call>\n<function=weather.get_forecast>\n<parameter=date>\n2026-11-20\n</parameter>\n</function>\n</tool_call>",
        "Đã thực hiện xong yêu cầu của bạn.",
    ]
    calls_col = [
        json.dumps([{"name": "weather.get_forecast", "arguments": {"date": "2026-11-20"}}]),
        json.dumps([]),
    ]
    rewards = reward_fn(prompts=prompts, completion_ids=completions, calls=calls_col)
    assert rewards == [1.0, 1.0]
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lfm3_reward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.lfm3_reward'`

- [x] **Step 3: Implement `src/agent/lfm3_reward.py`**

Implement `extract_predicted_calls` (supporting both `tokenizer.parse_response(completion_ids, prefix=prompt, tools=tools)` from `lfm3_grpo_with_trl.ipynb` and `parse_tool_calls` from `src/agent/tool_caller.py`), `compute_step_reward`, `validate_step_in_env` (replaying `prefix_calls` and checking `env.validate` on `SafeEnv`), and `create_lfm3_reward_func`.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lfm3_reward.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/agent/lfm3_reward.py tests/test_lfm3_reward.py
git commit -m "feat(lfm3): add step-wise verifiable reward function with SafeEnv validation"
```

---

### Task 4: SFT Warmup Engine, Merge Routine & Trainer Script

**Files:**
- Modify: `src/agent/llm_client.py`
- Modify: `src/core/config.py`
- Create: `src/agent/lfm3_sft.py`
- Create: `scripts/train_lfm3_sft.py`
- Create: `configs/lfm3_sft.yaml`
- Test: `tests/test_lfm3_sft.py`

**Interfaces:**
- Consumes: `LiquidAI/LFM3-1.7B`, `data/sft/lfm3_train_sft.jsonl`, `peft.LoraConfig`, `trl.SFTConfig`, `trl.SFTTrainer`.
- Produces:
  - `LFM3_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"]`
  - `build_lfm3_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05) -> LoraConfig`
  - `build_lfm3_sft_config(cfg: dict[str, Any], use_bf16: bool = True) -> SFTConfig`
  - `run_lfm3_sft_training(config_path: str | Path)` saving both LoRA adapter (`adapter/`) and merged model (`merged/` via `merge_and_unload()`).
  - Updated `TransformersBackend` in `src/agent/llm_client.py` with `padding_side="left"`, `dtype=torch_dtype`, and `repetition_penalty=1.1`.

- [x] **Step 1: Write failing test for SFT configuration and PEFT target modules**

```python
# tests/test_lfm3_sft.py
import pytest
from src.agent.lfm3_sft import (
    LFM3_TARGET_MODULES,
    build_lfm3_lora_config,
    build_lfm3_sft_config,
    load_yaml_config,
)


def test_lfm3_lora_target_modules():
    cfg = build_lfm3_lora_config(r=16, alpha=32, dropout=0.05)
    assert set(cfg.target_modules) == {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj", "out_proj",
        "w1", "w2", "w3",
    }
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert pytest.approx(cfg.lora_dropout, 0.001) == 0.05
    assert cfg.task_type == "CAUSAL_LM"


def test_load_sft_yaml_and_build_sft_config():
    raw = load_yaml_config("configs/lfm3_sft.yaml")
    assert raw["model_id"] == "LiquidAI/LFM3-1.7B"
    assert raw["lora"]["r"] == 16
    sft_cfg = build_lfm3_sft_config(raw, use_bf16=False)
    assert sft_cfg.per_device_train_batch_size == 4
    assert sft_cfg.gradient_accumulation_steps == 4
    assert pytest.approx(sft_cfg.learning_rate, 1e-6) == 2e-4
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lfm3_sft.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.lfm3_sft'`

- [x] **Step 3: Implement `src/agent/lfm3_sft.py`, `configs/lfm3_sft.yaml`, `scripts/train_lfm3_sft.py`, and update `src/agent/llm_client.py`**

1. Update `src/core/config.py` (`LLMConfig`) and `src/agent/llm_client.py` (`TransformersBackend`) to support `repetition_penalty: float = 1.0` (set to `1.1` in LFM3 configs), `padding_side="left"`, and `dtype=torch_dtype`.
2. Implement `src/agent/lfm3_sft.py` strictly adhering to `src/lfm3-post-training/lfm3_sft_with_trl.ipynb` (including `trainer.save_model(adapter_dir)` followed by `merged_model = trainer.model.merge_and_unload(); merged_model.save_pretrained(merged_dir); tokenizer.save_pretrained(merged_dir)`).
3. Create `configs/lfm3_sft.yaml` and `scripts/train_lfm3_sft.py`.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lfm3_sft.py tests/test_config.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/core/config.py src/agent/llm_client.py src/agent/lfm3_sft.py configs/lfm3_sft.yaml scripts/train_lfm3_sft.py tests/test_lfm3_sft.py
git commit -m "feat(lfm3): add SFT warmup trainer with merge_and_unload and repetition_penalty support"
```

---

### Task 5: GRPO Policy Gradient Engine & Trainer Script

**Files:**
- Create: `src/agent/lfm3_grpo.py`
- Create: `scripts/train_lfm3_grpo.py`
- Create: `configs/lfm3_grpo.yaml`
- Test: `tests/test_lfm3_grpo.py`

**Interfaces:**
- Consumes: SFT-merged checkpoint (`experiments/exp_20261104_lfm3_sft/merged` or `LiquidAI/LFM3-1.7B`), `data/sft/lfm3_train_grpo.jsonl`, `create_lfm3_reward_func`, `trl.GRPOConfig`, `trl.GRPOTrainer`.
- Produces:
  - `build_lfm3_grpo_lora_config(r: int = 32, alpha: int = 64, dropout: float = 0.0) -> LoraConfig`
  - `build_lfm3_grpo_config(cfg: dict[str, Any], use_bf16: bool = True) -> GRPOConfig`
  - `run_lfm3_grpo_training(config_path: str | Path)` saving both LoRA adapter (`adapter/`) and merged champion model (`merged/`).

- [x] **Step 1: Write failing test for GRPO configuration**

```python
# tests/test_lfm3_grpo.py
import pytest
from src.agent.lfm3_grpo import (
    build_lfm3_grpo_config,
    build_lfm3_grpo_lora_config,
)
from src.agent.lfm3_sft import load_yaml_config


def test_lfm3_grpo_lora_config():
    cfg = build_lfm3_grpo_lora_config(r=32, alpha=64, dropout=0.0)
    assert set(cfg.target_modules) == {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj", "out_proj",
        "w1", "w2", "w3",
    }
    assert cfg.r == 32
    assert cfg.lora_alpha == 64
    assert cfg.lora_dropout == 0.0


def test_load_grpo_yaml_and_build_grpo_config():
    raw = load_yaml_config("configs/lfm3_grpo.yaml")
    assert raw["lora"]["r"] == 32
    grpo_cfg = build_lfm3_grpo_config(raw, use_bf16=False)
    assert grpo_cfg.num_generations == 8
    assert grpo_cfg.generation_batch_size == 8
    assert grpo_cfg.max_completion_length == 256
    assert pytest.approx(grpo_cfg.beta, 0.001) == 0.04
    assert pytest.approx(grpo_cfg.learning_rate, 1e-7) == 1e-5
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lfm3_grpo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.lfm3_grpo'`

- [x] **Step 3: Implement `src/agent/lfm3_grpo.py`, `configs/lfm3_grpo.yaml`, and `scripts/train_lfm3_grpo.py`**

Implement:
1. `build_lfm3_grpo_lora_config(r=32, alpha=64, dropout=0.0)`.
2. `build_lfm3_grpo_config(cfg, use_bf16)` strictly following `src/lfm3-post-training/lfm3_grpo_with_trl.ipynb` (`max_steps=150`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=16`, `num_generations=8`, `generation_batch_size=8`, `max_completion_length=256`, `temperature=1.0`, `learning_rate=1e-5`, `beta=0.04`, `warmup_steps=5`).
3. `run_lfm3_grpo_training(config_path)` loading the SFT-merged model, attaching `peft_config`, wiring `create_lfm3_reward_func(catalog, tokenizer, split="train", root=...)`, running `trainer.train()`, and saving both `adapter/` and `merged/` via `merge_and_unload()`.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lfm3_grpo.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/agent/lfm3_grpo.py configs/lfm3_grpo.yaml scripts/train_lfm3_grpo.py tests/test_lfm3_grpo.py
git commit -m "feat(lfm3): add GRPO policy gradient trainer on SFT-merged reference model"
```

---

### Task 6: Remote Colab Execution, Dev Set Benchmark & Verification

**Files:**
- Modify: `configs/e2e_chunkformer_ctc_lfm3_17b.yaml`
- Test: Full test suite locally (`pytest -v`) + SFT & GRPO execution on Colab GPU L4 + evaluation on `dev_v1` (299 samples).

- [x] **Step 1: Run full local unit test suite**
  - Run: `pytest -v`
  - Expected: All unit tests pass with 0 warnings/errors.
- [x] **Step 2: Generate leak-free SFT & GRPO datasets with LFM3 native chat template**
  - Run `scripts/build_lfm3_data.py` to build `data/sft/lfm3_train_sft.jsonl` and `data/sft/lfm3_train_grpo.jsonl` (verifying 0 overlap with `configs/splits/dev_v1.json`).
- [x] **Step 3: Execute Stage 1 (SFT Warmup) on Colab GPU & Evaluate SFT Checkpoint**
  - Run `python scripts/train_lfm3_sft.py --config configs/lfm3_sft.yaml` on Colab GPU.
  - Benchmark SFT-merged checkpoint on `dev_v1` (299 samples) via `scripts/run_agent.py` and `scripts/run_eval.py`.
- [x] **Step 4: Execute Stage 2 (GRPO Policy Optimization) on Colab GPU & Evaluate Final Checkpoint**
  - Run `python scripts/train_lfm3_grpo.py --config configs/lfm3_grpo.yaml` starting from the SFT-merged model.
  - Benchmark GRPO-merged checkpoint on `dev_v1` (299 samples) via `scripts/run_agent.py` and `scripts/run_eval.py`.
- [x] **Step 5: Sync artifacts, verify parameter budget (< 6B), and document metrics**
  - Download evaluation metrics (`metrics.json`, `error_breakdown.json`, `predictions.jsonl`) to `experiments/` and push champion checkpoint to Hugging Face Hub.
