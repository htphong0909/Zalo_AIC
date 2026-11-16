# Phase 2: Agent with Oracle Transcripts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `Qwen/Qwen3.5-4B` ReAct tool-calling agent (tool selector/compactor, XML+JSON tool-call parser with schema type coercion, LifeMate prompt & history renderer, pre-flight validated ReAct loop) and benchmark it with oracle transcripts (`audio_text`) on the 299-sample `dev_v1` split.

**Architecture:**
- Pure-Python, deterministic preprocessing and loop orchestration in `src/agent/`, tested offline with `pytest` using a fake `ChatBackend`.
- Pre-flight schema validation via `SafeEnv.validate()` before every `SafeEnv.call()`: schema-invalid calls are repaired up to `max_repairs=2` and never sent to `SafeEnv.call()`, guaranteeing 0 logged schema rejections.
- GPU inference executed on Google Colab L4 via `scripts/run_agent.py --transcripts oracle` using `TransformersBackend` (`Qwen/Qwen3.5-4B`, `bfloat16`, `do_sample=False`, `enable_thinking=False`, `max_model_len=16384`).

**Tech Stack:** Python 3.10+, `pydantic>=2.7`, `transformers>=5.22.0`, `torch>=2.4`, `pytest`.

---

### Task 2.1: Tool Selector & Schema Compactor (`src/agent/tool_selector.py`)

**Files:**
- Create: `src/agent/__init__.py`
- Create: `src/agent/tool_selector.py`
- Create: `tests/test_tool_selector.py`

- [ ] **Step 1: Write the failing test in `tests/test_tool_selector.py`**
  - Verify `select_tools(sample, catalog)` keeps only `sample.allowed_tools` (`context.installed_apps` tools + `sys.ask_missing_info`).
  - Verify each returned item has OpenAI tool format `{"type": "function", "function": {"name", "description", "parameters"}}`.
  - Verify `x-one-of` groups are surfaced in the function `description`, and format hints (`date`, `date-time`, `time`) are surfaced in property descriptions.
  - Verify `estimate_tools_tokens(tools)` returns a positive integer token estimate.
- [ ] **Step 2: Run `pytest tests/test_tool_selector.py -v` and confirm failure**
- [ ] **Step 3: Implement `src/agent/tool_selector.py`**
- [ ] **Step 4: Run `pytest tests/test_tool_selector.py -v` and confirm pass**

---

### Task 2.2: Tool-Call Output Parser & Type Coercer (`src/agent/tool_caller.py`)

**Files:**
- Create: `src/agent/tool_caller.py`
- Create: `tests/test_tool_caller.py`

- [ ] **Step 1: Write the failing test in `tests/test_tool_caller.py`**
  - Test parsing `Qwen3.5-4B` native XML tool calls (`<tool_call>\n<function=NAME>\n<parameter=KEY>\nVAL\n</parameter>\n</function>\n</tool_call>`).
  - Test parsing fallback JSON tool calls (`<tool_call>{"name": ..., "arguments": ...}</tool_call>`).
  - Test stripping `<think>...</think>` blocks before extracting tool calls and final text.
  - Test schema-guided type coercion: `integer` (`"24"` -> `24`), `number` (`"2.5"` -> `2.5`), `boolean` (`"true"` -> `True`), `array` (`'["contact_id"]'` -> `["contact_id"]`), `object`, and `string` (preserving leading-zero phone numbers like `"0912345678"` when schema type is `"string"`).
  - Test detecting malformed/unclosed `<tool_call>` blocks (`has_malformed_tool_call`).
- [ ] **Step 2: Run `pytest tests/test_tool_caller.py -v` and confirm failure**
- [ ] **Step 3: Implement `src/agent/tool_caller.py`**
- [ ] **Step 4: Run `pytest tests/test_tool_caller.py -v` and confirm pass**

---

### Task 2.3: System Prompt & History Renderer (`src/agent/prompts.py`, `src/agent/history.py`)

**Files:**
- Create: `src/agent/prompts.py`
- Create: `src/agent/history.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test in `tests/test_prompts.py`**
  - Verify `build_system_prompt(sample)` includes `current_time` with Vietnamese weekday, `timezone`, `location`, `user_profile`, and core LifeMate rules (`sys.ask_missing_info`, read-before-write, history reuse, conditional check, out-of-scope silence).
  - Verify `build_messages(sample, current_text=..., history_texts=..., max_prompt_tokens=...)` renders history user turns, assistant tool calls + truncated tool results, and the current user utterance, and truncates oldest history turns when the token budget is tight.
- [ ] **Step 2: Run `pytest tests/test_prompts.py -v` and confirm failure**
- [ ] **Step 3: Implement `src/agent/prompts.py` and `src/agent/history.py`**
- [ ] **Step 4: Run `pytest tests/test_prompts.py -v` and confirm pass**

---

### Task 2.4: LLM Client & ReAct Loop (`src/agent/llm_client.py`, `src/agent/react_loop.py`)

**Files:**
- Create: `src/agent/llm_client.py`
- Create: `src/agent/react_loop.py`
- Create: `tests/test_react_loop.py`

- [ ] **Step 1: Write the failing test in `tests/test_react_loop.py`**
  - Using a deterministic `FakeBackend`, test:
    1. Multi-step read-then-write (`smart_home.get_status` -> `smart_home.light_set`) updates `env.get_state_hash()` and logs only valid calls.
    2. Pre-flight validation repair: when `FakeBackend` first emits a schema-invalid call (missing required arg), `SafeEnv.validate()` catches it, feeds the Vietnamese schema error back to the model, the model repairs it on retry, and `env.get_tool_calls()` contains ONLY the repaired valid call (0 rejected calls logged).
    3. Unrecoverable schema error after `max_repairs`: skipped without calling `env.call()`, so `env.get_tool_calls()` never records a rejected call.
    4. `sys.ask_missing_info` stops the loop immediately after executing the current step's valid calls.
    5. Out-of-scope (no tool calls) returns `tool_calls == []`.
- [ ] **Step 2: Run `pytest tests/test_react_loop.py -v` and confirm failure**
- [ ] **Step 3: Implement `src/agent/llm_client.py` and `src/agent/react_loop.py`**
- [ ] **Step 4: Run `pytest tests/test_react_loop.py -v` and confirm pass**

---

### Task 2.5: Oracle Agent CLI & Dev Benchmark on Colab (`scripts/run_agent.py`)

**Files:**
- Create: `scripts/run_agent.py`
- Create: `tests/test_run_agent.py`

- [ ] **Step 1: Write unit test in `tests/test_run_agent.py` using `--backend fake` or injected fake backend on 3 train samples**
- [ ] **Step 2: Implement `scripts/run_agent.py`**
- [ ] **Step 3: Verify `pytest -v` passes locally**
- [ ] **Step 4: Run Oracle Benchmark on Colab L4 (`dev_v1`, 299 samples), download predictions & traces to `experiments/exp_20261104_oracle_baseline/`, run `scripts/run_eval.py`, and stop Colab session**
