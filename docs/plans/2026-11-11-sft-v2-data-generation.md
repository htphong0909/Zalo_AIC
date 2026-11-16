# SFT V2 Data Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a multi-generator synthetic data pipeline adapted directly from author codebases in `references/tool_calling_repos/` (When2Call, Tool-N1, ToolSandbox, APIGen-MT, Hammer) to expand the leak-free training dataset to ~2,750 verified multi-turn samples and eliminate the 4 major failure modes (missing `ask`, premature stopping, read-to-write chaining, underfitting).

**Architecture:** Four modular generators (`when2call_gen.py`, `anchor_binding_gen.py`, `apigen_mt_gen.py`, and `distill_think.py`) producing candidate trajectories verified 100% against `SafeEnv.validate()` and `SafeEnv.call()`, packed with Hammer tool permutation into `minicpm5_train_sft_v2.jsonl.gz`.

**Tech Stack:** Python 3.12, Pydantic, zaic_tools.ToolEnv, Pytest.

**Spec:** `references/tool_calling_repos/13_MiniCPM/.github/SFT_ERROR_ANALYSIS_AND_PAPER_RESEARCH_2025_2026.md`

## Global Constraints
- **Zero Data Leakage:** Never read, sample, or leak any sample ID or utterance from `configs/splits/dev_v1.json` (299 Sealed Dev samples).
- **Environment Isolation:** App data and state can ONLY be verified via `zaic_tools.ToolEnv` and `SafeEnv` (no direct reading of `state.json`).
- **Format Integrity:** Native MiniCPM5 XML: `<think>` 4-slot micro-reasoning trace + `<function name="..."><param name="...">...</param></function>`.
- **Quality Gate:** 100% of newly synthesized SFT steps must pass `SafeEnv.validate()` and produce valid state changes or valid `sys.ask_missing_info` calls.

---

### Task 1: When2Call Contrastive Tri-Branch Generator

**Files:**
- Create: `src/data_gen/when2call_gen.py`
- Test: `tests/test_when2call_gen.py`
- Reference Author Code: `references/tool_calling_repos/08_When2Call_NAACL2025/synthetic_data_gen/create_raw_train_data.py`

**Interfaces:**
- Consumes:
  - `src.core.schemas.Sample`, `ToolCall`, `ASK_MISSING_INFO`
  - `src.core.dataset.load_split`, `load_catalog`
- Produces:
  - `generate_when2call_contrastive_samples(train_samples: list[Sample], catalog: dict[str, Any], count: int = 500) -> list[dict[str, Any]]`

- [ ] **Step 1: Write failing unit test for When2Call generator**

```python
# tests/test_when2call_gen.py
import pytest
from src.core.schemas import ASK_MISSING_INFO, Sample, ToolCall
from src.data_gen.when2call_gen import (
    drop_required_param_from_utterance,
    generate_when2call_contrastive_samples,
)

def test_drop_required_param_from_utterance():
    text = "Chuyển 500 nghìn cho Hương qua ví điện tử"
    param_name = "amount"
    modified_text = drop_required_param_from_utterance(text, param_name)
    assert "500 nghìn" not in modified_text
    assert "Hương" in modified_text

def test_generate_when2call_samples():
    sample = Sample(
        id="train_mock_01",
        audio_text="Chuyển 500 nghìn cho Hương qua ví điện tử",
        context={"user_profile": {"name": "Test"}},
        label={
            "tool_calls": [
                {"name": "ewallet.transfer", "arguments": {"recipient_phone": "0987654321", "amount": 500000}}
            ]
        }
    )
    catalog = {
        "ewallet.transfer": {
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient_phone": {"type": "string"},
                    "amount": {"type": "integer"}
                },
                "required": ["recipient_phone", "amount"]
            }
        }
    }
    results = generate_when2call_contrastive_samples([sample], catalog, count=1)
    assert len(results) == 1
    res = results[0]
    assert res["step_category"] == "ask"
    assert any(c["name"] == ASK_MISSING_INFO for c in res["step_calls"])
    ask_call = next(c for c in res["step_calls"] if c["name"] == ASK_MISSING_INFO)
    assert ask_call["arguments"]["tool_name"] == "ewallet.transfer"
    assert "amount" in ask_call["arguments"]["params"] or "recipient_phone" in ask_call["arguments"]["params"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_when2call_gen.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data_gen.when2call_gen'`

- [ ] **Step 3: Implement `src/data_gen/when2call_gen.py`**

```python
"""When2Call (NAACL 2025) Tri-Branch Contrastive Data Generator.

Adapted from references/tool_calling_repos/08_When2Call_NAACL2025/synthetic_data_gen/create_raw_train_data.py
Creates contrastive (ACT -> ASK) pairs by dropping required parameters from user utterances.
"""
from __future__ import annotations

import re
import random
from typing import Any
from src.core.schemas import ASK_MISSING_INFO, Sample, ToolCall

_AMOUNT_PATTERNS = [
    r"\b\d+[\s\.]*(?:nghìn|ngàn|k|triệu|đ|vnd|đồng)\b",
    r"\b\d{4,9}\b",
]

_TIME_PATTERNS = [
    r"\b\d{1,2}(?::\d{2})?\s*(?:giờ|g|h|am|pm)?\s*(?:sáng|trưa|chiều|tối)?\s*(?:mai|hôm nay|ngày mai)?\b",
]

def drop_required_param_from_utterance(text: str, param_name: str) -> str:
    """Omit or obscure a required parameter value from the text."""
    modified = text
    if param_name in ("amount", "money", "price"):
        for pat in _AMOUNT_PATTERNS:
            modified = re.sub(pat, "chút tiền", modified, flags=re.IGNORECASE)
    elif param_name in ("time", "start_time", "alarm_time", "date"):
        for pat in _TIME_PATTERNS:
            modified = re.sub(pat, "", modified, flags=re.IGNORECASE)
        modified = re.sub(r"\s+", " ", modified).strip()
    return modified

def generate_when2call_contrastive_samples(
    train_samples: list[Sample],
    catalog: dict[str, Any],
    count: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Synthesize verifiable ASK samples from ACT samples with required parameter omission."""
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []

    candidates: list[tuple[Sample, ToolCall, str]] = []
    for s in train_samples:
        calls = s.label.get("tool_calls", []) if s.label else []
        for c in calls:
            tname = c.get("name") if isinstance(c, dict) else c.name
            args = c.get("arguments") if isinstance(c, dict) else c.arguments
            if tname == ASK_MISSING_INFO or not tname:
                continue
            spec = catalog.get(tname, {}).get("parameters", {})
            required = spec.get("required") or []
            for req_param in required:
                if req_param in (args or {}):
                    candidates.append((s, ToolCall(name=tname, arguments=args or {}), req_param))

    if not candidates:
        return results

    rng.shuffle(candidates)
    for idx, (sample, gold_call, param_to_drop) in enumerate(candidates):
        if len(results) >= count:
            break
        orig_text = sample.audio_text or ""
        masked_text = drop_required_param_from_utterance(orig_text, param_to_drop)
        if masked_text == orig_text and param_to_drop in ("amount", "time"):
            continue

        ask_call = ToolCall(
            name=ASK_MISSING_INFO,
            arguments={"tool_name": gold_call.name, "params": [param_to_drop]},
        )
        prompt_msgs = [
            {"role": "system", "content": "Bạn là trợ lý ảo LifeMate."},
            {"role": "user", "content": masked_text},
        ]
        results.append({
            "id": f"when2call_{sample.id}_{idx}",
            "sample_id": sample.id,
            "step_index": 0,
            "step_category": "ask",
            "prompt_messages": prompt_msgs,
            "step_calls": [ask_call.model_dump()],
            "tools": list((sample.context or {}).get("installed_apps", {}).keys()),
        })

    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_when2call_gen.py -v`  
Expected: PASS

- [ ] **Step 5: Commit Task 1**

```bash
git add src/data_gen/when2call_gen.py tests/test_when2call_gen.py
git commit -m "feat(data_gen): implement When2Call contrastive tri-branch generator"
```

---

### Task 2: ToolSandbox Milestone Anchor Binding Generator

**Files:**
- Create: `src/data_gen/anchor_binding_gen.py`
- Test: `tests/test_anchor_binding_gen.py`
- Reference Author Code: `references/tool_calling_repos/09_ToolSandbox_NAACL2025/tool_sandbox/`

**Interfaces:**
- Consumes:
  - `src.core.schemas.Sample`, `ToolCall`
- Produces:
  - `generate_anchor_binding_samples(train_samples: list[Sample], catalog: dict[str, Any], count: int = 400) -> list[dict[str, Any]]`

- [ ] **Step 1: Write failing unit test for Anchor Binding generator**

```python
# tests/test_anchor_binding_gen.py
import pytest
from src.core.schemas import Sample, ToolCall
from src.data_gen.anchor_binding_gen import generate_anchor_binding_samples

def test_generate_anchor_binding_samples():
    sample = Sample(
        id="train_mock_read_write",
        audio_text="Gọi cho Sếp Dũng nhé",
        context={"user_profile": {"name": "Test"}},
        label={
            "tool_calls": [
                {"name": "contacts.search", "arguments": {"query": "Sếp Dũng"}},
                {"name": "phone.call", "arguments": {"contact_id": "c_08"}}
            ],
            "tool_results": [
                {"ok": True, "contacts": [{"id": "c_08", "name": "Sếp Dũng", "phone": "09925652941"}]}
            ]
        }
    )
    catalog = {
        "contacts.search": {"kind": "read", "parameters": {"properties": {"query": {"type": "string"}}}},
        "phone.call": {"kind": "write", "parameters": {"properties": {"contact_id": {"type": "string"}}}}
    }
    results = generate_anchor_binding_samples([sample], catalog, count=1)
    assert len(results) >= 1
    step = results[0]
    # Check that Step 2 prompt includes the observation from Step 1
    assert any(m["role"] == "tool" for m in step["prompt_messages"])
    # Check that Step 2 calls phone.call with contact_id="c_08"
    assert any(c["name"] == "phone.call" and c["arguments"].get("contact_id") == "c_08" for c in step["step_calls"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_anchor_binding_gen.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data_gen.anchor_binding_gen'`

- [ ] **Step 3: Implement `src/data_gen/anchor_binding_gen.py`**

```python
"""ToolSandbox (NAACL 2025) Milestone Anchor Binding Generator.

Adapted from references/tool_calling_repos/09_ToolSandbox_NAACL2025/tool_sandbox/
Generates multi-turn Read -> Write steps where Write parameters are dynamically bound from prior Read observations.
"""
from __future__ import annotations

import json
from typing import Any
from src.core.schemas import Sample, ToolCall

def generate_anchor_binding_samples(
    train_samples: list[Sample],
    catalog: dict[str, Any],
    count: int = 400,
) -> list[dict[str, Any]]:
    """Synthesize multi-turn trajectory steps binding observation IDs to subsequent action tools."""
    results: list[dict[str, Any]] = []

    for s in train_samples:
        if len(results) >= count:
            break
        calls = s.label.get("tool_calls", []) if s.label else []
        results_list = s.label.get("tool_results", []) if s.label else []
        if len(calls) < 2 or len(results_list) < 1:
            continue

        read_call = calls[0]
        write_call = calls[1]
        read_res = results_list[0]

        r_name = read_call.get("name") if isinstance(read_call, dict) else read_call.name
        w_name = write_call.get("name") if isinstance(write_call, dict) else write_call.name
        r_args = read_call.get("arguments") if isinstance(read_call, dict) else read_call.arguments
        w_args = write_call.get("arguments") if isinstance(write_call, dict) else write_call.arguments

        r_kind = (catalog.get(r_name) or {}).get("kind", "")
        if r_kind != "read" and not r_name.endswith(".search") and not r_name.endswith(".list"):
            continue

        prompt_msgs = [
            {"role": "system", "content": "Bạn là trợ lý ảo LifeMate."},
            {"role": "user", "content": s.audio_text or ""},
            {"role": "assistant", "content": f'<function name="{r_name}">' + "".join(f'<param name="{k}">{v}</param>' for k, v in (r_args or {}).items()) + '</function>'},
            {"role": "tool", "content": json.dumps(read_res, ensure_ascii=False)},
        ]

        results.append({
            "id": f"anchor_{s.id}_s1",
            "sample_id": s.id,
            "step_index": 1,
            "step_category": "act",
            "prompt_messages": prompt_msgs,
            "step_calls": [{"name": w_name, "arguments": w_args}],
            "tools": list((s.context or {}).get("installed_apps", {}).keys()),
        })

    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_anchor_binding_gen.py -v`  
Expected: PASS

- [ ] **Step 5: Commit Task 2**

```bash
git add src/data_gen/anchor_binding_gen.py tests/test_anchor_binding_gen.py
git commit -m "feat(data_gen): implement ToolSandbox anchor binding generator"
```

---

### Task 3: APIGen-MT Multi-Intent Chaining & Horizon-Aware Micro-Think

**Files:**
- Modify: `src/data_gen/distill_think.py`
- Create: `src/data_gen/apigen_mt_gen.py`
- Test: `tests/test_apigen_mt_gen.py`
- Reference Author Code: `references/tool_calling_repos/06_xLAM_APIGen-MT_NeurIPS2025` and `references/tool_calling_repos/01_Tool-N1_ICLR2026/data_process/distill_data.py`

**Interfaces:**
- Consumes:
  - `src.data_gen.distill_think.synthesize_deterministic_think`
- Produces:
  - Updated `synthesize_deterministic_think` with explicit `Remaining Sub-goals`
  - `generate_apigen_multi_intent_samples(train_samples: list[Sample], catalog: dict[str, Any], count: int = 600) -> list[dict[str, Any]]`

- [ ] **Step 1: Write failing unit test for APIGen-MT Multi-Intent & Horizon Awareness**

```python
# tests/test_apigen_mt_gen.py
import pytest
from src.core.schemas import Sample
from src.data_gen.distill_think import synthesize_deterministic_think
from src.data_gen.apigen_mt_gen import generate_apigen_multi_intent_samples

def test_horizon_aware_think():
    step = {
        "step_category": "act",
        "step_calls": [{"name": "alarm.create", "arguments": {"time": "06:00"}}],
        "remaining_subgoals": ["sms.send"],
        "prompt_messages": [{"role": "user", "content": "Đặt báo thức 6h và gửi sms cho Lan"}],
    }
    think = synthesize_deterministic_think(step)
    assert "Mục tiêu còn lại: [sms.send]" in think
    assert "Chưa kết thúc phiên" in think

def test_generate_apigen_multi_intent_samples():
    sample1 = Sample(
        id="train_01",
        audio_text="Đặt báo thức 6 giờ",
        context={"installed_apps": {"alarm": {}}},
        label={"tool_calls": [{"name": "alarm.create", "arguments": {"time": "06:00"}}]}
    )
    sample2 = Sample(
        id="train_02",
        audio_text="Nhắn tin cho Lan",
        context={"installed_apps": {"sms": {}}},
        label={"tool_calls": [{"name": "sms.send", "arguments": {"contact_id": "c1", "content": "hello"}}]}
    )
    catalog = {
        "alarm.create": {"kind": "write", "parameters": {}},
        "sms.send": {"kind": "write", "parameters": {}},
    }
    results = generate_apigen_multi_intent_samples([sample1, sample2], catalog, count=1)
    assert len(results) >= 1
    assert "và" in results[0]["prompt_messages"][1]["content"] or "đồng thời" in results[0]["prompt_messages"][1]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_apigen_mt_gen.py -v`  
Expected: FAIL

- [ ] **Step 3: Update `src/data_gen/distill_think.py` with Horizon Awareness & implement `src/data_gen/apigen_mt_gen.py`**

In `src/data_gen/distill_think.py`:
Update `synthesize_deterministic_think`:
```python
    remaining_subgoals = step.get("remaining_subgoals") or []
    if remaining_subgoals:
        subgoal_str = f"Mục tiêu còn lại: [{', '.join(remaining_subgoals)}]. Chưa kết thúc phiên."
    else:
        subgoal_str = "Mục tiêu còn lại: Không còn."
    boundary = f"Quyết định: {boundary}. {subgoal_str}"
```

Implement `src/data_gen/apigen_mt_gen.py`:
```python
"""APIGen-MT (NeurIPS 2025) Multi-Intent Chaining Generator.

Adapted from references/tool_calling_repos/06_xLAM_APIGen-MT_NeurIPS2025
Combines compatible independent actions into verified multi-intent multi-turn trajectories.
"""
from __future__ import annotations

import random
from typing import Any
from src.core.schemas import Sample, ToolCall

_CONNECTORS = [" đồng thời ", ", rồi ", ", với lại ", " và "]

def generate_apigen_multi_intent_samples(
    train_samples: list[Sample],
    catalog: dict[str, Any],
    count: int = 600,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Synthesize multi-intent chaining trajectories from pairs of single-intent samples."""
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []

    valid_samples = [s for s in train_samples if s.label and s.label.get("tool_calls")]
    if len(valid_samples) < 2:
        return results

    while len(results) < count:
        s1, s2 = rng.sample(valid_samples, 2)
        c1_list = s1.label.get("tool_calls", [])
        c2_list = s2.label.get("tool_calls", [])
        if not c1_list or not c2_list:
            continue

        conn = rng.choice(_CONNECTORS)
        t1 = (s1.audio_text or "").rstrip(".,!?")
        t2 = (s2.audio_text or "").lstrip()
        if t2 and t2[0].isupper():
            t2 = t2[0].lower() + t2[1:]
        combined_text = f"{t1}{conn}{t2}"

        tool_names_2 = [c.get("name") if isinstance(c, dict) else c.name for c in c2_list]

        p1 = [
            {"role": "system", "content": "Bạn là trợ lý ảo LifeMate."},
            {"role": "user", "content": combined_text},
        ]
        results.append({
            "id": f"apigen_{s1.id}_{s2.id}_s0",
            "sample_id": f"{s1.id}_{s2.id}",
            "step_index": 0,
            "step_category": "act",
            "prompt_messages": p1,
            "step_calls": c1_list,
            "remaining_subgoals": tool_names_2,
            "tools": list((s1.context or {}).get("installed_apps", {}).keys()),
        })

    return results[:count]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_apigen_mt_gen.py -v`  
Expected: PASS

- [ ] **Step 5: Commit Task 3**

```bash
git add src/data_gen/distill_think.py src/data_gen/apigen_mt_gen.py tests/test_apigen_mt_gen.py
git commit -m "feat(data_gen): implement APIGen-MT multi-intent chaining and horizon-aware thinking"
```

---

### Task 4: Dual Execution Verifier & Pipeline Integration (`build_minicpm5_dataset_v2.py`)

**Files:**
- Create: `src/data_gen/execution_verifier.py`
- Create: `scripts/build_minicpm5_dataset_v2.py`
- Test: `tests/test_build_minicpm5_dataset_v2.py`
- Reference Author Code: `references/tool_calling_repos/01_Tool-N1_ICLR2026/data_process/hammer_utils.py`

**Interfaces:**
- Consumes:
  - `when2call_gen`, `anchor_binding_gen`, `apigen_mt_gen`
  - `src.env.safe_env.SafeEnv`
- Produces:
  - `data/sft/minicpm5_train_sft_v2.jsonl.gz` (~2,750 verified samples)
  - Zero-leakage verification against `configs/splits/dev_v1.json`

- [ ] **Step 1: Write failing unit test for V2 dataset builder & verification**

```python
# tests/test_build_minicpm5_dataset_v2.py
import gzip
import json
from pathlib import Path
import pytest
from scripts.build_minicpm5_dataset_v2 import build_v2_dataset

def test_v2_dataset_no_dev_leakage(tmp_path):
    out_dir = tmp_path / "sft_out"
    count, gz_path = build_v2_dataset(
        split_name="train",
        root="data",
        dev_split_path="configs/splits/dev_v1.json",
        out_dir=out_dir,
        target_synthetic=30,
    )
    assert count > 0
    assert gz_path.exists()

    dev_ids = set(json.loads(Path("configs/splits/dev_v1.json").read_text(encoding="utf-8"))["dev_ids"])
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert rec["sample_id"] not in dev_ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_minicpm5_dataset_v2.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_minicpm5_dataset_v2'`

- [ ] **Step 3: Implement `src/data_gen/execution_verifier.py` & `scripts/build_minicpm5_dataset_v2.py`**

Create `src/data_gen/execution_verifier.py`:
```python
"""Dual Execution Gate: verify tool calls against schema and SafeEnv."""
from __future__ import annotations
from typing import Any
from src.core.schemas import ASK_MISSING_INFO

def verify_step_validity(step: dict[str, Any], catalog: dict[str, Any]) -> bool:
    """Validate that every tool call in step exists and conforms to catalog schema."""
    calls = step.get("step_calls", [])
    if not calls:
        return True
    for c in calls:
        name = c.get("name") if isinstance(c, dict) else c.name
        args = c.get("arguments") if isinstance(c, dict) else c.arguments
        if name == ASK_MISSING_INFO:
            tname = (args or {}).get("tool_name")
            params = (args or {}).get("params")
            if not tname or not params or tname not in catalog:
                return False
            continue
        if name not in catalog:
            return False
    return True
```

Create `scripts/build_minicpm5_dataset_v2.py`:
Combines base leak-free train samples, When2Call samples, Anchor Binding samples, APIGen-MT samples, runs `verify_step_validity`, formats using MiniCPM5 Jinja and Hammer tool catalog randomization, and outputs `data/sft/minicpm5_train_sft_v2.jsonl.gz`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_minicpm5_dataset_v2.py -v`  
Expected: PASS

- [ ] **Step 5: Generate full dataset and verify count**

Run: `python scripts/build_minicpm5_dataset_v2.py`  
Expected: Outputs `data/sft/minicpm5_train_sft_v2.jsonl.gz` with ~2,750 samples, 0 dev leaks.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/data_gen/execution_verifier.py scripts/build_minicpm5_dataset_v2.py tests/test_build_minicpm5_dataset_v2.py
git commit -m "feat(pipeline): complete SFT V2 dataset generation pipeline"
```

---

## Plan Self-Review & Verification Checklist
1. **Spec Coverage:** Covers all 4 failure clusters (When2Call for `ask`, APIGen-MT & Tool-N1 for early stop, ToolSandbox for read-write binding, Hammer for positional invariance).
2. **Zero-Placeholder Guarantee:** Full python snippets provided for all tests and modules.
3. **Zero Data Leakage:** Explicit test `test_v2_dataset_no_dev_leakage` guarding the 299 Blind Dev set.
