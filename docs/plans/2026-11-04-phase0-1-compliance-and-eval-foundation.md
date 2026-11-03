# Phase 0–1: Compliance Gate & Evaluation Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the chosen models fit the competition budget, then build the data loader, safe environment wrapper, official-metric reimplementation, gold-hash replay, dev split and evaluation runner.

**Architecture:** Phase 0 counts parameters from safetensors headers on the Hugging Face Hub (no weight download) and runs a Colab smoke test. Phase 1 is CPU-only and stdlib/pydantic-only: `src/core` (schemas, loader), `src/env` (whitelisted facade over `zaic_tools.ToolEnv`), `src/eval` (metrics, gold replay, report), plus scripts.

**Tech Stack:** Python ≥ 3.10 (local 3.12), pydantic 2, PyYAML, huggingface_hub, pytest; Colab CLI for GPU; vLLM / transformers on Colab only.

**Spec:** [2026-11-03-baseline-pipeline-qwen3-asr-qwen3.5-design.md](../specs/2026-11-03-baseline-pipeline-qwen3-asr-qwen3.5-design.md)

## Global Constraints

- LLM < 5,000,000,000 params; ASR + LLM < 6,000,000,000 params; LLM context ≤ 16,384 tokens.
- Qwen3-ASR-1.7B is reference-only, never shipped.
- No code may open anything under `<split>_environment/`; only `zaic_tools` itself loads state.
- Inference path may use only `validate`, `call`, `get_tool_calls`, `get_state_hash`. `get_state` is allowed only in train-split evaluation scripts.
- Data root is `data/` (contains `train/`, `public_test/`); tests needing data skip when it is absent.
- Outputs go to `experiments/exp_YYYYMMDD_<slug>/`; no files in repo root except project config.
- Weights via Hugging Face Hub only; GPU work via `colab` CLI.

## Metric assumptions (from docs/overview/Evaluation.md — re-check if organisers publish a scorer)

- Match: same tool name; every non-fuzzy argument identical; every `x-match: fuzzy` argument token-F1 > 0.6 after lowercase + NFC, whitespace tokenisation; one-to-one; order ignored.
- Arguments whose value is empty (`None`, `""`, `[]`, `{}`) are treated as absent (same rule as `zaic_tools.validate`).
- P = M / (M + U), R = M / |G|, F1 = 2PR / (P + R); sample score = F1 if state hash matches else 0; empty gold scores 1 iff prediction is empty; final = mean over samples.

---

### Task 1: Project config & parameter counter (Phase 0.1)

**Files:**
- Create: `pyproject.toml`, `src/__init__.py`, `src/core/__init__.py`, `src/env/__init__.py`, `src/eval/__init__.py`, `src/agent/__init__.py`, `src/asr/__init__.py`, `tests/conftest.py`
- Create: `configs/models.yaml`, `scripts/count_params.py`
- Test: `tests/test_count_params.py`

**Interfaces:**
- Produces: `summarize(tensors: dict[str, int], exclude_prefixes: tuple[str, ...] = (), depth: int = 2) -> dict`, `check_budget(asr_params: int, llm_params: int, limits: dict) -> list[str]`, `fetch_tensor_counts(repo_id: str, revision: str | None = None) -> dict[str, int]`.

- [ ] **Step 1: Create project config and package markers**

`pyproject.toml`:
```toml
[project]
name = "zalo-aic-lifemate"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["pydantic>=2", "pyyaml>=6", "huggingface_hub>=0.30"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

Empty files: `src/__init__.py`, `src/core/__init__.py`, `src/env/__init__.py`, `src/eval/__init__.py`, `src/agent/__init__.py`, `src/asr/__init__.py`.

`tests/conftest.py`:
```python
from pathlib import Path

import pytest

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def data_root() -> Path:
    if not (DATA_ROOT / "train" / "train.jsonl").is_file():
        pytest.skip("competition data not available under data/")
    return DATA_ROOT
```

`configs/models.yaml`:
```yaml
asr:
  repo_id: Qwen/Qwen3-ASR-0.6B
  exclude_prefixes: []
llm:
  repo_id: Qwen/Qwen3.5-4B
  exclude_prefixes: []        # set after Step 6 shows the vision-tower prefix
reference_asr:
  repo_id: Qwen/Qwen3-ASR-1.7B
  exclude_prefixes: []
limits:
  llm_max_params: 5000000000
  total_max_params: 6000000000
  llm_max_context: 16384
```

- [ ] **Step 2: Write the failing tests**

`tests/test_count_params.py`:
```python
from scripts.count_params import check_budget, summarize

LIMITS = {"llm_max_params": 5_000_000_000, "total_max_params": 6_000_000_000}


def test_summarize_totals_and_groups():
    tensors = {"model.layers.0.w": 10, "model.layers.1.w": 20, "visual.blocks.0.w": 5}
    out = summarize(tensors, exclude_prefixes=("visual.",), depth=2)
    assert out["total"] == 35
    assert out["excluded"] == 5
    assert out["kept"] == 30
    assert out["groups"] == {"model.layers": 30, "visual.blocks": 5}


def test_summarize_without_exclusion_keeps_everything():
    out = summarize({"a.b": 3, "c.d": 4})
    assert out["excluded"] == 0 and out["kept"] == 7


def test_check_budget_passes_within_limits():
    assert check_budget(900_000_000, 4_200_000_000, LIMITS) == []


def test_check_budget_flags_llm_and_total():
    errors = check_budget(2_100_000_000, 5_000_000_000, LIMITS)
    assert any("LLM" in e for e in errors)
    assert any("total" in e for e in errors)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_count_params.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.count_params'`.

- [ ] **Step 4: Implement `scripts/count_params.py`**

```python
"""Count model parameters from safetensors headers on the Hugging Face Hub (no weight download)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml


def summarize(tensors: dict[str, int], exclude_prefixes: tuple[str, ...] = (), depth: int = 2) -> dict:
    total = sum(tensors.values())
    excluded = sum(n for name, n in tensors.items() if name.startswith(exclude_prefixes))
    groups: dict[str, int] = defaultdict(int)
    for name, n in tensors.items():
        groups[".".join(name.split(".")[:depth])] += n
    return {
        "total": total,
        "excluded": excluded,
        "kept": total - excluded,
        "groups": dict(sorted(groups.items(), key=lambda kv: -kv[1])),
    }


def check_budget(asr_params: int, llm_params: int, limits: dict) -> list[str]:
    errors = []
    if llm_params >= limits["llm_max_params"]:
        errors.append(f"LLM has {llm_params:,} params (limit < {limits['llm_max_params']:,})")
    if asr_params + llm_params >= limits["total_max_params"]:
        errors.append(f"total {asr_params + llm_params:,} params (limit < {limits['total_max_params']:,})")
    return errors


def fetch_tensor_counts(repo_id: str, revision: str | None = None) -> dict[str, int]:
    from huggingface_hub import HfApi

    meta = HfApi().get_safetensors_metadata(repo_id, revision=revision)
    return {name: info.parameter_count for fm in meta.files_metadata.values() for name, info in fm.tensors.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--out", default=None, help="optional JSON report path")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    report = {}
    for role in ("asr", "llm", "reference_asr"):
        entry = cfg[role]
        summary = summarize(fetch_tensor_counts(entry["repo_id"]), tuple(entry.get("exclude_prefixes", [])), args.depth)
        report[role] = {"repo_id": entry["repo_id"], **summary}
        print(f"\n== {role}: {entry['repo_id']}")
        print(f"total={summary['total']:,} excluded={summary['excluded']:,} kept={summary['kept']:,}")
        for group, n in list(summary["groups"].items())[:15]:
            print(f"  {group:<40} {n:>15,}")

    errors = check_budget(report["asr"]["kept"], report["llm"]["kept"], cfg["limits"])
    ref_errors = check_budget(report["reference_asr"]["kept"], report["llm"]["kept"], cfg["limits"])
    print("\nSubmission system:", "OK" if not errors else "; ".join(errors))
    print("With reference ASR (info only):", "OK" if not ref_errors else "; ".join(ref_errors))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
```

Also create empty `scripts/__init__.py` so tests can import `scripts.count_params`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_count_params.py -v`
Expected: 4 passed.

- [ ] **Step 6: Run the real count and fix the exclusion prefix**

Run: `python scripts/count_params.py --out experiments/exp_20261104_param_budget/report.json`
(If the local network blocks the Hub, run the same command on Colab: `colab exec "cd /content/ZALO_AIC && python scripts/count_params.py"`.)
- If a repo id 404s, find the correct id on the Hub (e.g. a `-hf` suffix) and update `configs/models.yaml`.
- Read the `groups` listing of `llm`; put the vision-tower prefix(es) into `llm.exclude_prefixes`. If MTP tensors (e.g. `mtp.*`) are present and the runtime does not use them, decide explicitly whether to exclude them and record why.
- Re-run. Expected: `Submission system: OK`; reference ASR line expected to fail the total budget.

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml src/ tests/conftest.py tests/test_count_params.py scripts/__init__.py scripts/count_params.py configs/models.yaml
git commit -m "feat: add parameter budget counter and project config"
```

---

### Task 2: Colab smoke test & text-only LLM decision (Phase 0.2–0.3, spike)

This task is an investigation. Its output is a findings note plus pinned versions; code written here is kept only if it becomes `scripts/smoke_models.py`.

**Files:**
- Create: `scripts/smoke_models.py`, `requirements.txt`, `docs/notes/2026-11-04-phase0-findings.md`

- [ ] **Step 1: Provision Colab and install**

```powershell
colab new --gpu
colab upload src /content/ZALO_AIC/src
colab upload scripts /content/ZALO_AIC/scripts
colab upload configs /content/ZALO_AIC/configs
colab upload pyproject.toml /content/ZALO_AIC/pyproject.toml
colab exec "pip install -U vllm transformers huggingface_hub qwen-asr pyyaml"
colab exec "nvidia-smi && pip freeze | grep -Ei 'vllm|transformers|torch|qwen'"
```
(Never upload `data/` as a whole — it contains `*_environment/`. Upload code only; the single wav in Step 3 is the only data file.)

- [ ] **Step 2: Write `scripts/smoke_models.py`**

```python
"""Smoke test: load ASR and LLM on one GPU, run one transcription and one tool-calling generation."""
from __future__ import annotations

import argparse
import json
import time


def run_llm(model_id: str, max_len: int) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_id, max_model_len=max_len, seed=0, gpu_memory_utilization=0.6)
    tools = [{
        "type": "function",
        "function": {
            "name": "smart_home.light_set",
            "description": "Bật/tắt đèn",
            "parameters": {"type": "object", "properties": {
                "device_id": {"type": "string"}, "power": {"type": "string", "enum": ["on", "off"]}},
                "required": ["device_id", "power"]},
        },
    }]
    messages = [{"role": "user", "content": "Tắt đèn bếp (device_id: light_kitchen)."}]
    t0 = time.time()
    out = llm.chat(messages, SamplingParams(temperature=0.0, max_tokens=256), tools=tools)
    print(json.dumps({"llm_seconds": round(time.time() - t0, 2), "text": out[0].outputs[0].text}, ensure_ascii=False))


def run_asr(model_id: str, wav: str) -> None:
    from transformers import pipeline

    t0 = time.time()
    pipe = pipeline("automatic-speech-recognition", model=model_id, device_map="auto")
    print(json.dumps({"asr_seconds": round(time.time() - t0, 2), "text": pipe(wav)["text"]}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--llm", required=True)
    p.add_argument("--asr", required=True)
    p.add_argument("--wav", required=True)
    p.add_argument("--max-len", type=int, default=16384)
    a = p.parse_args()
    run_asr(a.asr, a.wav)
    run_llm(a.llm, a.max_len)
```

- [ ] **Step 3: Run the smoke test**

Upload one train wav (audio only, never `*_environment/`), e.g. `colab upload data/train/audio/train_0008/train_0008.wav /content/sample.wav`, then:
```powershell
colab exec "cd /content/ZALO_AIC && python scripts/smoke_models.py --asr Qwen/Qwen3-ASR-0.6B --llm Qwen/Qwen3.5-4B --wav /content/sample.wav"
```
Expected: one Vietnamese transcript; one tool call to `smart_home.light_set` with `light_kitchen` / `off`. If the ASR pipeline path fails, try the `qwen-asr` package API from the model card and update `run_asr`.

- [ ] **Step 4: Decide the text-only path**

Investigate, in this order, and record which works:
1. vLLM option that skips the vision encoder when all multimodal limits are 0.
2. A text-only causal-LM class/config in transformers that loads Qwen3.5 weights without the vision tower; save it and push to HF `<user>/qwen3.5-4b-text-only`.
Acceptance: the shipped checkpoint's stored tensors (re-run Task 1 with that repo id) total < 5B and ASR + LLM < 6B, and the smoke test passes on it.

- [ ] **Step 5: Pin versions and write findings**

`requirements.txt`: exact versions from `pip freeze` for `torch`, `vllm`, `transformers`, `huggingface_hub`, `qwen-asr` (if used), `pydantic`, `pyyaml`.
`docs/notes/2026-11-04-phase0-findings.md`: GPU type, versions, param report (from Task 1), load time, smoke latencies, peak VRAM (`nvidia-smi`), chosen text-only path, open issues.

- [ ] **Step 6: Stop Colab and commit**

```powershell
colab stop <session_id>
git add scripts/smoke_models.py requirements.txt docs/notes/2026-11-04-phase0-findings.md configs/models.yaml
git commit -m "chore: phase 0 smoke test, pinned versions and findings"
```

**Gate:** do not start Task 3 if the budget check fails; report to the user instead.

---

### Task 3: Data schemas & loader (Phase 1.1)

**Files:**
- Create: `src/core/schemas.py`, `src/core/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Produces: `ToolCall(name: str, arguments: dict)`, `Turn`, `Label`, `Sample` (with `.allowed_tools -> set[str]`); `load_split(split: str, root: str | Path) -> list[Sample]`; `load_catalog(split: str, root: str | Path) -> dict[str, dict]` (tool name → tool spec); `audio_path(split: str, root: str | Path, rel: str) -> Path`.

- [ ] **Step 1: Write the failing tests**

`tests/test_dataset.py`:
```python
import json
from pathlib import Path

import pytest

from src.core.dataset import audio_path, load_catalog, load_split
from src.core.schemas import Sample

ROW = {
    "id": "train_9999",
    "audio": "audio/train_9999/train_9999.wav",
    "audio_text": "Tắt đèn bếp",
    "context": {"current_time": "2026-11-16T17:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh",
                "installed_apps": {"smart_home": {"tools": ["smart_home.get_status", "smart_home.light_set"]}}},
    "history": [{"role": "user", "audio": "audio/train_9999/train_9999_u1.wav"},
                {"role": "assistant", "text": "OK", "tool_calls": [{"name": "smart_home.get_status", "arguments": {}}]}],
    "label": {"tool_calls": [{"name": "smart_home.light_set", "arguments": {"device_id": "light_kitchen", "power": "off"}}],
              "tool_results": [{"ok": True}], "final_state": {}},
}


def test_sample_parses_and_exposes_allowed_tools():
    s = Sample.model_validate(ROW)
    assert s.history[1].tool_calls[0].name == "smart_home.get_status"
    assert s.label.tool_calls[0].arguments["power"] == "off"
    assert s.allowed_tools == {"smart_home.get_status", "smart_home.light_set", "sys.ask_missing_info"}


def test_load_split_from_tmp(tmp_path: Path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "train.jsonl").write_text(json.dumps(ROW, ensure_ascii=False) + "\n", encoding="utf-8")
    samples = load_split("train", tmp_path)
    assert [s.id for s in samples] == ["train_9999"]


def test_audio_path_rejects_environment_dir(tmp_path: Path):
    with pytest.raises(PermissionError):
        audio_path("train", tmp_path, "train_environment/train_0001/state.json")


def test_real_data_counts(data_root: Path):
    assert len(load_split("train", data_root)) == 1545
    assert len(load_split("public_test", data_root)) == 634
    catalog = load_catalog("train", data_root)
    assert len(catalog) == 147 and "sys.ask_missing_info" in catalog
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.dataset'`.

- [ ] **Step 3: Implement `src/core/schemas.py`**

```python
"""Typed views of the competition jsonl records."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

ASK_MISSING_INFO = "sys.ask_missing_info"


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    audio: str | None = None
    audio_text: str | None = None
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[Any] = Field(default_factory=list)


class Label(BaseModel):
    model_config = ConfigDict(extra="allow")
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[Any] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)


class Sample(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    audio: str
    context: dict[str, Any]
    history: list[Turn] = Field(default_factory=list)
    audio_text: str | None = None
    label: Label | None = None

    @property
    def allowed_tools(self) -> set[str]:
        apps = self.context.get("installed_apps", {})
        return {t for app in apps.values() for t in app.get("tools", [])} | {ASK_MISSING_INFO}
```

- [ ] **Step 4: Implement `src/core/dataset.py`**

```python
"""Loaders for split jsonl, tool catalog and audio paths. Never touches <split>_environment/."""
from __future__ import annotations

import json
from pathlib import Path

from src.core.schemas import Sample

FORBIDDEN_DIR_SUFFIX = "_environment"


def _split_dir(split: str, root: str | Path) -> Path:
    return Path(root) / split


def load_split(split: str, root: str | Path) -> list[Sample]:
    path = _split_dir(split, root) / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Sample.model_validate_json(line) for line in f if line.strip()]


def load_catalog(split: str, root: str | Path) -> dict[str, dict]:
    path = _split_dir(split, root) / f"{split}_tools_catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {tool["name"]: tool for app in data["apps"] for tool in app["tools"]}


def audio_path(split: str, root: str | Path, rel: str) -> Path:
    path = _split_dir(split, root) / rel
    if any(part.endswith(FORBIDDEN_DIR_SUFFIX) for part in path.parts):
        raise PermissionError(f"access to {FORBIDDEN_DIR_SUFFIX}/ is forbidden: {path}")
    return path
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```powershell
git add src/core/schemas.py src/core/dataset.py tests/test_dataset.py
git commit -m "feat: add sample schemas and split/catalog loaders"
```

---

### Task 4: Safe environment wrapper (Phase 1.2)

**Files:**
- Create: `src/env/safe_env.py`
- Test: `tests/test_safe_env.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `load_toolenv_factory(split: str, root: str | Path) -> Callable` (imports `<root>/<split>/zaic_tools`); `SafeEnv(split: str, sample_id: str, root: str | Path)` with exactly `validate(name, arguments) -> dict`, `call(name, arguments) -> dict`, `get_tool_calls() -> list[dict]`, `get_state_hash() -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_safe_env.py`:
```python
from pathlib import Path

from src.core.dataset import load_split
from src.env.safe_env import SafeEnv


def test_safe_env_exposes_only_permitted_methods():
    public = {n for n in dir(SafeEnv) if not n.startswith("_")}
    assert public == {"validate", "call", "get_tool_calls", "get_state_hash"}


def test_safe_env_validate_call_log_and_hash(data_root: Path):
    sample = load_split("train", data_root)[0]
    env = SafeEnv("train", sample.id, data_root)
    assert env.validate("no_such_app.tool", {})["ok"] is False
    assert env.get_tool_calls() == []  # validate does not log
    env.call("sys.ask_missing_info", {"tool_name": "sms.send", "params": ["contact_id"]})
    assert env.get_tool_calls() == [{"name": "sys.ask_missing_info",
                                     "arguments": {"tool_name": "sms.send", "params": ["contact_id"]}}]
    h = env.get_state_hash()
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_safe_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.env.safe_env'`.

- [ ] **Step 3: Implement `src/env/safe_env.py`**

```python
"""Inference-time facade over zaic_tools.ToolEnv exposing only the four permitted methods."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable


def load_toolenv_factory(split: str, root: str | Path) -> Callable[..., Any]:
    """Import the split's own zaic_tools package (<root>/<split>/zaic_tools). One split per process."""
    split_dir = (Path(root) / split).resolve()
    loaded = sys.modules.get("zaic_tools")
    if loaded is not None:
        if Path(loaded.__file__).resolve().parent.parent != split_dir:
            raise RuntimeError(f"zaic_tools already loaded from {loaded.__file__}; use one split per process")
        return loaded.ToolEnv
    sys.path.insert(0, str(split_dir))
    return importlib.import_module("zaic_tools").ToolEnv


class SafeEnv:
    __slots__ = ("_env",)

    def __init__(self, split: str, sample_id: str, root: str | Path):
        self._env = load_toolenv_factory(split, root)(split, sample_id, root=str(root))

    def validate(self, name: str, arguments: dict | None = None) -> dict:
        return self._env.validate(name, arguments or {})

    def call(self, name: str, arguments: dict | None = None) -> dict:
        return self._env.call(name, arguments or {})

    def get_tool_calls(self) -> list[dict]:
        return self._env.get_tool_calls()

    def get_state_hash(self) -> str:
        return self._env.get_state_hash()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_safe_env.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/env/safe_env.py tests/test_safe_env.py
git commit -m "feat: add SafeEnv wrapper restricted to permitted ToolEnv methods"
```

---

### Task 5: Official metric reimplementation (Phase 1.3)

**Files:**
- Create: `src/eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `token_f1(pred, gold) -> float`; `fuzzy_params(tool_spec: dict | None) -> set[str]`; `calls_match(pred: dict, gold: dict, catalog: dict[str, dict], threshold: float = 0.6) -> bool`; `count_matches(pred_calls: list[dict], gold_calls: list[dict], catalog) -> int`; `SampleResult` dataclass (`score, state_match, matched, unmatched_pred, n_gold, precision, recall, f1`); `score_sample(pred_calls, pred_hash, gold_calls, gold_hash, catalog) -> SampleResult`.

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
import unicodedata

import pytest

from src.eval.metrics import calls_match, count_matches, score_sample, token_f1

CATALOG = {
    "sms.send": {"parameters": {"properties": {
        "contact_id": {"type": "string"}, "content": {"type": "string", "x-match": "fuzzy"}}}},
    "smart_home.light_set": {"parameters": {"properties": {
        "device_id": {"type": "string"}, "power": {"type": "string"}}}},
}


def sms(content, contact="c_01"):
    return {"name": "sms.send", "arguments": {"contact_id": contact, "content": content}}


def test_token_f1_basic_and_nfc():
    assert token_f1("gọi cho mẹ", "gọi lại cho mẹ") == pytest.approx(6 / 7)
    assert token_f1("Hương", unicodedata.normalize("NFD", "hương")) == 1.0
    assert token_f1("", "") == 1.0
    assert token_f1("a", "") == 0.0


def test_fuzzy_threshold():
    assert calls_match(sms("gọi cho mẹ"), sms("gọi lại cho mẹ"), CATALOG)
    assert not calls_match(sms("mẹ"), sms("gọi lại cho mẹ"), CATALOG)


def test_non_fuzzy_must_be_identical_and_empty_args_ignored():
    gold = {"name": "smart_home.light_set", "arguments": {"device_id": "light_kitchen", "power": "off"}}
    assert calls_match({"name": "smart_home.light_set",
                        "arguments": {"device_id": "light_kitchen", "power": "off", "color": ""}}, gold, CATALOG)
    assert not calls_match({"name": "smart_home.light_set",
                            "arguments": {"device_id": "light_living", "power": "off"}}, gold, CATALOG)
    assert not calls_match(sms("x", "c_02"), sms("x", "c_01"), CATALOG)


def test_matching_is_one_to_one_and_maximal():
    gold = [sms("a b c"), sms("a b d")]
    preds = [sms("a b c d"), sms("a c")]  # greedy would pair pred0-gold0 and leave pred1 unmatched
    assert count_matches(preds, gold, CATALOG) == 2
    assert count_matches([sms("a b c"), sms("a b c")], [sms("a b c")], CATALOG) == 1


def test_score_sample_rules():
    g = [sms("xin chào")]
    assert score_sample([], "h", [], "h", CATALOG).score == 1.0
    assert score_sample(g, "h", [], "h", CATALOG).score == 0.0
    assert score_sample(g, "h", g, "h", CATALOG).score == 1.0
    mismatch = score_sample(g, "h1", g, "h2", CATALOG)
    assert mismatch.score == 0.0 and mismatch.f1 == 1.0
    extra = score_sample(g + [sms("khác hẳn", "c_09")], "h", g, "h", CATALOG)
    assert extra.precision == 0.5 and extra.recall == 1.0
    assert extra.score == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.metrics'`.

- [ ] **Step 3: Implement `src/eval/metrics.py`**

```python
"""Local reimplementation of the LifeMate metric (see docs/overview/Evaluation.md)."""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

EMPTY = (None, "", [], {})


def _tokens(value: Any) -> list[str]:
    return unicodedata.normalize("NFC", str(value)).lower().split()


def token_f1(pred: Any, gold: Any) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p and not g:
        return 1.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def fuzzy_params(tool_spec: dict | None) -> set[str]:
    props = (tool_spec or {}).get("parameters", {}).get("properties", {})
    return {name for name, spec in props.items() if spec.get("x-match") == "fuzzy"}


def _clean(arguments: dict | None) -> dict:
    return {k: v for k, v in (arguments or {}).items() if v not in EMPTY}


def calls_match(pred: dict, gold: dict, catalog: dict[str, dict], threshold: float = 0.6) -> bool:
    if pred.get("name") != gold.get("name"):
        return False
    fuzzy = fuzzy_params(catalog.get(gold["name"]))
    p, g = _clean(pred.get("arguments")), _clean(gold.get("arguments"))
    if set(p) != set(g):
        return False
    for key, gold_value in g.items():
        if key in fuzzy:
            if token_f1(p[key], gold_value) <= threshold:
                return False
        elif p[key] != gold_value:
            return False
    return True


def count_matches(pred_calls: list[dict], gold_calls: list[dict], catalog: dict[str, dict]) -> int:
    """Maximum one-to-one matching (Kuhn's augmenting paths)."""
    adj = [[j for j, g in enumerate(gold_calls) if calls_match(p, g, catalog)] for p in pred_calls]
    owner = [-1] * len(gold_calls)

    def assign(i: int, seen: set[int]) -> bool:
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] == -1 or assign(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(assign(i, set()) for i in range(len(pred_calls)))


@dataclass(frozen=True)
class SampleResult:
    score: float
    state_match: bool
    matched: int
    unmatched_pred: int
    n_gold: int
    precision: float
    recall: float
    f1: float


def score_sample(pred_calls: list[dict], pred_hash: str, gold_calls: list[dict], gold_hash: str,
                 catalog: dict[str, dict]) -> SampleResult:
    state_match = pred_hash == gold_hash
    if not gold_calls:
        ok = float(not pred_calls)
        return SampleResult(ok, state_match, 0, len(pred_calls), 0, ok, ok, ok)
    matched = count_matches(pred_calls, gold_calls, catalog)
    unmatched = len(pred_calls) - matched
    precision = matched / len(pred_calls) if pred_calls else 0.0
    recall = matched / len(gold_calls)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return SampleResult(f1 if state_match else 0.0, state_match, matched, unmatched, len(gold_calls),
                        precision, recall, f1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/eval/metrics.py tests/test_metrics.py
git commit -m "feat: reimplement LifeMate tool-call matching and sample score"
```

---

### Task 6: Gold-hash replay & label consistency check (Phase 1.4)

**Files:**
- Create: `src/eval/gold.py`, `scripts/check_gold_replay.py`
- Test: `tests/test_gold.py`

**Interfaces:**
- Consumes: `Sample` (Task 3), `load_toolenv_factory` (Task 4).
- Produces: `replay_gold(sample: Sample, root: str | Path) -> GoldReplay` where `GoldReplay(state_hash: str, results: list[dict], changed_ok: bool)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_gold.py`:
```python
from pathlib import Path

import pytest

from src.core.dataset import load_split
from src.core.schemas import Sample
from src.eval.gold import replay_gold


def test_replay_requires_label(data_root: Path):
    s = load_split("train", data_root)[0].model_copy(update={"label": None})
    with pytest.raises(ValueError):
        replay_gold(s, data_root)


def test_replay_first_50_train_samples(data_root: Path):
    for sample in load_split("train", data_root)[:50]:
        g = replay_gold(sample, data_root)
        assert len(g.state_hash) == 64
        assert all(r.get("ok") for r in g.results), sample.id
        assert g.changed_ok, sample.id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.gold'`.

- [ ] **Step 3: Implement `src/eval/gold.py`**

```python
"""EVALUATION ONLY (train split): replay gold calls to obtain the gold state hash. Never import from inference code."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.schemas import Sample
from src.env.safe_env import load_toolenv_factory


@dataclass(frozen=True)
class GoldReplay:
    state_hash: str
    results: list[dict]
    changed_ok: bool  # get_state()[app] == label.final_state[app] for every app in final_state


def replay_gold(sample: Sample, root: str | Path) -> GoldReplay:
    if sample.label is None:
        raise ValueError(f"{sample.id} has no label; gold replay is train-only")
    env = load_toolenv_factory("train", root)("train", sample.id, root=str(root))
    results = [env.call(c.name, c.arguments) for c in sample.label.tool_calls]
    state = env.get_state()  # allowed: train split, evaluation only
    changed_ok = all(state.get(app) == st for app, st in sample.label.final_state.items())
    return GoldReplay(env.get_state_hash(), results, changed_ok)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gold.py -v`
Expected: 2 passed. If `changed_ok` fails for some samples, inspect one manually and adjust the comparison (e.g. canonicalisation) — do not loosen silently; document the reason in the docstring.

- [ ] **Step 5: Implement `scripts/check_gold_replay.py` and run on all train**

```python
"""Replay every train label and report failures (rejected gold calls, final_state mismatches)."""
from __future__ import annotations

import argparse

from src.core.dataset import load_split
from src.eval.gold import replay_gold

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    a = p.parse_args()
    bad_calls, bad_state, samples = [], [], load_split("train", a.root)
    for s in samples:
        g = replay_gold(s, a.root)
        if not all(r.get("ok") for r in g.results):
            bad_calls.append(s.id)
        if not g.changed_ok:
            bad_state.append(s.id)
    print(f"samples={len(samples)} rejected_gold_calls={len(bad_calls)} final_state_mismatch={len(bad_state)}")
    print("examples:", bad_calls[:10], bad_state[:10])
```

Run: `python -m scripts.check_gold_replay --root data`
Expected: `rejected_gold_calls=0 final_state_mismatch=0` (any non-zero count is reported to the user before continuing).

- [ ] **Step 6: Commit**

```powershell
git add src/eval/gold.py scripts/check_gold_replay.py tests/test_gold.py
git commit -m "feat: add train-only gold replay and label consistency check"
```

---

### Task 7: Stratified dev split (Phase 1.5)

**Files:**
- Create: `src/eval/splits.py`, `scripts/make_dev_split.py`, `configs/splits/dev_v1.json` (generated)
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: `Sample`, `ASK_MISSING_INFO` (Task 3).
- Produces: `stratum(sample: Sample) -> str`; `make_dev_split(samples: list[Sample], dev_size: int = 300, seed: int = 2026) -> list[str]`; file `configs/splits/dev_v1.json` = `{"seed", "dev_ids", "train_ids"}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_splits.py`:
```python
from collections import Counter

from src.core.schemas import Sample
from src.eval.splits import make_dev_split, stratum


def mk(i: int, n_calls: int, ask: bool = False, history: bool = False) -> Sample:
    calls = [{"name": "sys.ask_missing_info" if ask else "a.b", "arguments": {}}] * n_calls
    hist = [{"role": "user", "audio": "x.wav"}] if history else []
    return Sample.model_validate({"id": f"s{i:04d}", "audio": "a.wav", "context": {}, "history": hist,
                                  "label": {"tool_calls": calls}})


def test_stratum_labels():
    assert stratum(mk(0, 0)) == "oos|0|n"
    assert stratum(mk(1, 1, ask=True, history=True)) == "ask|1|h"
    assert stratum(mk(2, 3)) == "act|2-3|n"
    assert stratum(mk(3, 5)) == "act|4+|n"


def test_split_is_deterministic_and_proportional():
    samples = [mk(i, i % 5, history=bool(i % 2)) for i in range(1000)]
    a, b = make_dev_split(samples, 200, 7), make_dev_split(samples, 200, 7)
    assert a == b
    assert abs(len(a) - 200) <= 10
    dev = {s.id for s in samples if s.id in set(a)}
    by_stratum = Counter(stratum(s) for s in samples if s.id in dev)
    full = Counter(stratum(s) for s in samples)
    for key, n in full.items():
        assert abs(by_stratum[key] / len(a) - n / len(samples)) < 0.02
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_splits.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.splits'`.

- [ ] **Step 3: Implement `src/eval/splits.py`**

```python
"""Deterministic stratified dev split of the train set."""
from __future__ import annotations

import random
from collections import defaultdict

from src.core.schemas import ASK_MISSING_INFO, Sample


def stratum(sample: Sample) -> str:
    calls = sample.label.tool_calls if sample.label else []
    n = len(calls)
    kind = "oos" if n == 0 else "ask" if any(c.name == ASK_MISSING_INFO for c in calls) else "act"
    size = "0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else "4+"
    return f"{kind}|{size}|{'h' if sample.history else 'n'}"


def make_dev_split(samples: list[Sample], dev_size: int = 300, seed: int = 2026) -> list[str]:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        groups[stratum(s)].append(s.id)
    dev: list[str] = []
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        dev.extend(ids[: round(len(ids) * dev_size / len(samples))])
    return sorted(dev)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_splits.py -v`
Expected: 2 passed.

- [ ] **Step 5: Implement `scripts/make_dev_split.py` and generate the split**

```python
"""Write configs/splits/dev_v1.json from the train labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.dataset import load_split
from src.eval.splits import make_dev_split

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--size", type=int, default=300)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="configs/splits/dev_v1.json")
    a = p.parse_args()
    samples = load_split("train", a.root)
    dev = make_dev_split(samples, a.size, a.seed)
    dev_set = set(dev)
    out = {"seed": a.seed, "dev_ids": dev, "train_ids": sorted(s.id for s in samples if s.id not in dev_set)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"dev={len(dev)} train={len(out['train_ids'])} -> {a.out}")
```

Run: `python -m scripts.make_dev_split`
Expected: `dev≈300 train≈1245`.

- [ ] **Step 6: Commit**

```powershell
git add src/eval/splits.py scripts/make_dev_split.py configs/splits/dev_v1.json tests/test_splits.py
git commit -m "feat: add stratified dev split (dev_v1)"
```

---

### Task 8: Evaluation runner & report (Phase 1.6)

**Files:**
- Create: `src/eval/evaluator.py`, `scripts/run_eval.py`, `scripts/make_gold_predictions.py`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: `load_split`, `load_catalog` (Task 3), `replay_gold` (Task 6), `score_sample`, `SampleResult` (Task 5), `stratum` (Task 7).
- Produces: `evaluate(predictions: dict[str, dict], samples: list[Sample], catalog: dict, root) -> dict[str, SampleResult]`; `summarize_results(results: dict[str, SampleResult], samples: list[Sample]) -> dict`; CLI `python -m scripts.run_eval --predictions <jsonl> [--ids configs/splits/dev_v1.json] --out <dir>` writing `metrics.json` and `per_sample.jsonl`.

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluator.py`:
```python
from pathlib import Path

import pytest

from src.core.dataset import load_catalog, load_split
from src.eval.evaluator import evaluate, summarize_results
from src.eval.gold import replay_gold


@pytest.fixture(scope="module")
def subset(data_root: Path):
    return load_split("train", data_root)[:40], load_catalog("train", data_root)


def test_gold_predictions_score_one(subset, data_root: Path):
    samples, catalog = subset
    preds = {s.id: {"tool_calls": [c.model_dump() for c in s.label.tool_calls],
                    "state_hash": replay_gold(s, data_root).state_hash} for s in samples}
    summary = summarize_results(evaluate(preds, samples, catalog, data_root), samples)
    assert summary["final_score"] == pytest.approx(1.0)


def test_empty_predictions_score_equals_oos_share(subset, data_root: Path):
    samples, catalog = subset
    summary = summarize_results(evaluate({}, samples, catalog, data_root), samples)
    oos = sum(1 for s in samples if not s.label.tool_calls) / len(samples)
    assert summary["final_score"] == pytest.approx(oos)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evaluator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.evaluator'`.

- [ ] **Step 3: Implement `src/eval/evaluator.py`**

```python
"""Score predictions (submission-format rows) against train labels."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from src.core.schemas import Sample
from src.eval.gold import replay_gold
from src.eval.metrics import SampleResult, score_sample
from src.eval.splits import stratum


def evaluate(predictions: dict[str, dict], samples: list[Sample], catalog: dict, root: str | Path) -> dict[str, SampleResult]:
    results = {}
    for s in samples:
        pred = predictions.get(s.id, {"tool_calls": [], "state_hash": ""})
        gold_calls = [c.model_dump() for c in s.label.tool_calls]
        results[s.id] = score_sample(pred.get("tool_calls", []), pred.get("state_hash", ""), gold_calls,
                                     replay_gold(s, root).state_hash, catalog)
    return results


def summarize_results(results: dict[str, SampleResult], samples: list[Sample]) -> dict:
    by_kind: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        kind, _, hist = stratum(s).split("|")
        by_kind[kind].append(results[s.id].score)
        by_kind["history" if hist == "h" else "no_history"].append(results[s.id].score)
    action = [results[s.id] for s in samples if s.label.tool_calls]
    return {
        "n": len(samples),
        "final_score": mean(r.score for r in results.values()),
        "state_match_rate": mean(r.state_match for r in action) if action else None,
        "precision": mean(r.precision for r in action) if action else None,
        "recall": mean(r.recall for r in action) if action else None,
        "f1": mean(r.f1 for r in action) if action else None,
        "by_category": {k: {"n": len(v), "score": mean(v)} for k, v in sorted(by_kind.items())},
    }


def per_sample_rows(results: dict[str, SampleResult]) -> list[dict]:
    return [{"id": sid, **asdict(r)} for sid, r in results.items()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluator.py -v`
Expected: 2 passed.

- [ ] **Step 5: Implement CLIs**

`scripts/run_eval.py`:
```python
"""Evaluate a submission-format jsonl on (a subset of) train."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.dataset import load_catalog, load_split
from src.eval.evaluator import evaluate, per_sample_rows, summarize_results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--root", default="data")
    p.add_argument("--ids", default=None, help="split json with dev_ids (default: all train)")
    p.add_argument("--out", required=True, help="experiments/exp_YYYYMMDD_<slug>/eval")
    a = p.parse_args()
    samples = load_split("train", a.root)
    if a.ids:
        keep = set(json.loads(Path(a.ids).read_text(encoding="utf-8"))["dev_ids"])
        samples = [s for s in samples if s.id in keep]
    with open(a.predictions, encoding="utf-8") as f:
        preds = {row["id"]: row for row in map(json.loads, f) if row}
    results = evaluate(preds, samples, load_catalog("train", a.root), a.root)
    summary = summarize_results(results, samples)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "per_sample.jsonl", "w", encoding="utf-8") as f:
        for row in per_sample_rows(results):
            f.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
```

`scripts/make_gold_predictions.py` (sanity baseline for the runner):
```python
"""Write gold-label predictions for train in submission format (runner sanity check)."""
from __future__ import annotations

import argparse
import json

from src.core.dataset import load_split
from src.eval.gold import replay_gold

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    with open(a.out, "w", encoding="utf-8") as f:
        for s in load_split("train", a.root):
            row = {"id": s.id, "tool_calls": [c.model_dump() for c in s.label.tool_calls],
                   "state_hash": replay_gold(s, a.root).state_hash}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

- [ ] **Step 6: End-to-end sanity run**

```powershell
python -m scripts.make_gold_predictions --out experiments/exp_20261104_eval_sanity/gold.jsonl
python -m scripts.run_eval --predictions experiments/exp_20261104_eval_sanity/gold.jsonl --ids configs/splits/dev_v1.json --out experiments/exp_20261104_eval_sanity/eval
python -m pytest -v
```
Expected: dev `final_score` = 1.0; full test suite green.

- [ ] **Step 7: Commit and push**

```powershell
git add src/eval/evaluator.py scripts/run_eval.py scripts/make_gold_predictions.py tests/test_evaluator.py
git commit -m "feat: add evaluation runner with per-category report"
git push origin main
```

---

## Exit criteria for Phase 0–1

- Param report shows ASR + LLM (as shipped) < 6B and LLM < 5B; versions pinned; smoke test passed.
- `python -m pytest` passes locally.
- Gold replay: 0 rejected gold calls, 0 final_state mismatches (or documented exceptions).
- Gold predictions score 1.0 on `dev_v1`; empty predictions score = out-of-scope share.
