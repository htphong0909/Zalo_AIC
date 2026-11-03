# Baseline Pipeline (Qwen3-ASR-0.6B + Qwen3.5-4B) — Roadmap Plan

> **For agentic workers:** This is a phase-level roadmap. Before executing a phase, write its detailed TDD task plan (`docs/plans/YYYY-MM-DD-<phase>.md`) using superpowers:writing-plans, then execute with superpowers:executing-plans or subagent-driven-development.

**Goal:** Deliver a measurable end-to-end baseline and evaluation harness for LifeMate.

**Spec:** [2026-11-03-baseline-pipeline-qwen3-asr-qwen3.5-design.md](../specs/2026-11-03-baseline-pipeline-qwen3-asr-qwen3.5-design.md)

## Global Constraints

- Total params < 6B; LLM < 5B; LLM context ≤ 16,384 tokens.
- Qwen3-ASR-1.7B is development-only (reference); never shipped.
- Never read `<split>_environment/`; at inference only `validate`, `call`, `get_tool_calls`, `get_state_hash`. `get_state` only in train evaluation.
- One ToolEnv per sample at inference.
- Greedy decoding, fixed seeds, pinned versions.
- Weights/data via Hugging Face Hub only; GPU work via `colab` CLI; outputs in `experiments/exp_YYYYMMDD_<slug>/`.

---

## Phase 0 — Compliance gate

| Task | Files | Acceptance |
|---|---|---|
| 0.1 Param counter | `scripts/count_params.py` (reads safetensors headers, sums tensor numels per model, optional exclusion of vision tower prefixes) | Prints exact totals for ASR-0.6B, ASR-1.7B, Qwen3.5-4B full and text-only |
| 0.2 Text-only export | `scripts/export_text_only_llm.py` → push to HF repo `<user>/qwen3.5-4b-text-only` | Loads in vLLM; tensor count < 5B; combined with ASR-0.6B < 6B |
| 0.3 Smoke test on Colab | `scripts/smoke_models.py`, `configs/models.yaml` | Both models load on one GPU, `max_model_len=16384`, one ASR + one tool-call generation succeed; versions pinned in `requirements.txt` |

**Stop condition:** if any budget check fails, revisit model choice before Phase 1.

## Phase 1 — Foundation (CPU-only, local)

| Task | Files | Interfaces produced | Acceptance |
|---|---|---|---|
| 1.1 Data models & loader | `src/core/schemas.py`, `src/core/dataset.py`, `tests/test_dataset.py` | `Sample`, `Turn`, `ToolCall`; `load_split(split, root) -> list[Sample]`; `load_catalog(split, root) -> Catalog` | Loads train/public_test; never touches `*_environment/` |
| 1.2 Safe env wrapper | `src/env/safe_env.py`, `tests/test_safe_env.py` | `SafeEnv(split, sample_id, root)` exposing only `validate`, `call`, `tool_calls`, `state_hash` | Attribute access to anything else raises |
| 1.3 Metrics | `src/eval/metrics.py`, `tests/test_metrics.py` | `token_f1(a, b)`, `match_calls(pred, gold, catalog) -> (M, U)`, `sample_score(...)` | Unit cases: empty gold, fuzzy threshold 0.6, rejected calls in U, one-to-one matching |
| 1.4 Gold-hash replay | `src/eval/gold.py`, `tests/test_gold.py` | `gold_hash(sample) -> str` (train only) | Replaying gold calls on all train samples succeeds |
| 1.5 Dev split | `scripts/make_dev_split.py` → `configs/splits/dev_v1.json` | Fixed list of ~300 ids | Stratified by #calls, out-of-scope, ask-missing, history |
| 1.6 Eval runner | `scripts/run_eval.py`, `src/eval/report.py` | `--predictions file.jsonl --split train --ids dev_v1` → metrics JSON + markdown report | Gold predictions score 1.0; empty predictions score = out-of-scope share |

## Phase 2 — Agent with oracle transcripts

| Task | Files | Acceptance |
|---|---|---|
| 2.1 Tool selector & compactor | `src/agent/tool_selector.py`, `tests/test_tool_selector.py` | Only installed apps + `sys.ask_missing_info`; token count logged |
| 2.2 Prompt & history renderer | `src/agent/prompts.py`, `src/agent/history.py`, tests | Deterministic prompt; fits 16K with truncation policy |
| 2.3 LLM client | `src/agent/llm_client.py` (vLLM offline engine, greedy) | Returns parsed tool calls or final text |
| 2.4 ReAct loop | `src/agent/react_loop.py`, tests with a fake LLM | validate-before-call, max 2 repairs, max 8 steps, 25 s budget |
| 2.5 Oracle run | `scripts/run_agent.py --transcripts oracle` on Colab | Dev metrics + per-category breakdown in `experiments/exp_YYYYMMDD_oracle_baseline/` |

## Phase 3 — ASR

| Task | Files | Acceptance |
|---|---|---|
| 3.1 ASR engine | `src/asr/qwen3_asr_engine.py`, `configs/asr/qwen3_asr_0.6b.yaml` | Batched transcription of current + history turns, greedy |
| 3.2 ASR benchmark | `scripts/run_asr_eval.py` | WER/CER + latency for 0.6B and 1.7B on dev audio |

## Phase 4 — End-to-end & submission

| Task | Files | Acceptance |
|---|---|---|
| 4.1 Pipeline | `src/pipeline.py` | ASR → agent → output per sample, per-stage timings |
| 4.2 Submission tools | `scripts/make_submission.py`, `scripts/verify_submission.py` | public_test file passes `python -m zaic_tools --validate` |
| 4.3 Real-ASR dev run | Colab run | Report: oracle vs real-ASR gap, p95 latency, peak VRAM |

## Phase 5 — Optimization backlog (each item = experiment with dev delta)

1. Few-shot retrieval of similar train trajectories into the prompt.
2. Thinking mode with token budget vs non-thinking (accuracy vs latency).
3. LoRA SFT of Qwen3.5-4B on train trajectories (ASR-noised transcripts), checkpoints on HF.
4. ASR context biasing; ASR fine-tune on train audio.
5. Tool retrieval (Top-K) when installed tools are many.

## Phase 6 — Offline Docker

`Dockerfile`, `scripts/run_inference.sh`: mounts `/data`, writes `result/submission.jsonl`, weights baked in, no network; two identical runs produce identical output.
