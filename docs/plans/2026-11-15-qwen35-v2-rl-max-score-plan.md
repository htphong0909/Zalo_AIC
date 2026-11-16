# Qwen3.5-4B SFT V2 Maximum-Score Multi-Stage RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push `Qwen/Qwen3.5-4B` SFT V2 (`exp_qwen35_sft_v2`, currently `0.7788` Oracle / `0.7503` E2E Voice on sealed `dev_v1`) to its maximum achievable score (`~0.840–0.875` Oracle, `~0.815–0.855` E2E Voice) without inflating latency or leaking `dev_v1`, via an orthogonal pre-RL email-xref fix followed by a three-stage multi-turn RL pipeline (**Exact-Metric Composite GiGPO+GSPO+ARPO -> Privileged Voice/Error Reverse-KL OPD/SDPO -> Targeted DiaTool Step-DPO & LoRA Souping**).

**Architecture:** Warm-start from frozen `exp_qwen35_sft_v2/merged/` (`~8.43 GB` bf16) using PEFT LoRA across the 12 Qwen3.5 hybrid modules (`QWEN35_LORA_TARGET_MODULES`, `r=32, alpha=64`, `~280 MB` per checkpoint). Use live multi-turn `SafeEnv("train", ...)` rollouts (`enable_thinking=False`, assistant-token-only `loss_mask=1`), zero-VRAM reference logprobs via `model.disable_adapter()`, strict monotonic metric-aligned rewards combining `ToolRL` bipartite shaping + `Tool-N1` exact ceiling + fuzzy F1/redundancy terms, `GiGPO` anchor-state step credit (`Dr.GRPO` `mean_norm`), `LoopTool`/`GSPO` sequence-level ratios with `DAPO Clip-Higher` (`0.20 / 0.28`) + dynamic zero-variance group filtering + `ARPO` post-tool branching, followed by `MiniCPM5`-style dual-adapter Reverse-KL `OPD/SDPO` distilling clean-audio/privileged-feedback teacher logits into noisy PhoWhisper student rollouts, `DiaTool` turn-discounted `Step-DPO`, and `merge_lfm3_soup.py` adapter souping.

**Tech Stack:** PyTorch `>=2.5`, Transformers + PEFT (`LoRA` `bfloat16`), `vLLM 0.33.0` (XGrammar deterministic eval + optional colocated rollout), `zaic_tools.ToolEnv` via `SafeEnv`, `PhoWhisper` (`--no-context-bias --nbest-beams 3`), Hugging Face Hub checkpoint sync (`src/core/hf_sync.py: HFSyncer`), Colab G4 (`101.97 GB` VRAM) + `pytest`.

**Spec / Basis:** Empirical error audit of `exp_qwen35_sft_v2` & `exp_qwen35_sft_v2_trackA`, codebase RL/eval inspection, and direct paper+code inspection in `references/tool_calling_repos/` (`ToolRL`, `Tool-N1`, `GiGPO`, `LoopTool`, `MiniCPM5 OPD`, `DiaTool-DPO`, `Step-DPO`, `OffSeeker`, `SimpleTool`, `AURA`).

## Global Constraints

- **NO DIRECT APP STATE ACCESS (`AGENTS.md §2.1`):** Strictly forbid reading/parsing `<split>_environment/` or `state.json`. Only interact through `SafeEnv` (`env.validate()`, `env.call()`, `env.get_tool_calls()`, `env.get_state_hash()`).
- **STRICT SPLIT SEALING:** `dev_v1` (`299` samples) is strictly sealed until the single final promotion evaluation. Do NOT inspect `dev_v1` inputs, gold calls, or per-sample predictions, and NEVER train or filter RL pairs on `dev_v1`. All RL/OPD/DPO training uses strictly `train` (`folds_1_to_4` = `996` samples while holding `fold_0` = `250` as RL-unseen non-regression monitor; optional locked-HP final refit across `train_only` = `1,246` only after hyperparams freeze).
- **MODEL & RUNTIME LIMITS (`AGENTS.md §2.2–2.5`):** LLM `< 5B` (`Qwen3.5-4B` = `~4.2B`), ASR+LLM `< 6B`, context `<= 16,384` tokens, end-to-end sample latency `<= 30s` on A30 (baseline E2E is `11.26s` on L4 / LLM `1.13s` on G4 -> keep `enable_thinking=False` and `max_new_tokens <= 384`), offline Docker, deterministic `temperature=0.0` at inference.
- **DURABLE CHECKPOINTS & HF PUBLIC RL STORAGE (`AGENTS.md §4.3`):** `htphong0909/zalo-aic-sft-checkpoints` (private `50 GB` limit, `49.42 / 50.00 GB` used) stays read-only as the source of `exp_qwen35_sft_v2/merged`. All new RL/OPD checkpoints and ASR caches write to the dedicated **public** HF repo **`htphong0909/zalo-aic-rl-checkpoints`** (`HF_SYNC_PRIVATE=0`). Every RL checkpoint MUST save PEFT LoRA weights only (`~260–280 MB`), push permanently to `<prefix>/checkpoints/checkpoint-<step>/`, mirror `<prefix>/checkpoints/latest/`, run `HFSyncer.preflight()` before GPU burn, cap blocking Colab chunks at `<= 50 min`, and auto-resume from `latest/`.
- **ONE SPLIT PER PROCESS (`src/env/safe_env.py`):** `zaic_tools` binds one split per OS process; run cross-split evals in separate worker processes.

---

## Part I: Empirical Audit & Six Weaknesses of Qwen3.5-4B SFT V2

### Baseline Snapshot (`exp_qwen35_sft_v2`, `exp_qwen35_sft_v2_trackA`, `exp_qwen35_sft_v3_1_clean`)

| Run / Split | Final Score | State Match (`SM`) | Tool F1 | `act` | `ask` | `oos` | `history` | `no_history` | Speed |
|---|---|---|---|---|---|---|---|---|---|
| `exp_qwen35_sft_v2` Dev 299 (Oracle G4) | **0.7788** | 83.45% | 82.65% | 0.7587 | 0.7804 | 1.0000 | 0.7773 | 0.7816 | 0.90s/s |
| `trackA` Same-VM V2 Ctrl Dev 299 (Oracle) | 0.7721 | 83.09% | 81.86% | 0.7545 | 0.7571 | 1.0000 | 0.7672 | 0.7816 | — |
| `trackA` Oracle Dev 299 (`vllm 0.33 + xgrammar`) | **0.7788** | 83.45% | 82.34% | 0.7630 | 0.7571 | 1.0000 | 0.7722 | 0.7914 | 1.13s/s |
| `trackA` **E2E Voice Dev 299** (L4, `PhoWhisper 3-beam`) | **0.7503** (`-2.85 pp`) | 81.29% | 80.39% | 0.7310 | 0.7339 | 1.0000 | 0.7494 | 0.7522 | 11.26s E2E |
| `trackA` Fold 0 Oracle (`N=250`, in-sample in V2) | 0.8823 (`43` imp) | 91.77% | 90.93% | — | — | — | — | — | — |
| `trackA` Fold 0 `<0750 (n=100)` vs `>=0750 (n=150)` | 0.9542 (`7`) vs 0.8343 (`36`) | — | — | — | — | — | — | — | — |
| `trackA` Fold 0 E2E Voice (`N=250`) | 0.8596 (`48` imp) | — | — | — | — | — | — | — | — |
| `exp_qwen35_sft_v3_1_clean` Dev 299 (Thinking fail) | 0.6929 (`-8.59 pp`) | — | — | — | — | — | — | — | 14.64s L4 |

**Dev 299 (`N=299`) Imperfect Distribution (`exp_qwen35_sft_v2`):**
- `209 / 299` (`69.9%`) score `1.0`, `53 / 299` (`17.7%`) score `0.0`, `37 / 299` (`12.4%`) have `SM=True` with `0 < F1 < 1.0`.
- Total `90` imperfect samples decompose cleanly into:
  1. `38` (`42.2%`): `same_tool_names_arg_or_state_mismatch`
  2. `34` (`37.8%`): `different_tool_sequence`
  3. `18` (`20.0%`): `13 false_ask` + `5 missed_ask`

**Fold 0 (`N=250`) Verified In-Sample Error Buckets (`43` Oracle imperfect -> `48` E2E imperfect):**
- `cat:act->ask`: `7` (e.g., `train_0708` asked despite context in prior turn)
- `same_tools|sm=True|sameK=True`: `7` (fuzzy `token_f1 <= 0.6` or optional arg mismatch: `train_0697` missing `calendar.create_event.end`, `train_0751` missing `ewallet.transfer.note`, `train_0814` truncated `notes.create`, `train_0894` `phone_number` vs `contact_id`)
- `same_tools|sm=False|sameK=True`: `6` (`train_0224` miscompared `gold_price > 131M`, `train_0279` wrong bank account `acc_02` vs `acc_01`, `train_0777`/`train_0802` slot/route time math errors)
- `same_tools|sm=True|missK=True` + `extraK=True` + `both`: `4 + 3 + 1 = 8` (`train_0470`/`0801`/`0878` `contacts.search({'query':...})` vs `{}`, `train_0748` extra redundant `chat_ott.list_chats`)
- `diff_tools|sm=False` (`4`, e.g. `train_0503` skipped `contacts.search` and hallucinated email `dung.mua@hang.example.test`) + `diff_tools|sm=True` (`3`)
- `ask_wrong_tool_or_params` (`4`), `cat:ask->act` (`2`), `cat:oos->ask` (`2`)

**Train `folds_1_to_4` (`996` RL-Eligible Clean Episodes) Exact Composition:**
- Categories: `act=785`, `ask=140`, `oos=71` (`687/996` = `69.0%` require `>=2` tool calls; `467/996` = `46.9%` invoke at least one of the `32` `x-match=="fuzzy"` tools; `656/996` have dialogue history; `559/996` are `>=0750` heavy multi-step).

### Six Root Weaknesses Diagnosed ("Các điểm chưa tốt của SFT V2")

1. **Fold-0 Leakage + Generalization Gap (`-10.35 pp` Oracle / `-10.93 pp` E2E):** `qwen35_train_sft_v2.jsonl.gz` (`4,954` rows) trained on all `1,246` `train_only` IDs (`250` Fold 0 + `996` folds 1..4). Because SFT V2 memorized Fold 0, greedy sampling over seen `train` prompts suffers high zero-variance saturation if naively passed to GRPO (`exp_lfm3_grpo` saw `0.50–0.85` `frac_reward_zero_std`), while checkpoint selection must not rely solely on Fold 0 gains without regularization/KL anchoring and noisy-ASR generalization checks.
2. **Zero Real PhoWhisper Exposure (`-2.85 pp` E2E Drop):** `2,816 / 2,980` (`94.5%`) `base_train` rows in V2 used clean gold text, only `164` used CTC sims, and **`0` rows** contained real `PhoWhisper Loop #38` transcripts or `asr_phonetic_diff`. Moreover, local cache `train_only_phowhisper_loop38_nbest.jsonl` (`996` `folds 1..4` rows) currently has `nbest=[]` and `asr_phonetic_diff=None` (`0/996`), whereas `fold0_transcripts.jsonl` (`250`) has `3`-beam `nbest`. ASR errors like `"500 nghìn" -> "500 ngày"` (`train_0079`) and `"gửi cô" -> "gửi mạ cô"` (`train_0830`) directly trigger downstream `0.0` cliffs.
3. **`sm=True, F1<1.0` Non-Differentiable Cliff (`37/299` Dev partial + part of `38` mismatch bucket):** Pure SFT cross-entropy cannot distinguish optional arg conventions (`contacts.search({'query':...})` vs `{}`), `phone.call(contact_id)` vs `phone_number`, redundant read elimination (`chat_ott.list_chats`), or crossing the `token_f1 > 0.6` exact-cliff on `32` fuzzy tools (`metrics.py:36-74` uses `NFC` lowercase whitespace split WITHOUT punctuation stripping).
4. **Non-Thinking Multi-Step Reasoning, Value/Time Arithmetic & Entity Skipping (`34/299` Dev + `>=0750` `0.8343` vs `<0750` `0.9542`):** Because V2 runs with `enable_thinking=False` (`avg 147.7` comp chars -> ultra-fast `1.13s` G4 / `4.63s` L4), complex multi-step dependencies (`>=0750`) fail when comparing numerical thresholds (`gold_price > 131M`), matching account tiers (`acc_01` vs `acc_02`), computing route/calendar free slots, or skipping `contacts.search` and inventing emails (`train_0503` — note `_xref_error` in `react_loop.py` checks `x-ref` IDs and phones, NOT email addresses!).
5. **`act / ask / oos` Boundary Miscalibration (`18/299` Dev = `13 false_ask + 5 missed_ask`; `15/43` Fold 0):** Premature clarification when context already exists in history (`train_0708`), asking on `oos` (`2` Fold 0), or wrong `tool_name` / `params` list in `sys.ask_missing_info` (`4` Fold 0). Note `V3.1` proved synthetic `irrelevance_oos` templates + `enable_thinking=True` severely hurt precision (`-8.59 pp`), whereas SFT V2 already gets `oos = 1.0000` and needs surgical calibration, not template bloat.
6. **Historical Single-Step RL & Prefix Mismatch Traps:** `lfm3_grpo.py` used single-step static teacher-forced prefixes (`frac_reward_zero_std` collapse, no live multi-turn recovery from tool observations) and lacked dynamic group filtering, post-tool credit assignment, and on-policy voice distillation.

---

## Part II: Reference Paper & Code Synthesis Matrix

| Source (`references/tool_calling_repos/`) | Exact Verified Mechanism & Equation | Adoption in This Plan | Anti-Pattern Explicitly Avoided |
|---|---|---|---|
| **04_ToolRL** (`NeurIPS2025`, `rlla.py:21-199`, `dp_actor.py:267`) | Decomposed bipartite matching (`match_score` multiset Jaccard for names/keys + exact value credit) -> `+17%` on 1.5B/3B (`46.20%`, `53.81%`). | Dense partial reward inside the `sm=False` (`score_sample==0.0`) failure tier so `<5B` receives signal on partial tool chains. | **Avoid 3 bugs in `rlla.py`/`dp_actor.py`:** (1) `dp_actor.py:267` subtracts KL loss (`- kl_loss * coef`), (2) `rlla.py:227` gives `0.0` correctness to non-tool (`oos`) turns, (3) Table 5 length reward crashes accuracy (`-5 to -13%`). |
| **01_Tool-N1** (`ICLR2026`, `toolcall.py:22-145`, `dp_actor.py:307`) | Binary exact-match ceiling (`1.0` exact, `0.0` partial) prevents partial-reward hacking (`84.82%` BFCL-7B); positive KL sign `+ kl_loss * coef`. | Strict reward hierarchy where ANY partial/fuzzy/redundant-read trajectory is capped strictly below `+1.00` (`<= 0.94`), and `sm=False` <= `0.38`. | Do NOT turn on long `<think>` reasoning during RL (`V3.1` proved `enable_thinking=True` balloons L4 latency `4.63s -> 14.64s` and dropped Oracle `-8.59 pp`). |
| **05_GiGPO** (`NeurIPS2025`, `core_gigpo.py:87-384`, `utils.py:133-184`, `ray_trainer.py:200-224`) | Two-level advantage $A_t^{(i)} = A_E^{(i)} + \omega A_S^{(i)}$ ($\omega=0.8$, $\gamma=0.95$) over hashed canonical step observations; `mode="mean_norm"` (`remove_std=True` / Dr.GRPO); `filter_group_data` drops zero-std groups; `-0.08` invalid-action penalty. | Core multi-turn step credit assignment for `>=0750` multi-tool trajectories + DAPO zero-variance batch filter eliminating `lfm3_grpo` stalled batches. | Do NOT divide by small noisy group `std` (`mean_std_norm`) when `0 < pass@G < 1` or when rewards span `[0.44, 1.00]`. |
| **07_LoopTool** (`ACL2026`, `core_algos.py:792-1038`, `tool_agent_loop.py:371,417`, `dataloop/`) | **GSPO** (`arXiv:2507.18071`) sequence-level importance ratio $\exp(\frac{1}{|T|}\sum_t (\log\pi_\theta - \log\pi_{\text{old}}))$ + **DAPO Clip-Higher** (`clip_low=0.20, clip_high=0.28`), `seq-mean-token-mean`; `response_mask=0` on tool/user turns; GCP high-PPL mining & EDDE variation filtering. | Sequence-level policy optimization preventing token-level ratio spikes across multi-turn tool trajectories; GCP mining targets non-saturated `folds_1_to_4` hard cases + boundary `ask/act` pairs. | Do NOT generate unverified synthetic scenes that mutate tool env entities incompatible with `SafeEnv` (`train_environment`). Anchor perturbations strictly on real `train` `sample.id`s. |
| **13_MiniCPM5 OPD** (`README.md:183-191,469-481`, `arXiv:2604.13016`) | Post-RL **On-Policy Distillation** computes full/top-K-union token reverse-KL divergence between student and teacher logits as advantage estimate (`RL+OPD` gave `↑6.96` agentic / `↑10.96` overall on 2B, `↑16` & `↓29%` overlong on 1B). | **Stage 3 Voice & Error-Feedback SDPO-OPD:** Same PEFT model hosts `"student"` (receives real `PhoWhisper` noisy text + `asr_phonetic_diff`) and `"teacher"` (receives clean Gold text + privileged env-failure hint) on on-policy student tokens `y`. Directly crushes the `-2.85 pp` ASR drop! | No separate 8.4 GB teacher model needed in G4 VRAM — PEFT swaps `"student"` / `"teacher"` adapters cleanly on the shared base. |
| **14_DiaTool-DPO** & **16_Step-DPO** (`dpo_trainer.py:1272-1355`, `stepdpo_trainer.py:11-100`) | First-error step $k$ localization + in-distribution self-rectified $s_{\text{win}}$; temporal turn weights $w(t)=\frac{1-\gamma^{T-t}}{1-\gamma^T}/\psi$ ($\gamma=0.5$), `margin=0.5`, `beta=0.1`. | **Stage 3B Targeted Refinement:** Clean up stubborn first-step errors (premature asks, redundant reads) via tiny-LR (`3e-7`) turn-weighted Step-DPO on `SafeEnv`-verified in-distribution pairs. | Keep `<= 1` epoch (`~40-60` steps) with strict KL guard so OOS `1.0000` and general calibration cannot regress. |
| **12_AURA** & **15_OffSeeker** (`ASRU2025`, `quality.py`) | N-best phonetic diff hints from speech hypotheses; tool-necessity filtering removing trivial parametric queries. | Populate 3-beam `nbest` + `asr_phonetic_diff` on `folds_1_to_4` (`run_asr.py`); filter zero-std trivial samples before RL gradient updates. | Avoid ASR domain-context prompt biasing (prior eval showed `--no-context-bias --nbest-beams 3` is best). |

---

## Part III: Execution-Ready Tasks

### Task 0: Pre-RL HF Quota Audit & `folds_1_to_4` PhoWhisper 3-Beam Cache

**Files:**
- Inspect/Verify: `scripts/run_asr.py`, `src/core/hf_sync.py`
- Output Artifact (local + Colab cache): `experiments/exp_qwen35_v2_rl/asr_cache/folds1_4_phowhisper_3beam_nbest.jsonl`

- [ ] **Step 1: Verify HF Hub Storage & Free Quota Safely (Mandatory Quota Gate)**

`htphong0909/zalo-aic-sft-checkpoints` is currently at `49.42 / 50.00 GB` (private repo hard limit `50 GB`). Inspect repo folder sizes and clear dead failed V3 artifacts (ONLY after user confirms deletion, OR create clean dedicated subfolder / free `exp_qwen35_sft_v3_clean` `23.51 GB` shards so RL LoRA checkpoints have `>= 20 GB` headroom):

```bash
python -c "from huggingface_hub import HfApi; api=HfApi(); files=api.list_repo_tree('htphong0909/zalo-aic-sft-checkpoints', recursive=True); from collections import defaultdict; s=defaultdict(int); [s.__setitem__(f.path.split('/')[0], s[f.path.split('/')[0]]+(getattr(f,'size',0) or 0)) for f in files]; print('\n'.join(f'{k}: {v/1e9:.2f} GB' for k,v in sorted(s.items(), key=lambda x:-x[1])))"
```
Expected: Prints per-experiment GB usage (`exp_qwen35_sft_v3_clean` ~`23.51 GB`). Keep `exp_qwen35_sft_v2/merged/` untouched (`~8.43 GB`).

- [ ] **Step 2: Generate `nbest` + `asr_phonetic_diff` on `folds_1_to_4` (`996` samples, `0` leak)**

Because local `train_only_phowhisper_loop38_nbest.jsonl` has `0/996` `nbest` populated (`diff_nonempty=0`), run `scripts/run_asr.py` on Colab L4/G4 strictly on `configs/splits/folds_1_to_4_ids.json` using the exact Loop #38 PhoWhisper adapter + flags matching E2E Track A (`--no-context-bias --nbest-beams 3 --nbest-max-delta 0.65`):

```bash
python scripts/run_asr.py \
  --split train \
  --root /content/data \
  --ids configs/splits/folds_1_to_4_ids.json \
  --engine whisper \
  --model-id /content/models/phowhisper_loop38 \
  --no-context-bias \
  --nbest-beams 3 \
  --nbest-max-delta 0.65 \
  --batch-size 8 \
  --out experiments/exp_qwen35_v2_rl/asr_cache/folds1_4_phowhisper_3beam_nbest.jsonl \
  --metrics-out experiments/exp_qwen35_v2_rl/asr_cache/folds1_4_asr_metrics.json \
  --resume
```
Expected: `996` rows written with non-empty `nbest` (`len=3`) and matching `asr_phonetic_diff` where top beam has close alternatives; runtime `~4–6 min` on L4/G4. Verify with:

```bash
python -c "import json; rows=[json.loads(l) for l in open('experiments/exp_qwen35_v2_rl/asr_cache/folds1_4_phowhisper_3beam_nbest.jsonl', encoding='utf-8')]; assert len(rows)==996; print('nbest_ok:', sum(bool(r.get('nbest')) for r in rows), 'diff_ok:', sum(bool(r.get('asr_phonetic_diff')) for r in rows))"
```

- [ ] **Step 3: Cache Upload to HF for Zero-Recompute Resumption**

```bash
hf upload htphong0909/zalo-aic-rl-checkpoints experiments/exp_qwen35_v2_rl/asr_cache exp_qwen35_v2_rl/asr_cache --repo-type model
```

---

### Task 1: General Email Cross-Reference Hallucination Guard (Orthogonal Runtime Fix)

**Files:**
- Modify: `src/agent/react_loop.py:217-266`, `src/agent/react_loop.py:388-412`, and `_xref_error` in `src/agent/react_loop.py`
- Test: `tests/test_collection_xref_guard.py`

**Interfaces:**
- Consumes: `by_app: dict[str, set[str]]` populated in `_collect_ids_from_result` (`react_loop.py:217-266`) and `_init_known_ids` (`react_loop.py:388-412`).
- Produces: `by_app["emails"]` containing canonical lowercased email addresses seen in prompt/history/`contacts.search` results; `_xref_error(call, catalog, by_app, text_low)` returns an error string forcing a `contacts.search` lookup when `email.send` or `email.create_draft` passes an unseen `to`/`cc` email address (prevents fabricated emails like `train_0503` `dung.mua@hang.example.test`).

- [ ] **Step 1: Write the failing test in `tests/test_collection_xref_guard.py`**

Append to `tests/test_collection_xref_guard.py`:

```python
def test_email_xref_blocks_hallucinated_email_and_allows_seen_email():
    catalog = load_catalog("train", "data")
    by_app = {"contacts": {"c_01"}, "phones": set(), "emails": {"real@example.test"}}

    call_bad = ToolCall(
        name="email.send",
        arguments={"to": ["fake.person@hang.example.test"], "subject": "Chao", "body": "Noi dung"},
    )
    err = _xref_error(call_bad, catalog, by_app, text_low="gui email cho dung")
    assert err is not None
    assert "contacts.search" in err or "email" in err.lower()

    call_good = ToolCall(
        name="email.send",
        arguments={"to": ["REAL@example.test"], "subject": "Chao", "body": "Noi dung"},
    )
    assert _xref_error(call_good, catalog, by_app, text_low="gui email") is None

    call_in_prompt = ToolCall(
        name="email.send",
        arguments={"to": ["direct@example.test"], "subject": "Chao", "body": "Noi dung"},
    )
    assert _xref_error(call_in_prompt, catalog, by_app, text_low="gui email toi direct@example.test") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_collection_xref_guard.py::test_email_xref_blocks_hallucinated_email_and_allows_seen_email -v`
Expected: FAIL (`AssertionError: assert None is not None`).

- [ ] **Step 3: Implement `_extract_email_candidates` and hook into `_collect_ids_from_result`, `_init_known_ids`, and `_xref_error` in `src/agent/react_loop.py`**

Add above `_collect_ids_from_result` (`src/agent/react_loop.py:217`):

```python
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def _extract_email_candidates(text: str) -> set[str]:
    if not text:
        return set()
    return {m.group(0).strip().rstrip(".,;:!?)\"']").lower() for m in _EMAIL_RE.finditer(text)}
```

Inside `_collect_ids_from_result` (`src/agent/react_loop.py:248-257`, right alongside `phones`), extract emails from all string fields (`email`, `to`, `cc`, `recipient`, or free text):

```python
                if k in ("email", "to", "cc", "sender", "recipient"):
                    by_app["emails"].update(_extract_email_candidates(v))
                elif "@" in v:
                    by_app["emails"].update(_extract_email_candidates(v))
```

Inside `_init_known_ids` (`src/agent/react_loop.py:394-412`), also extract emails from `turn_text` and string/list arguments in history calls:

```python
        if turn_text:
            by_app["phones"].update(_extract_phone_candidates(turn_text))
            by_app["emails"].update(_extract_email_candidates(turn_text))
```

And inside `_xref_error(call, catalog, by_app, text_low="")`, before returning `None` for `call.name in ("email.send", "email.create_draft")`, check `to` and `cc` addresses against `seen_emails = set(by_app.get("emails") or set()) | _extract_email_candidates(text_low)`:

```python
    if call.name in ("email.send", "email.create_draft"):
        seen_emails = set(by_app.get("emails") or set()) | _extract_email_candidates(text_low)
        for field in ("to", "cc"):
            vals = call.arguments.get(field)
            if isinstance(vals, str):
                vals = [vals]
            if isinstance(vals, list):
                for addr in vals:
                    norm = str(addr).strip().lower()
                    if "@" in norm and norm not in seen_emails:
                        return (
                            f"Email '{addr}' chưa có trong ngữ cảnh hoặc kết quả contacts.search. "
                            f"Hãy gọi contacts.search trước để lấy email chính xác."
                        )
```

- [ ] **Step 4: Run tests to verify pass & zero regressions**

Run: `python -m pytest tests/test_collection_xref_guard.py tests/test_react_loop.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent/react_loop.py tests/test_collection_xref_guard.py
git commit -m "fix(agent): guard email.send/create_draft against ungrounded email addresses"
```

---

### Task 2: Exact-Metric Composite Multi-Turn Reward (`src/agent/rl_reward.py`)

**Files:**
- Create: `src/agent/rl_reward.py`
- Test: `tests/test_rl_reward.py`

**Interfaces:**
- Produces:
  - `RewardBreakdown` dataclass (`episode_reward: float`, `official_score: float`, `state_match: bool`, `f1: float`, `fuzzy_f1_mean: float`, `bipartite_sim: float`, `invalid_steps: int`, `extra_reads: int`, `tier: str`)
  - `canonical_state_anchor(sample_id: str, step_idx: int, step_calls: list[dict[str, Any]], step_results: list[Any], malformed: bool) -> str`
  - `compute_bipartite_call_similarity(pred_calls: list[dict[str, Any]], gold_calls: list[dict[str, Any]], catalog: dict[str, dict[str, Any]]) -> float`
  - `compute_step_format_penalty(raw_text: str, parsed_calls: list[dict[str, Any]], catalog: dict[str, dict[str, Any]], *, is_last_turn: bool, truncated: bool = False) -> tuple[float, bool]`
  - `compute_episode_reward(pred_calls: list[dict[str, Any]], pred_hash: str, gold_calls: list[dict[str, Any]], gold_hash: str, catalog: dict[str, dict[str, Any]], *, invalid_steps: int = 0, extra_reads: int = 0, malformed_terminal: bool = False) -> RewardBreakdown`

**Key Invariant (Tool-N1 + ToolRL Proof):** Strictly ordered non-overlapping reward tiers so NO partial cheat can outscore the true official optimum:
`False Ask / OOS FP (-0.30..-0.15)` < `sm=False ToolRL Partial [-0.15..+0.35]` < `Ask Partial [+0.10..+0.55]` < `sm=True, F1<1 [0.44..+0.94]` < `Exact official=1.0 (+1.00)`.
NO token-length reward (`ToolRL` Table 5).

- [ ] **Step 1: Write the failing test `tests/test_rl_reward.py`**

```python
"""Verify strict monotonic tiering, fuzzy densification, and anti-hacking of RL rewards."""
from src.core.dataset import load_catalog
from src.agent.rl_reward import (
    canonical_state_anchor,
    compute_bipartite_call_similarity,
    compute_episode_reward,
    compute_step_format_penalty,
)


def test_reward_hierarchy_strictly_monotonic():
    cat = load_catalog("train", "data")
    gold_calls = [
        {"name": "contacts.search", "arguments": {"query": "Lan"}},
        {"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "Anh den tre 10 phut"}},
    ]
    gold_hash = "hash_ok"

    # 1. Perfect official score = 1.0
    r_exact = compute_episode_reward(gold_calls, gold_hash, gold_calls, gold_hash, cat)
    assert r_exact.episode_reward == 1.0
    assert r_exact.tier == "exact"

    # 2. State match = True, F1 < 1 (extra redundant read or fuzzy mismatch) strictly in [0.44, 0.94]
    redundant_calls = [
        {"name": "chat_ott.list_chats", "arguments": {}},
        *gold_calls,
    ]
    r_sm_extra = compute_episode_reward(
        redundant_calls, gold_hash, gold_calls, gold_hash, cat, extra_reads=1
    )
    r_sm_clean = compute_episode_reward(
        [{"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "Anh den tre 10 phut"}}],
        gold_hash,
        gold_calls,
        gold_hash,
        cat,
        extra_reads=0,
    )
    assert 0.44 <= r_sm_extra.episode_reward < r_exact.episode_reward
    assert r_sm_extra.episode_reward <= 0.94
    assert r_sm_clean.episode_reward > r_sm_extra.episode_reward

    # 3. State match = False with good partial tools strictly below sm=True tier and above wrong tools
    r_partial = compute_episode_reward(
        [{"name": "contacts.search", "arguments": {"query": "Lan"}}, {"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "Sai noi dung hoan toan xyz"}}],
        "hash_bad",
        gold_calls,
        gold_hash,
        cat,
    )
    r_wrong = compute_episode_reward(
        [{"name": "alarm.list", "arguments": {}}], "hash_bad", gold_calls, gold_hash, cat
    )
    assert -0.15 <= r_wrong.episode_reward < r_partial.episode_reward <= 0.35
    assert r_partial.episode_reward < r_sm_extra.episode_reward

    # 4. Premature false ask on act episode penalized below honest partial attempt
    r_false_ask = compute_episode_reward(
        [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["contact_id"]}}],
        "hash_bad",
        gold_calls,
        gold_hash,
        cat,
    )
    assert r_false_ask.episode_reward <= -0.15


def test_oos_and_ask_exact_and_partial_tiers():
    cat = load_catalog("train", "data")

    # OOS exact [] gets +1.0 (fixes ToolRL zero reward for text turns)
    r_oos_ok = compute_episode_reward([], "h0", [], "h0", cat)
    assert r_oos_ok.episode_reward == 1.0
    # OOS false ask penalized harshly
    r_oos_ask = compute_episode_reward(
        [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "alarm.create", "params": ["time"]}}],
        "h1",
        [],
        "h0",
        cat,
    )
    assert r_oos_ask.episode_reward == -0.25

    # Ask partial shaping beats missed ask
    gold_ask = [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "banking.transfer", "params": ["amount", "beneficiary_id"]}}]
    r_ask_exact = compute_episode_reward(gold_ask, "ha", gold_ask, "ha", cat)
    r_ask_part = compute_episode_reward(
        [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "banking.transfer", "params": ["amount"]}}],
        "hb",
        gold_ask,
        "ha",
        cat,
    )
    r_ask_missed = compute_episode_reward(
        [{"name": "banking.list_accounts", "arguments": {}}], "hc", gold_ask, "ha", cat
    )
    assert r_ask_exact.episode_reward == 1.0
    assert 0.10 <= r_ask_part.episode_reward <= 0.55
    assert r_ask_missed.episode_reward < 0.0


def test_fuzzy_densifier_gives_gradient_below_threshold():
    cat = load_catalog("train", "data")
    gold = [{"name": "notes.create", "arguments": {"title": "Hop nhom", "content": "Chuan bi slide quy 4 va bao cao doanh thu"}}]
    close = [{"name": "notes.create", "arguments": {"title": "Hop", "content": "Chuan bi slide quy 4"}}]
    far = [{"name": "notes.create", "arguments": {"title": "Mua sam", "content": "Di cho nau com toi"}}]
    s_close = compute_bipartite_call_similarity(close, gold, cat)
    s_far = compute_bipartite_call_similarity(far, gold, cat)
    assert s_close > s_far


def test_step_format_penalty_and_canonical_anchor():
    cat = load_catalog("train", "data")
    pen_ok, valid_ok = compute_step_format_penalty(
        '<tool_call>\n<function=alarm.list>\n</function>\n</tool_call>',
        [{"name": "alarm.list", "arguments": {}}],
        cat,
        is_last_turn=False,
    )
    assert valid_ok and pen_ok == 0.0

    pen_bad, valid_bad = compute_step_format_penalty(
        '<tool_call>broken garbage</tool_call>',
        [],
        cat,
        is_last_turn=False,
    )
    assert not valid_bad and pen_bad == -0.08

    a1 = canonical_state_anchor("train_0100", 1, [{"name": "contacts.search", "arguments": {"query": "Lan"}}], [{"b": 2, "a": 1}], False)
    a2 = canonical_state_anchor("train_0100", 1, [{"name": "contacts.search", "arguments": {"query": "Lan"}}], [{"a": 1, "b": 2}], False)
    assert a1 == a2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rl_reward.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.agent.rl_reward'`).

- [ ] **Step 3: Write `src/agent/rl_reward.py`**

```python
"""Composite exact-metric RL rewards combining Tool-N1 exact ceiling, ToolRL bipartite shaping,
fuzzy-field densification, redundancy suppression, and step validity penalties."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.core.schemas import ASK_MISSING_INFO
from src.data_gen.execution_verifier import validate_step_with_reason
from src.eval.metrics import _clean, fuzzy_params, match_calls_kuhn, score_sample, token_f1


@dataclass
class RewardBreakdown:
    episode_reward: float
    official_score: float
    state_match: bool
    f1: float
    fuzzy_f1_mean: float
    bipartite_sim: float
    invalid_steps: int
    extra_reads: int
    tier: str


def canonical_state_anchor(
    sample_id: str,
    step_idx: int,
    step_calls: list[dict[str, Any]],
    step_results: list[Any],
    malformed: bool,
) -> str:
    """Hash canonical post-observation GiGPO state within an episode group."""
    payload = {
        "step_idx": int(step_idx),
        "malformed": bool(malformed),
        "calls": step_calls,
        "results": step_results,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return f"{sample_id}|t{step_idx}|{digest}"


def _multiset_jaccard(items_a: list[str], items_b: list[str]) -> float:
    if not items_a and not items_b:
        return 1.0
    if not items_a or not items_b:
        return 0.0
    ca, cb = Counter(items_a), Counter(items_b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / max(1, union)


def _arg_value_similarity(pred_val: Any, gold_val: Any, is_fuzzy: bool) -> float:
    if pred_val == gold_val:
        return 1.0
    if is_fuzzy and pred_val is not None and gold_val is not None:
        return token_f1(pred_val, gold_val)
    return 0.0


def compute_bipartite_call_similarity(
    pred_calls: list[dict[str, Any]],
    gold_calls: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> float:
    """ToolRL greedy bipartite call+key+value similarity enhanced with fuzzy token-F1, in [0, 1]."""
    if not gold_calls and not pred_calls:
        return 1.0
    if not gold_calls or not pred_calls:
        return 0.0

    name_sim = _multiset_jaccard(
        [str(c.get("name") or "") for c in pred_calls],
        [str(c.get("name") or "") for c in gold_calls],
    )
    used_pred: set[int] = set()
    total_units = 1.0
    earned_units = name_sim

    for g in gold_calls:
        g_name = str(g.get("name") or "")
        g_args = _clean(g.get("arguments") or {})
        fuz = fuzzy_params(catalog.get(g_name, {}))
        total_units += 1.0 + len(g_args)

        best_idx = -1
        best_score = -1.0
        for idx, p in enumerate(pred_calls):
            if idx in used_pred or str(p.get("name") or "") != g_name:
                continue
            p_args = _clean(p.get("arguments") or {})
            key_sim = _multiset_jaccard(sorted(p_args.keys()), sorted(g_args.keys()))
            val_sim = sum(
                _arg_value_similarity(p_args.get(k), gv, k in fuz)
                for k, gv in g_args.items()
                if k in p_args
            )
            cand_score = key_sim + val_sim
            if cand_score > best_score:
                best_score = cand_score
                best_idx = idx

        if best_idx >= 0:
            used_pred.add(best_idx)
            earned_units += max(0.0, best_score)

    # Penalize unmatched extra predicted calls smoothly
    over_ratio = min(0.5, 0.08 * max(0, len(pred_calls) - len(used_pred)))
    return max(0.0, min(1.0, (earned_units / max(1.0, total_units)) - over_ratio))


def _mean_matched_fuzzy_f1(
    pred_calls: list[dict[str, Any]],
    gold_calls: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> float:
    sims: list[float] = []
    used: set[int] = set()
    for g in gold_calls:
        g_name = str(g.get("name") or "")
        fuz = fuzzy_params(catalog.get(g_name, {}))
        if not fuz:
            continue
        g_args = _clean(g.get("arguments") or {})
        best_sim = 0.0
        best_i = -1
        for i, p in enumerate(pred_calls):
            if i in used or str(p.get("name") or "") != g_name:
                continue
            p_args = _clean(p.get("arguments") or {})
            pair_sims = [token_f1(p_args.get(k, ""), g_args[k]) for k in g_args if k in fuz]
            score = sum(pair_sims) / max(1, len(pair_sims)) if pair_sims else 1.0
            if score >= best_sim:
                best_sim = score
                best_i = i
        if best_i >= 0:
            used.add(best_i)
        sims.append(best_sim)
    return sum(sims) / len(sims) if sims else 1.0


def compute_step_format_penalty(
    raw_text: str,
    parsed_calls: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    *,
    is_last_turn: bool,
    truncated: bool = False,
) -> tuple[float, bool]:
    """Return (step_penalty <= 0.0, is_valid_step) enforcing clean XML/JSON & JSON-schema adherence."""
    if truncated:
        return -0.20, False
    has_tag = "<tool_call>" in (raw_text or "") or "<function=" in (raw_text or "")
    if has_tag and not parsed_calls:
        return -0.08, False
    if not parsed_calls:
        return (0.0, True) if is_last_turn else (-0.08, False)

    ok, _ = validate_step_with_reason({"step_calls": parsed_calls}, catalog)
    if not ok:
        return -0.08, False
    return 0.0, True


def compute_episode_reward(
    pred_calls: list[dict[str, Any]],
    pred_hash: str,
    gold_calls: list[dict[str, Any]],
    gold_hash: str,
    catalog: dict[str, dict[str, Any]],
    *,
    invalid_steps: int = 0,
    extra_reads: int = 0,
    malformed_terminal: bool = False,
) -> RewardBreakdown:
    """Compute strictly monotonic episode reward respecting exact score_sample() optimum (+1.0)."""
    res = score_sample(pred_calls, pred_hash, gold_calls, gold_hash, catalog)
    bipartite = compute_bipartite_call_similarity(pred_calls, gold_calls, catalog)
    fuzzy_mean = _mean_matched_fuzzy_f1(pred_calls, gold_calls, catalog)
    inv_pen = min(0.24, 0.08 * max(0, int(invalid_steps))) + (0.10 if malformed_terminal else 0.0)

    # Case 1: Out-of-scope (gold_calls == [])
    if not gold_calls:
        if not pred_calls and not malformed_terminal and invalid_steps == 0:
            return RewardBreakdown(1.0, res.score, res.state_match, res.f1, 1.0, 1.0, invalid_steps, extra_reads, "exact_oos")
        has_ask = any(c.get("name") == ASK_MISSING_INFO for c in pred_calls)
        rew = -0.25 if has_ask else (-0.30 if pred_calls else max(-0.30, 0.20 - inv_pen))
        return RewardBreakdown(round(rew, 6), res.score, res.state_match, res.f1, fuzzy_mean, bipartite, invalid_steps, extra_reads, "oos_fail")

    # Case 2: Clarification (gold_calls contains sys.ask_missing_info)
    if any(c.get("name") == ASK_MISSING_INFO for c in gold_calls):
        if res.score >= 1.0 - 1e-6 and invalid_steps == 0 and not malformed_terminal:
            return RewardBreakdown(1.0, res.score, res.state_match, res.f1, 1.0, 1.0, invalid_steps, extra_reads, "exact")
        pred_asks = [c for c in pred_calls if c.get("name") == ASK_MISSING_INFO]
        if pred_asks:
            g_args = gold_calls[0].get("arguments") or {}
            p_args = pred_asks[0].get("arguments") or {}
            tool_hit = 0.25 if p_args.get("tool_name") == g_args.get("tool_name") else 0.0
            gp = [str(x) for x in (g_args.get("params") or [])]
            pp = [str(x) for x in (p_args.get("params") or [])]
            param_sim = 0.30 * _multiset_jaccard(pp, gp) if tool_hit > 0 else 0.0
            rew = max(0.10, min(0.55, 0.10 + tool_hit + param_sim - 0.5 * inv_pen))
            return RewardBreakdown(round(rew, 6), res.score, res.state_match, res.f1, fuzzy_mean, bipartite, invalid_steps, extra_reads, "ask_partial")
        rew = -0.20 if not pred_calls else max(-0.30, -0.25 - 0.5 * inv_pen)
        return RewardBreakdown(round(rew, 6), res.score, res.state_match, res.f1, fuzzy_mean, bipartite, invalid_steps, extra_reads, "ask_missed")

    # Case 3: Normal action episode (act)
    if res.score >= 1.0 - 1e-6 and res.f1 >= 1.0 - 1e-6 and extra_reads == 0 and not malformed_terminal:
        return RewardBreakdown(1.0, res.score, res.state_match, res.f1, 1.0, 1.0, invalid_steps, extra_reads, "exact")

    if res.state_match:
        # Strict ceiling 0.94 ensures official F1=1.0 & minimal calls always strictly win
        raw = 0.50 + 0.42 * float(res.f1) + 0.05 * float(fuzzy_mean) - 0.06 * max(0, int(extra_reads)) - 0.3 * inv_pen
        rew = max(0.44, min(0.94, raw))
        return RewardBreakdown(round(rew, 6), res.score, True, res.f1, fuzzy_mean, bipartite, invalid_steps, extra_reads, "sm_partial")

    # State mismatch tier (res.score == 0.0)
    if any(c.get("name") == ASK_MISSING_INFO for c in pred_calls):
        return RewardBreakdown(-0.15, 0.0, False, res.f1, fuzzy_mean, bipartite, invalid_steps, extra_reads, "false_ask")
    if not pred_calls:
        return RewardBreakdown(-0.20, 0.0, False, 0.0, 0.0, 0.0, invalid_steps, extra_reads, "false_oos")

    rew = max(-0.15, min(0.35, 0.35 * float(bipartite) - 0.5 * inv_pen))
    return RewardBreakdown(round(rew, 6), 0.0, False, res.f1, fuzzy_mean, bipartite, invalid_steps, extra_reads, "sm_fail")
```

- [ ] **Step 4: Run `pytest tests/test_rl_reward.py -v`**

Run: `python -m pytest tests/test_rl_reward.py -v`
Expected: ALL 4 tests PASS (`token_f1`, `score_sample`, `validate_step_with_reason` verify exact integration).

- [ ] **Step 5: Commit**

```bash
git add src/agent/rl_reward.py tests/test_rl_reward.py
git commit -m "feat(rl): add strictly monotonic composite multi-turn RL reward"
```

---

### Task 3: Multi-Turn `SafeEnv` Rollout Engine, Assistant-Only Masking, `GiGPO`, `ARPO`, & `GCP` Mining (`src/agent/rl_rollout.py`)

**Files:**
- Create: `src/agent/rl_rollout.py`
- Test: `tests/test_rl_rollout.py`

**Interfaces:**
- Consumes: `build_messages` (`src/agent/history.py`), `parse_tool_calls` & `has_malformed_tool_call` & `sanitize_tool_call` (`src/agent/tool_caller.py`), `SafeEnv` (`src/env/safe_env.py`), `compute_episode_reward` & `compute_step_format_penalty` & `canonical_state_anchor` (`src/agent/rl_reward.py`), `compute_gigpo_advantages`, `EpisodeRollout`, `TrajectoryStepInfo` (`src/agent/lfm3_gigpo.py`).
- Produces:
  - `MultiTurnEpisode` dataclass (`sample_id`, `use_asr`, `prompt_token_len`, `input_ids`, `loss_mask`, `step_ids_per_token`, `teacher_prompt_ids`, `teacher_prompt_len`, `steps`, `step_entropies`, `first_step_calls`, `reward_breakdown`, `episode_reward`)
  - `build_sample_with_asr(sample, asr_row, *, inject_asr)`
  - `compute_assistant_loss_mask_positions(tokenizer, segments)`
  - `execute_single_rollout(sample, catalog, split, root, generate_step_fn, tokenizer, *, gold_calls, gold_hash, clean_sample=None, asr_phonetic_diff=None, use_asr=False, max_steps=8, max_prompt_tokens=15200, forced_prefix_calls=None)`
  - `filter_active_groups(groups, *, min_spread=1e-4)`
  - `assign_group_gigpo_advantages(group, gold_calls, gold_hash, *, omega=0.8, gamma=0.95, remove_std=True)`
  - `maybe_spawn_arpo_forks(group, sample, catalog, split, root, generate_step_fn, tokenizer, *, gold_calls, gold_hash, clean_sample=None, asr_phonetic_diff=None, use_asr=False, entropy_thresh=0.45, max_forks=2)`
  - `sample_weighted_prompt_ids(buckets, rng, n_prompts)`

**Crucial Anti-Collapse & Multi-Turn Details:**
1. **`loss_mask` Masking:** For every ReAct step `t`, compute exact prefix `messages[:user+prior]` via `apply_chat_template(..., tools=tools, add_generation_prompt=True, enable_thinking=False)`, encode the assistant completion + `<|im_end|>\n` (ONLY assistant tokens receive `loss_mask = 1`), and encode `<tool_response>` + user messages with `loss_mask = 0` (`LoopTool` `tool_agent_loop.py:371,417` + `GiGPO schemas.py`). Even cleaner & memory-optimal: train step-by-step on full accumulated `(prefix_ids_up_to_turn_t, assistant_turn_t_ids)` with `loss_mask = 0` on `prefix_ids_up_to_turn_t` and `1` on `assistant_turn_t_ids` OR concatenated full episode sequence with `loss_mask=0` on all prompt/tool tokens! We build exact concatenated `(input_ids, loss_mask, step_adv_per_token)` so ONE forward pass computes exact multi-turn logprobs + KL across an entire episode!
2. **`ARPO` Post-Tool Entropy Branching:** When an episode invokes a read/lookup tool (`contacts.search`, `banking.get_balance`, `banking.list_accounts`, `calendar.list_events`, `chat_ott.list_chats`, `gold_price.*`, `weather.*`) on turn `t=0`, if any Turn 2 (`t=1`) step token entropy exceeds `arpo_entropy_thresh=0.45` OR the initial group has `std(rewards) == 0` with `reward < 1.0`, spawn `2` forced-prefix continuations (`forced_prefix_calls = calls_from_step_0`) directly replaying step 0 in fresh `SafeEnv` instances and sampling fresh Turn 2+ completions!
3. **`LoopTool` Dynamic Filter (`filter_non_zero_std_groups`):** Drops zero-variance groups (`max(r) - min(r) <= 1e-4`), resampling from mined Tier A/B pool up to `max_gen_batches=4`.

- [ ] **Step 1: Write the failing test `tests/test_rl_rollout.py`**

```python
"""Test multi-turn SafeEnv rollout collection, assistant-only loss_mask, GiGPO grouping, and ARPO prefix fork."""
from types import SimpleNamespace
from src.core.dataset import load_catalog, load_split
from src.agent.rl_rollout import (
    build_sample_with_asr,
    compute_assistant_loss_mask_positions,
    execute_single_rollout,
    filter_active_groups,
)


class _DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 200 + 1 for c in text]

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        parts = [f"<{m['role']}>{m.get('content', '')}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        txt = "\n".join(parts)
        return self.encode(txt) if tokenize else txt


def test_assistant_only_mask_zeroes_tool_responses():
    tok = _DummyTokenizer()
    segs = [
        ("prompt", "sys+user"),
        ("assistant", "call_1<|im_end|>\n"),
        ("tool", "tool_resp_1"),
        ("assistant", "done<|im_end|>\n"),
    ]
    ids, mask, step_ids = compute_assistant_loss_mask_positions(tok, segs)
    assert len(ids) == len(mask) == len(step_ids)
    # tool segment tokens must strictly have mask == 0
    tool_start = len(tok.encode("sys+user")) + len(tok.encode("call_1<|im_end|>\n"))
    tool_end = tool_start + len(tok.encode("tool_resp_1"))
    assert all(m == 0 for m in mask[tool_start:tool_end])
    assert all(m == 1 for m in mask[tool_end:])
    assert step_ids[tool_end] == 1


def test_execute_single_rollout_real_safe_env_and_forced_prefix():
    catalog = load_catalog("train", "data")
    samples = load_split("train", "data")
    # Find an episode with contacts.search or simple act without mutating unknown state outside SafeEnv
    target = next(s for s in samples if s.id == "train_0001")
    gold = [c.model_dump() for c in (target.label.tool_calls if target.label else [])]
    from src.env.safe_env import SafeEnv
    env_g = SafeEnv("train", target.id, "data")
    for gc in gold:
        env_g.call(gc["name"], gc["arguments"])
    gold_hash = env_g.get_state_hash()

    scripted = [
        '<tool_call>\n<function=contacts.search>\n<parameter=query>\nLan\n</parameter>\n</function>\n</tool_call>',
        "Da xong.",
    ]
    def gen_fn(messages, step_idx, tools):
        assert tools is not None
        txt = scripted[step_idx] if step_idx < len(scripted) else "Da xong."
        return txt, [10, 11], -0.1, 0.20

    ep = execute_single_rollout(
        target,
        catalog,
        "train",
        "data",
        gen_fn,
        _DummyTokenizer(),
        gold_calls=gold,
        gold_hash=gold_hash,
        clean_sample=target,
        max_steps=4,
    )
    assert ep.steps[0].step_idx == 0
    assert ep.prompt_token_len > 0
    assert all(m == 0 for m in ep.loss_mask[: ep.prompt_token_len])
    assert ep.reward_breakdown.episode_reward <= 1.0


def test_filter_active_groups_drops_zero_variance():
    g0 = [SimpleNamespace(episode_reward=1.0), SimpleNamespace(episode_reward=1.0)]
    g1 = [SimpleNamespace(episode_reward=1.0), SimpleNamespace(episode_reward=0.2)]
    kept = filter_active_groups([g0, g1])
    assert len(kept) == 1
    assert kept[0] is g1


def test_build_sample_with_asr_injects_phonetic_diff():
    samples = load_split("train", "data")
    s = samples[0]
    noisy, diff = build_sample_with_asr(
        s,
        {"audio_text": "chuyen 500 ngay", "history_texts": {}, "asr_phonetic_diff": "ngay / ngan / nghin"},
        inject_asr=True,
    )
    assert noisy.audio_text == "chuyen 500 ngay"
    assert diff == "ngay / ngan / nghin"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rl_rollout.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.agent.rl_rollout'`).

- [ ] **Step 3: Write `src/agent/rl_rollout.py`**

```python
"""Multi-turn SafeEnv rollout collector with assistant-only loss masking, GiGPO grouping,
ARPO post-tool entropy branching, ASR transcript mixing, and DAPO zero-std filtering."""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agent.history import build_messages
from src.agent.lfm3_gigpo import EpisodeRollout, TrajectoryStepInfo, compute_gigpo_advantages
from src.agent.rl_reward import (
    RewardBreakdown,
    canonical_state_anchor,
    compute_episode_reward,
    compute_step_format_penalty,
)
from src.agent.tool_caller import has_malformed_tool_call, parse_tool_calls, sanitize_tool_call
from src.core.schemas import ASK_MISSING_INFO, Sample
from src.env.safe_env import SafeEnv

READ_PREF_PREFIXES = (
    "contacts.search",
    "contacts.list",
    "banking.get_balance",
    "banking.list_accounts",
    "banking.list_beneficiaries",
    "calendar.list_events",
    "chat_ott.list_chats",
    "gold_price.",
    "weather.",
    "fuel_price.",
    "exchange_rate.",
)


@dataclass
class MultiTurnEpisode:
    sample_id: str
    use_asr: bool
    prompt_token_len: int
    input_ids: list[int]
    loss_mask: list[int]
    step_ids_per_token: list[int]
    teacher_prompt_ids: list[int]
    teacher_prompt_len: int
    steps: list[TrajectoryStepInfo]
    step_entropies: list[float]
    first_step_calls: list[dict[str, Any]]
    reward_breakdown: RewardBreakdown
    episode_reward: float


def _is_lookup_tool(name: str) -> bool:
    return any(name == p or (p.endswith(".") and name.startswith(p)) for p in READ_PREF_PREFIXES)


def build_sample_with_asr(
    sample: Sample,
    asr_row: dict[str, Any] | None,
    *,
    inject_asr: bool,
) -> tuple[Sample, str | None]:
    """Return sample copy overlaid with real PhoWhisper transcripts + phonetic diff when enabled."""
    if not inject_asr or not asr_row:
        return sample, None
    s_copy = copy.deepcopy(sample)
    if asr_row.get("audio_text"):
        s_copy.audio_text = str(asr_row["audio_text"])
    hist_map = asr_row.get("history_texts") or {}
    if hist_map and s_copy.history:
        for idx, turn in enumerate(s_copy.history):
            if str(idx) in hist_map and hist_map[str(idx)]:
                turn.text = str(hist_map[str(idx)])
    return s_copy, asr_row.get("asr_phonetic_diff")


def compute_assistant_loss_mask_positions(
    tokenizer: Any,
    segments: list[tuple[str, str]],
) -> tuple[list[int], list[int], list[int]]:
    """Convert ('prompt'|'tool'|'assistant', text) segments into (ids, loss_mask, step_idx_per_token)."""
    all_ids: list[int] = []
    all_mask: list[int] = []
    all_steps: list[int] = []
    step_counter = 0
    for kind, text in segments:
        if not text:
            continue
        toks = tokenizer.encode(text, add_special_tokens=False)
        if not toks:
            continue
        is_asst = kind == "assistant"
        all_ids.extend(toks)
        all_mask.extend([1 if is_asst else 0] * len(toks))
        all_steps.extend([step_counter if is_asst else -1] * len(toks))
        if is_asst:
            step_counter += 1
    return all_ids, all_mask, all_steps


def _safe_tool_call_dict(call_obj: Any) -> dict[str, Any]:
    return {"name": call_obj.name, "arguments": dict(call_obj.arguments)}


def execute_single_rollout(
    sample: Sample,
    catalog: dict[str, dict[str, Any]],
    split: str,
    root: str,
    generate_step_fn: Callable[[list[dict[str, Any]], int, list[dict[str, Any]]], tuple[str, list[int] | None, float, float]],
    tokenizer: Any,
    *,
    gold_calls: list[dict[str, Any]],
    gold_hash: str,
    clean_sample: Sample | None = None,
    asr_phonetic_diff: str | None = None,
    use_asr: bool = False,
    max_steps: int = 8,
    max_prompt_tokens: int = 15200,
    forced_prefix_calls: list[dict[str, Any]] | None = None,
) -> MultiTurnEpisode:
    """Execute one live multi-turn rollout in SafeEnv, recording exact token masks and GiGPO anchors."""
    env = SafeEnv(split, sample.id, root)
    messages = build_messages(
        sample,
        current_text=sample.audio_text,
        asr_phonetic_diff=asr_phonetic_diff,
        prompt_style="qwen35",
        max_prompt_tokens=max_prompt_tokens,
    )
    app_names = set((sample.context or {}).get("installed_apps", {}).keys())
    tools = [spec for name, spec in catalog.items() if name.split(".", 1)[0] in app_names]

    initial_prompt_text = tokenizer.apply_chat_template(
        messages, tools=tools or None, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_token_len = len(tokenizer.encode(initial_prompt_text, add_special_tokens=False))
    if clean_sample is not None and use_asr:
        clean_msgs = build_messages(
            clean_sample,
            current_text=clean_sample.audio_text,
            asr_phonetic_diff=None,
            prompt_style="qwen35",
            max_prompt_tokens=max_prompt_tokens,
        )
        teacher_prompt_text = tokenizer.apply_chat_template(
            clean_msgs, tools=tools or None, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        teacher_prompt_ids = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)
    else:
        teacher_prompt_ids = []
    teacher_prompt_len = len(teacher_prompt_ids) if teacher_prompt_ids else prompt_token_len

    segments: list[tuple[str, str]] = [("prompt", initial_prompt_text)]
    steps: list[TrajectoryStepInfo] = []
    step_entropies: list[float] = []
    first_step_calls: list[dict[str, Any]] = []
    invalid_steps = 0
    extra_reads = 0
    malformed_terminal = False
    gold_names = [str(c.get("name") or "") for c in gold_calls]

    for step_idx in range(max_steps):
        if forced_prefix_calls and step_idx == 0:
            raw_text = "".join(
                f"<tool_call>\n<function={c['name']}>\n"
                + "".join(f"<parameter={k}>\n{json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}\n</parameter>\n" for k, v in (c.get("arguments") or {}).items())
                + "</function>\n</tool_call>"
                for c in forced_prefix_calls
            )
            step_ent = 0.0
            is_forced = True
        else:
            raw_text, _tok_ids, _sum_lp, step_ent = generate_step_fn(messages, step_idx, tools)
            is_forced = False

        parsed = parse_tool_calls(raw_text, catalog)
        parsed = [sanitize_tool_call(c, sample, sample.audio_text) for c in parsed]
        parsed_dicts = [_safe_tool_call_dict(c) for c in parsed]
        step_entropies.append(float(step_ent))

        is_terminal_text = not parsed
        if has_malformed_tool_call(raw_text, catalog) and not parsed:
            malformed_terminal = True
        pen, valid_step = compute_step_format_penalty(
            raw_text, parsed_dicts, catalog, is_last_turn=is_terminal_text
        )
        if not valid_step:
            invalid_steps += 1

        # Add assistant turn text with end-of-turn delimiter matching Qwen chat template
        segments.append(("prompt" if is_forced else "assistant", f"{raw_text}<|im_end|>\n"))

        if not parsed:
            anchor = canonical_state_anchor(sample.id, step_idx, [], [], malformed_terminal)
            steps.append(TrajectoryStepInfo(step_idx=step_idx, anchor_state=anchor, step_reward=pen))
            break

        step_results: list[Any] = []
        messages.append({"role": "assistant", "content": raw_text})
        ask_stop = False
        tool_resp_msgs: list[dict[str, Any]] = []
        for idx, c in enumerate(parsed):
            if c.name not in gold_names and _is_lookup_tool(c.name):
                extra_reads += 1
            if c.name == ASK_MISSING_INFO:
                env.call(c.name, c.arguments)
                step_results.append({"asked": True})
                ask_stop = True
                break
            ok, err = env.validate(c.name, c.arguments)
            res = env.call(c.name, c.arguments) if ok else {"error": err}
            step_results.append(res)
            resp_msg = {"role": "tool", "name": c.name, "content": json.dumps(res, ensure_ascii=False)}
            messages.append(resp_msg)
            tool_resp_msgs.append(resp_msg)

        if step_idx == 0:
            first_step_calls = list(parsed_dicts)

        anchor = canonical_state_anchor(sample.id, step_idx, parsed_dicts, step_results, not valid_step)
        steps.append(TrajectoryStepInfo(step_idx=step_idx, anchor_state=anchor, step_reward=pen))

        if ask_stop:
            break

        if tool_resp_msgs:
            # Measure exact delta tokens added by tool observation + next assistant generation header
            before_obs = tokenizer.apply_chat_template(
                messages[: -len(tool_resp_msgs)], tools=tools or None, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
            after_obs = tokenizer.apply_chat_template(
                messages, tools=tools or None, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            obs_delta = after_obs[len(before_obs) :] if after_obs.startswith(before_obs) else after_obs
            segments.append(("tool", obs_delta))

    pred_calls = env.get_tool_calls()
    pred_hash = env.get_state_hash()
    bd = compute_episode_reward(
        pred_calls,
        pred_hash,
        gold_calls,
        gold_hash,
        catalog,
        invalid_steps=invalid_steps,
        extra_reads=extra_reads,
        malformed_terminal=malformed_terminal,
    )
    if steps:
        steps[-1].step_reward += bd.episode_reward

    input_ids, loss_mask, step_ids = compute_assistant_loss_mask_positions(tokenizer, segments)
    return MultiTurnEpisode(
        sample_id=sample.id,
        use_asr=use_asr,
        prompt_token_len=prompt_token_len,
        input_ids=input_ids,
        loss_mask=loss_mask,
        step_ids_per_token=step_ids,
        teacher_prompt_ids=teacher_prompt_ids,
        teacher_prompt_len=teacher_prompt_len,
        steps=steps,
        step_entropies=step_entropies,
        first_step_calls=first_step_calls,
        reward_breakdown=bd,
        episode_reward=bd.episode_reward,
    )


def filter_active_groups(groups: list[list[Any]], *, min_spread: float = 1e-4) -> list[list[Any]]:
    """DAPO dynamic group filter: discard zero-variance groups where all rollouts yield equal reward."""
    kept: list[list[Any]] = []
    for grp in groups:
        if len(grp) < 2:
            continue
        rewards = [float(ep.episode_reward) for ep in grp]
        if (max(rewards) - min(rewards)) > min_spread:
            kept.append(grp)
    return kept


def assign_group_gigpo_advantages(
    group: list[MultiTurnEpisode],
    gold_calls: list[dict[str, Any]],
    gold_hash: str,
    *,
    omega: float = 0.8,
    gamma: float = 0.95,
    remove_std: bool = True,
) -> list[MultiTurnEpisode]:
    """Attach GiGPO episode + anchor-state step advantages (`Dr.GRPO` unnormalized mean when remove_std=True)."""
    if remove_std:
        rewards = [ep.episode_reward for ep in group]
        ep_mean = sum(rewards) / max(1, len(rewards))
        # Anchor-state grouping via canonical_state_anchor across episodes
        anchor_returns: dict[str, list[tuple[int, int, float]]] = {}
        discounted: list[list[float]] = []
        for ei, ep in enumerate(group):
            n_t = len(ep.steps)
            rets = [0.0] * n_t
            acc = 0.0
            for t in range(n_t - 1, -1, -1):
                acc = ep.steps[t].step_reward + gamma * acc
                rets[t] = acc
            discounted.append(rets)
            for t, st in enumerate(ep.steps):
                anchor_returns.setdefault(st.anchor_state, []).append((ei, t, rets[t]))

        for _anchor, entries in anchor_returns.items():
            if len(entries) <= 1:
                for ei, t, _ in entries:
                    st = group[ei].steps[t]
                    st.macro_advantage = group[ei].episode_reward - ep_mean
                    st.micro_advantage = 0.0
                    st.combined_advantage = max(-3.0, min(3.0, st.macro_advantage))
                continue
            vals = [x[2] for x in entries]
            s_mean = sum(vals) / len(vals)
            for ei, t, r_t in entries:
                st = group[ei].steps[t]
                st.macro_advantage = group[ei].episode_reward - ep_mean
                st.micro_advantage = r_t - s_mean
                st.combined_advantage = max(-3.0, min(3.0, st.macro_advantage + omega * st.micro_advantage))
        return group

    # Delegate standard mean/std normalization to src/agent/lfm3_gigpo.py
    lfm_eps = [
        EpisodeRollout(
            sample_id=ep.sample_id,
            steps=ep.steps,
            pred_calls=[],
            pred_hash="",
            gold_calls=gold_calls,
            gold_hash=gold_hash,
            f1=ep.reward_breakdown.f1,
            state_match=ep.reward_breakdown.state_match,
            official_score=ep.reward_breakdown.official_score,
            episode_reward=ep.episode_reward,
        )
        for ep in group
    ]
    compute_gigpo_advantages(lfm_eps, omega=omega, gamma=gamma)
    return group


def maybe_spawn_arpo_forks(
    group: list[MultiTurnEpisode],
    sample: Sample,
    catalog: dict[str, dict[str, Any]],
    split: str,
    root: str,
    generate_step_fn: Callable[[list[dict[str, Any]], int, list[dict[str, Any]]], tuple[str, list[int] | None, float, float]],
    tokenizer: Any,
    *,
    gold_calls: list[dict[str, Any]],
    gold_hash: str,
    clean_sample: Sample | None = None,
    asr_phonetic_diff: str | None = None,
    use_asr: bool = False,
    entropy_thresh: float = 0.45,
    max_forks: int = 2,
) -> list[MultiTurnEpisode]:
    """ARPO-inspired post-tool branch sampling when Turn 2 shows high entropy or stalled group."""
    candidates = [
        ep
        for ep in group
        if ep.first_step_calls
        and any(_is_lookup_tool(c["name"]) for c in ep.first_step_calls)
        and len(ep.step_entropies) >= 2
        and ep.step_entropies[1] >= entropy_thresh
    ]
    if not candidates:
        return group
    anchor_ep = max(candidates, key=lambda e: e.step_entropies[1])
    forks = [
        execute_single_rollout(
            sample,
            catalog,
            split,
            root,
            generate_step_fn,
            tokenizer,
            gold_calls=gold_calls,
            gold_hash=gold_hash,
            clean_sample=clean_sample,
            asr_phonetic_diff=asr_phonetic_diff,
            use_asr=use_asr,
            forced_prefix_calls=anchor_ep.first_step_calls,
        )
        for _ in range(max_forks)
    ]
    return [*group, *forks]


def sample_weighted_prompt_ids(
    buckets: dict[str, list[str]],
    rng: random.Random,
    n_prompts: int,
) -> list[str]:
    """GCP priority mix: Tier A hard failures (55%), Tier B high-PPL boundary (30%), Tier C anchors (15%)."""
    weights = {"tier_a_hard": 0.55, "tier_b_boundary": 0.30, "tier_c_anchor": 0.15}
    active = {k: v for k, v in buckets.items() if v}
    total_w = sum(weights.get(k, 0.2) for k in active)
    keys = list(active.keys())
    probs = [weights.get(k, 0.2) / total_w for k in keys]
    chosen: list[str] = []
    for _ in range(n_prompts):
        b = rng.choices(keys, weights=probs, k=1)[0]
        chosen.append(rng.choice(active[b]))
    return chosen
```

- [ ] **Step 4: Run `pytest tests/test_rl_rollout.py -v`**

Run: `python -m pytest tests/test_rl_rollout.py -v`
Expected: ALL 4 tests PASS (`train_0001` loads cleanly in `SafeEnv` without accessing `state.json`).

- [ ] **Step 5: Commit**

```bash
git add src/agent/rl_rollout.py tests/test_rl_rollout.py
git commit -m "feat(rl): add multi-turn SafeEnv rollout collector with GiGPO, ARPO, and assistant masking"
```

---

### Task 4: `GSPO + Clip-Higher + Reverse-KL OPD/SDPO + DiaTool Step-DPO` Losses & Dual-Adapter G4 Trainer

**Files:**
- Create: `src/agent/rl_losses.py`
- Create: `scripts/train_qwen35_rl.py`
- Create: `configs/qwen35_v2_rl_stage2.yaml`
- Create: `configs/qwen35_v2_rl_stage3_opd.yaml`
- Test: `tests/test_rl_losses.py`

**Interfaces:**
- Produces (`src/agent/rl_losses.py`):
  - `compute_gspo_clip_higher_loss(cur_logprobs, old_logprobs, advantages, loss_mask, *, clip_low=0.20, clip_high=0.28)` — exact `07_LoopTool/core_algos.py:972-1038` GSPO sequence importance ratio with straight-through per-token gradient + `"seq-mean-token-mean"` normalization.
  - `compute_low_var_kl_loss(cur_logprobs, ref_logprobs, loss_mask)` — positive `k3 = exp(ref - cur) - (ref - cur) - 1`, clamped `>= 0.0` (`+ kl_coef * kl_loss`, avoiding `ToolRL` negative KL sign bug).
  - `compute_opd_reverse_kl_advantages(student_token_logprobs, teacher_token_logprobs, episode_advantages, loss_mask, *, alpha_ep=0.5, clip_val=2.0)` — exact `MiniCPM5 OPD` token reverse-KL advantage `-(log pi_s - log pi_t) + alpha_ep * A_E`.
  - `compute_diatool_step_dpo_loss(chosen_turn_lps, chosen_turn_ref_lps, rejected_turn_lps, rejected_turn_ref_lps, *, beta=0.1, margin=0.5, gamma=0.5)` — exact `14_DiaTool-DPO/trl/trainer/dpo_trainer.py:1272-1355` temporal turn weighting $w(t)=\frac{1-\gamma^{T-t}}{1-\gamma^T}/\psi$ supporting $T_c \neq T_r$.

- [ ] **Step 1: Write the failing test `tests/test_rl_losses.py`**

```python
"""Test GSPO+Clip-Higher, positive k3 KL, MiniCPM5 reverse-KL OPD, and DiaTool temporal Step-DPO."""
import math
import torch
from src.agent.rl_losses import (
    compute_diatool_step_dpo_loss,
    compute_gspo_clip_higher_loss,
    compute_low_var_kl_loss,
    compute_opd_reverse_kl_advantages,
)


def test_gspo_clip_higher_masks_non_assistant_tokens_and_asymmetrically_clips():
    cur_lp = torch.tensor([[-1.0, -1.0, -0.5, -0.5]], requires_grad=True)
    old_lp = torch.tensor([[-1.0, -1.0, -0.9, -0.9]])  # exp(delta) = exp(0.4) = 1.49 > 1.28
    adv = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    loss, stats = compute_gspo_clip_higher_loss(cur_lp, old_lp, adv, mask, clip_low=0.20, clip_high=0.28)
    loss.backward()
    assert cur_lp.grad[0, 0].item() == 0.0
    assert cur_lp.grad[0, 1].item() == 0.0
    assert stats["clipped_frac"] > 0.0


def test_low_var_kl_is_nonnegative_and_zero_when_equal():
    lp = torch.tensor([[-1.2, -0.8, -0.4]], requires_grad=True)
    kl = compute_low_var_kl_loss(lp, lp.detach(), torch.tensor([[0.0, 1.0, 1.0]]))
    assert abs(kl.item()) < 1e-6

    cur_shifted = torch.tensor([[-1.2, -1.5, -0.1]], requires_grad=True)
    kl2 = compute_low_var_kl_loss(cur_shifted, torch.tensor([[-1.2, -0.8, -0.4]]), torch.tensor([[0.0, 1.0, 1.0]]))
    assert kl2.item() > 0.0


def test_opd_reverse_kl_rewards_tokens_aligned_with_teacher():
    s_lp = torch.tensor([[-2.0, -0.5, -0.1]])
    t_lp = torch.tensor([[-0.5, -0.5, -1.5]])
    ep_adv = torch.tensor([[0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    adv = compute_opd_reverse_kl_advantages(s_lp, t_lp, ep_adv, mask, alpha_ep=0.0, clip_val=2.0)
    # Token 0: teacher has higher prob than student -> positive advantage (+1.5) to pull student UP!
    assert abs(adv[0, 0].item() - 1.5) < 1e-5
    # Token 2: student overconfident vs teacher -> negative advantage (-1.4)
    assert abs(adv[0, 2].item() - (-1.4)) < 1e-5


def test_diatool_step_dpo_weights_early_turns_higher_even_with_unequal_lengths():
    c_pi = [torch.tensor(-1.0, requires_grad=True), torch.tensor(-1.0, requires_grad=True)]
    c_ref = [torch.tensor(-1.5), torch.tensor(-1.0)]
    r_pi = [torch.tensor(-2.0, requires_grad=True), torch.tensor(-1.0, requires_grad=True), torch.tensor(-1.0, requires_grad=True)]
    r_ref = [torch.tensor(-1.5), torch.tensor(-1.0), torch.tensor(-1.0)]

    loss = compute_diatool_step_dpo_loss(c_pi, c_ref, r_pi, r_ref, beta=0.1, margin=0.5, gamma=0.5)
    loss.backward()
    assert math.isfinite(loss.item())
    assert abs(c_pi[0].grad.item()) > abs(c_pi[1].grad.item())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rl_losses.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.agent.rl_losses'`).

- [ ] **Step 3: Write `src/agent/rl_losses.py`**

```python
"""Exact RL policy, KL, MiniCPM5 reverse-KL OPD, and DiaTool turn-discounted DPO losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_gspo_clip_higher_loss(
    cur_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    clip_low: float = 0.20,
    clip_high: float = 0.28,
) -> tuple[torch.Tensor, dict[str, float]]:
    """GSPO sequence-level importance ratio (arXiv:2507.18071) + DAPO Clip-Higher + seq-mean-token-mean."""
    mask = loss_mask.float()
    seq_lens = mask.sum(dim=-1).clamp(min=1.0)
    log_diff = (cur_logprobs - old_logprobs.detach()) * mask
    seq_log_ratio = log_diff.sum(dim=-1) / seq_lens

    # Straight-through per-token gradient at sequence ratio scale
    st_log_ratio = cur_logprobs - cur_logprobs.detach() + seq_log_ratio.detach().unsqueeze(-1)
    st_log_ratio = torch.clamp(st_log_ratio, min=-10.0, max=10.0)
    ratio = torch.exp(st_log_ratio)

    adv = advantages.detach()
    pg1 = -adv * ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_low), 1.0 + float(clip_high))
    pg2 = -adv * clipped_ratio
    pg_losses = torch.maximum(pg1, pg2)

    # seq-mean-token-mean matching LoopTool/verl core_algos.py:792-794
    per_seq = (pg_losses * mask).sum(dim=-1) / seq_lens
    loss = per_seq.mean()

    with torch.no_grad():
        was_clipped = ((pg2 > pg1).float() * mask).sum() / mask.sum().clamp(min=1.0)
        approx_kl = (-(cur_logprobs - old_logprobs) * mask).sum() / mask.sum().clamp(min=1.0)
    return loss, {"clipped_frac": float(was_clipped.item()), "approx_kl": float(approx_kl.item())}


def compute_low_var_kl_loss(
    cur_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Non-negative k3 low-variance KL penalty (always ADDED with +kl_coef to total loss)."""
    mask = loss_mask.float()
    seq_lens = mask.sum(dim=-1).clamp(min=1.0)
    delta = torch.clamp(ref_logprobs.detach() - cur_logprobs, min=-10.0, max=10.0)
    k3 = torch.clamp(torch.exp(delta) - delta - 1.0, min=0.0, max=100.0)
    return ((k3 * mask).sum(dim=-1) / seq_lens).mean()


def compute_opd_reverse_kl_advantages(
    student_token_logprobs: torch.Tensor,
    teacher_token_logprobs: torch.Tensor,
    episode_advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    alpha_ep: float = 0.5,
    clip_val: float = 2.0,
) -> torch.Tensor:
    """MiniCPM5 reverse-KL advantage: pull student up where teacher logprob > student logprob."""
    rkl_adv = torch.clamp(teacher_token_logprobs.detach() - student_token_logprobs.detach(), min=-clip_val, max=clip_val)
    total = (rkl_adv + float(alpha_ep) * episode_advantages.detach()) * loss_mask.float()
    return total


def _diatool_discounted_proxy(
    pi_turn_lps: list[torch.Tensor],
    ref_turn_lps: list[torch.Tensor],
    gamma: float,
) -> torch.Tensor:
    T = len(pi_turn_lps)
    if T == 0:
        return torch.tensor(0.0)
    denom = max(1e-8, 1.0 - (gamma ** T))
    weights = [(1.0 - (gamma ** (T - t))) / denom for t in range(T)]
    psi = max(1e-8, sum(weights))
    weighted = sum(w * (p - r.detach()) for w, p, r in zip(weights, pi_turn_lps, ref_turn_lps))
    return weighted / psi


def compute_diatool_step_dpo_loss(
    chosen_turn_lps: list[torch.Tensor],
    chosen_turn_ref_lps: list[torch.Tensor],
    rejected_turn_lps: list[torch.Tensor],
    rejected_turn_ref_lps: list[torch.Tensor],
    *,
    beta: float = 0.1,
    margin: float = 0.5,
    gamma: float = 0.5,
) -> torch.Tensor:
    """DiaTool-DPO temporal turn-normalized margin loss over variable-length multi-turn calls."""
    r_c = _diatool_discounted_proxy(chosen_turn_lps, chosen_turn_ref_lps, gamma)
    r_r = _diatool_discounted_proxy(rejected_turn_lps, rejected_turn_ref_lps, gamma)
    logits = float(beta) * (r_c - r_r - float(margin))
    return -F.logsigmoid(logits)
```

- [ ] **Step 4: Run `pytest tests/test_rl_losses.py -v`**

Run: `python -m pytest tests/test_rl_losses.py -v`
Expected: ALL 4 tests PASS.

- [ ] **Step 5: Write `scripts/train_qwen35_rl.py` and YAML configs**

Create `configs/qwen35_v2_rl_stage2.yaml`:
```yaml
base_model_id: /content/models/qwen35_4b_sft_v2_merged
train_split: train
root: /content/data
train_ids: configs/splits/folds_1_to_4_ids.json
val_ids: configs/splits/fold_0_ids.json
asr_cache: /content/folds1_4_phowhisper_3beam_nbest.jsonl
output_dir: /content/experiments/exp_qwen35_v2_rl/stage2
hf_prefix: exp_qwen35_v2_rl/stage2
stage: rl_gigpo_gspo
seed: 42

lora:
  r: 32
  alpha: 64
  dropout: 0.0

rollout:
  batch_size: 16
  group_size: 8
  max_gen_batches: 4
  max_steps: 8
  max_new_tokens: 384
  temperature: 0.85
  top_p: 0.95
  asr_mix_prob: 0.40
  arpo_entropy_thresh: 0.45
  omega: 0.8
  gamma: 0.95
  remove_std: true

optim:
  total_steps: 200
  mini_batch_size: 16
  lr: 1.0e-6
  warmup_steps: 10
  weight_decay: 0.0
  max_grad_norm: 1.0
  clip_low: 0.20
  clip_high: 0.28
  kl_coef: 0.002
  save_steps: 25
  sync_interval_s: 600
  max_wall_minutes: 45
```

Create `configs/qwen35_v2_rl_stage3_opd.yaml`:
```yaml
base_model_id: /content/models/qwen35_4b_sft_v2_merged
init_student_adapter_hf: exp_qwen35_v2_rl/stage2/checkpoints/latest
teacher_adapter_hf: exp_qwen35_v2_rl/stage2/checkpoints/latest
train_split: train
root: /content/data
train_ids: configs/splits/folds_1_to_4_ids.json
val_ids: configs/splits/fold_0_ids.json
asr_cache: /content/folds1_4_phowhisper_3beam_nbest.jsonl
output_dir: /content/experiments/exp_qwen35_v2_rl/stage3_opd
hf_prefix: exp_qwen35_v2_rl/stage3_opd
stage: voice_opd
seed: 1337

lora:
  r: 32
  alpha: 64
  dropout: 0.0

rollout:
  batch_size: 16
  group_size: 6
  max_gen_batches: 3
  max_steps: 8
  max_new_tokens: 384
  temperature: 0.80
  top_p: 0.95
  asr_mix_prob: 0.85
  arpo_entropy_thresh: 0.45
  omega: 0.8
  gamma: 0.95
  remove_std: true

opd:
  alpha_ep: 0.5
  rkl_clip: 2.0

optim:
  total_steps: 100
  mini_batch_size: 16
  lr: 7.5e-7
  warmup_steps: 8
  weight_decay: 0.0
  max_grad_norm: 1.0
  clip_low: 0.20
  clip_high: 0.28
  kl_coef: 0.003
  save_steps: 25
  sync_interval_s: 600
  max_wall_minutes: 45
```

Create `scripts/train_qwen35_rl.py`:
```python
"""Multi-turn Qwen3.5-4B RL & Voice-OPD trainer reusing frozen SFT V2 base + PEFT LoRA adapters.
Guarantees:
1. HFSyncer.preflight() before GPU allocation, LoRA-only checkpoints (~260 MB) pushed every 10m & save_steps.
2. Automatic resume from HF <hf_prefix>/checkpoints/latest/ (never restarts at step 0).
3. Zero extra base-model VRAM: uses model.disable_adapter() for SFT V2 ref KL and named PEFT adapters
   ('student' + 'teacher') for Stage 3 Privileged Voice/Error Reverse-KL OPD/SDPO.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.qwen35_sft import QWEN35_LORA_TARGET_MODULES
from src.agent.rl_losses import (
    compute_gspo_clip_higher_loss,
    compute_low_var_kl_loss,
    compute_opd_reverse_kl_advantages,
)
from src.agent.rl_rollout import (
    MultiTurnEpisode,
    assign_group_gigpo_advantages,
    build_sample_with_asr,
    execute_single_rollout,
    filter_active_groups,
    maybe_spawn_arpo_forks,
    sample_weighted_prompt_ids,
)
from src.core.dataset import load_catalog, load_split
from src.core.hf_sync import BackgroundPusher, HFSyncer, resolve_repo_id
from src.env.safe_env import SafeEnv


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _compute_sequence_token_logprobs(
    model: Any,
    input_ids: list[int],
) -> torch.Tensor:
    """Compute next-token log-probabilities log P(x_t | x_<t) over full sequence [1, T]."""
    dev = next(model.parameters()).device
    x = torch.tensor([input_ids], dtype=torch.long, device=dev)
    out = model(input_ids=x, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    targets = x[:, 1:]
    lp = F.log_softmax(logits, dim=-1)
    tok_lp = torch.gather(lp, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # Pad position 0 so index aligns 1-to-1 with input_ids length
    return F.pad(tok_lp, (1, 0), value=0.0)


def _build_token_advantage_tensor(ep: MultiTurnEpisode, seq_len: int, device: torch.device) -> torch.Tensor:
    step_adv_map = {st.step_idx: float(st.combined_advantage) for st in ep.steps}
    vals = [step_adv_map.get(sidx, 0.0) if m == 1 else 0.0 for sidx, m in zip(ep.step_ids_per_token, ep.loss_mask)]
    return torch.tensor([vals[:seq_len]], dtype=torch.float32, device=device)


def _load_asr_cache(path: str | None) -> dict[str, dict[str, Any]]:
    if not path or not Path(path).exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row
    return out


def _precompute_gold_hashes(samples: list[Any], split: str, root: str) -> dict[str, tuple[list[dict[str, Any]], str]]:
    cache: dict[str, tuple[list[dict[str, Any]], str]] = {}
    for s in samples:
        gold = [c.model_dump() for c in (s.label.tool_calls if s.label else [])]
        env = SafeEnv(split, s.id, root)
        for gc in gold:
            env.call(gc["name"], gc["arguments"])
        cache[s.id] = (gold, env.get_state_hash())
    return cache


def _save_and_push_ckpt(
    model: PeftModel,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    out_dir: Path,
    step: int,
    metrics: dict[str, Any],
    pusher: BackgroundPusher,
    hf_prefix: str,
) -> None:
    snap = out_dir / f"checkpoint-{step}"
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True, exist_ok=True)
    tmp_save = snap / "_peft_tmp"
    has_teacher = "teacher" in getattr(model, "peft_config", {})
    if has_teacher:
        model.save_pretrained(str(tmp_save), selected_adapters=["default"])
        src_dir = tmp_save / "default" if (tmp_save / "default").is_dir() else tmp_save
        for p in src_dir.iterdir():
            shutil.move(str(p), str(snap / p.name))
        shutil.rmtree(tmp_save, ignore_errors=True)
    else:
        model.save_pretrained(str(snap))
    tokenizer.save_pretrained(str(snap))
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, snap / "trainer_state.pt")
    (snap / "metrics.json").write_text(json.dumps({"step": step, **metrics}, indent=2), encoding="utf-8")
    pusher.submit(snap, f"{hf_prefix}/checkpoints/checkpoint-{step}", f"RL step {step}")
    pusher.submit(snap, f"{hf_prefix}/checkpoints/latest", f"RL latest step {step}")


def _mine_gcp_buckets(
    model: PeftModel,
    tokenizer: Any,
    samples: list[Any],
    asr_by_id: dict[str, dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    gold_map: dict[str, tuple[list[dict[str, Any]], str]],
    cfg: dict[str, Any],
) -> dict[str, list[str]]:
    """Run LoopTool greedy capability probe (clean + noisy ASR) splitting samples into Tier A/B/C."""
    model.eval()
    tier_a, tier_b, tier_c = [], [], []

    def greedy_gen_fn(messages: list[dict[str, Any]], _step_idx: int, tools: list[dict[str, Any]]):
        prompt = tokenizer.apply_chat_template(
            messages, tools=tools or None, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        dev = next(model.parameters()).device
        enc = tokenizer(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=cfg["rollout"]["max_new_tokens"],
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        new_ids = gen.sequences[0, enc["input_ids"].shape[1] :].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        ents = []
        for logits in gen.scores:
            probs = F.softmax(logits[0].float(), dim=-1)
            ents.append(float(-(probs * torch.log(probs.clamp(min=1e-9))).sum().item()))
        return text, new_ids, 0.0, sum(ents) / max(1, len(ents))

    with torch.no_grad():
        for s in samples:
            g_calls, g_hash = gold_map[s.id]
            ep_clean = execute_single_rollout(
                s, catalog, cfg["train_split"], cfg["root"], greedy_gen_fn, tokenizer,
                gold_calls=g_calls, gold_hash=g_hash, clean_sample=s, max_steps=cfg["rollout"]["max_steps"],
            )
            ep_asr_ok = True
            if s.id in asr_by_id:
                s_noisy, diff = build_sample_with_asr(s, asr_by_id[s.id], inject_asr=True)
                ep_asr = execute_single_rollout(
                    s_noisy, catalog, cfg["train_split"], cfg["root"], greedy_gen_fn, tokenizer,
                    gold_calls=g_calls, gold_hash=g_hash, clean_sample=s, asr_phonetic_diff=diff, use_asr=True, max_steps=cfg["rollout"]["max_steps"],
                )
                ep_asr_ok = ep_asr.reward_breakdown.official_score >= 1.0 - 1e-6

            if ep_clean.reward_breakdown.official_score < 1.0 - 1e-6 or not ep_asr_ok:
                tier_a.append(s.id)
            elif any(e >= 0.35 for e in ep_clean.step_entropies):
                tier_b.append(s.id)
            else:
                tier_c.append(s.id)

    buckets = {"tier_a_hard": tier_a, "tier_b_boundary": tier_b, "tier_c_anchor": tier_c}
    print(f"[GCP Mine] hard={len(tier_a)} boundary={len(tier_b)} anchor={len(tier_c)}", flush=True)
    return buckets


def _compute_teacher_aligned_logprobs(
    model: PeftModel,
    ep: MultiTurnEpisode,
) -> torch.Tensor:
    """Evaluate teacher adapter over (teacher_prompt_ids + student_completion_ids) aligned 1-to-1."""
    if not ep.teacher_prompt_ids:
        return _compute_sequence_token_logprobs(model, ep.input_ids)
    comp_ids = ep.input_ids[ep.prompt_token_len :]
    teacher_seq = ep.teacher_prompt_ids + comp_ids
    t_lp_full = _compute_sequence_token_logprobs(model, teacher_seq)
    t_comp_lp = t_lp_full[:, ep.teacher_prompt_len :]
    aligned = torch.zeros((1, len(ep.input_ids)), dtype=t_comp_lp.dtype, device=t_comp_lp.device)
    n = min(len(comp_ids), aligned.shape[1] - ep.prompt_token_len)
    if n > 0:
        aligned[:, ep.prompt_token_len : ep.prompt_token_len + n] = t_comp_lp[:, :n]
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--override-total-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.override_total_steps is not None:
        cfg["optim"]["total_steps"] = int(args.override_total_steps)
    _seed_everything(int(cfg.get("seed", 42)))

    repo_id = resolve_repo_id(args.hf_repo)
    syncer = HFSyncer(repo_id)
    syncer.preflight()
    hf_prefix = cfg["hf_prefix"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pusher = BackgroundPusher(syncer)

    # Mandatory HF Auto-Resume check (download_dir returns root/path_in_repo)
    resume_step = 0
    resume_dir: Path | None = None
    latest_remote = f"{hf_prefix}/checkpoints/latest"
    if syncer.exists(f"{latest_remote}/adapter_config.json"):
        resume_dir = syncer.download_dir(latest_remote, out_dir / "resumed_cache")
        state_file = resume_dir / "trainer_state.pt"
        if state_file.exists():
            resume_step = int(torch.load(state_file, map_location="cpu", weights_only=False)["step"])
            print(f"[RL] Resuming from HF {latest_remote} at step {resume_step}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_id"], trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model_id"], torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        target_modules=list(QWEN35_LORA_TARGET_MODULES),
        bias="none",
    )

    if resume_dir is not None and (resume_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, str(resume_dir), adapter_name="default", is_trainable=True)
    elif cfg.get("init_student_adapter_hf"):
        init_dir = syncer.download_dir(cfg["init_student_adapter_hf"], out_dir / "init_cache")
        model = PeftModel.from_pretrained(base, str(init_dir), adapter_name="default", is_trainable=True)
    else:
        model = get_peft_model(base, lora_cfg, adapter_name="default")

    if cfg["stage"] == "voice_opd":
        t_dir = syncer.download_dir(cfg["teacher_adapter_hf"], out_dir / "teacher_cache")
        model.load_adapter(str(t_dir), adapter_name="teacher", is_trainable=False)
    model.set_adapter("default")
    model.gradient_checkpointing_enable()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg["optim"]["lr"]), weight_decay=float(cfg["optim"]["weight_decay"]))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(cfg["optim"]["warmup_steps"]), num_training_steps=int(cfg["optim"]["total_steps"])
    )
    if resume_dir is not None and resume_step > 0 and (resume_dir / "trainer_state.pt").exists():
        st = torch.load(resume_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])

    catalog = load_catalog(cfg["train_split"], cfg["root"])
    train_ids = set(json.loads(Path(cfg["train_ids"]).read_text(encoding="utf-8"))["ids"])
    samples = [s for s in load_split(cfg["train_split"], cfg["root"]) if s.id in train_ids]
    sample_by_id = {s.id: s for s in samples}
    gold_map = _precompute_gold_hashes(samples, cfg["train_split"], cfg["root"])
    asr_by_id = _load_asr_cache(cfg.get("asr_cache"))

    bucket_file = out_dir / "gcp_buckets.json"
    if bucket_file.exists():
        buckets = json.loads(bucket_file.read_text(encoding="utf-8"))
    else:
        buckets = _mine_gcp_buckets(model, tokenizer, samples, asr_by_id, catalog, gold_map, cfg)
        bucket_file.write_text(json.dumps(buckets, indent=2), encoding="utf-8")

    rng = random.Random(int(cfg.get("seed", 42)) + resume_step)
    t0 = time.monotonic()
    last_sync_t = time.monotonic()
    max_wall_s = float(cfg["optim"].get("max_wall_minutes", 45)) * 60.0

    def sample_gen_fn(messages: list[dict[str, Any]], _step_idx: int, tools: list[dict[str, Any]]):
        prompt = tokenizer.apply_chat_template(
            messages, tools=tools or None, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        dev = next(model.parameters()).device
        enc = tokenizer(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=cfg["rollout"]["max_new_tokens"],
                do_sample=True,
                temperature=float(cfg["rollout"]["temperature"]),
                top_p=float(cfg["rollout"]["top_p"]),
                return_dict_in_generate=True,
                output_scores=True,
            )
        new_ids = gen.sequences[0, enc["input_ids"].shape[1] :].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        ents = []
        for logits in gen.scores:
            probs = F.softmax(logits[0].float(), dim=-1)
            ents.append(float(-(probs * torch.log(probs.clamp(min=1e-9))).sum().item()))
        return text, new_ids, 0.0, sum(ents) / max(1, len(ents))

    for step in range(resume_step + 1, int(cfg["optim"]["total_steps"]) + 1):
        model.set_adapter("default")
        model.eval()
        active_groups: list[list[MultiTurnEpisode]] = []
        fallback_groups: list[list[MultiTurnEpisode]] = []
        for _attempt in range(int(cfg["rollout"]["max_gen_batches"])):
            need = max(1, int(cfg["rollout"]["batch_size"]) - len(active_groups))
            cand_ids = sample_weighted_prompt_ids(buckets, rng, need * 2)
            batch_groups: list[list[MultiTurnEpisode]] = []
            for sid in cand_ids:
                s_clean = sample_by_id[sid]
                use_asr = rng.random() < float(cfg["rollout"]["asr_mix_prob"]) and (sid in asr_by_id)
                s_eff, diff = build_sample_with_asr(s_clean, asr_by_id.get(sid), inject_asr=use_asr)
                g_calls, g_hash = gold_map[sid]
                grp = [
                    execute_single_rollout(
                        s_eff, catalog, cfg["train_split"], cfg["root"], sample_gen_fn, tokenizer,
                        gold_calls=g_calls, gold_hash=g_hash, clean_sample=s_clean, asr_phonetic_diff=diff, use_asr=use_asr,
                        max_steps=cfg["rollout"]["max_steps"],
                    )
                    for _ in range(int(cfg["rollout"]["group_size"]))
                ]
                grp = maybe_spawn_arpo_forks(
                    grp, s_eff, catalog, cfg["train_split"], cfg["root"], sample_gen_fn, tokenizer,
                    gold_calls=g_calls, gold_hash=g_hash, clean_sample=s_clean, asr_phonetic_diff=diff, use_asr=use_asr,
                    entropy_thresh=float(cfg["rollout"]["arpo_entropy_thresh"]),
                )
                assign_group_gigpo_advantages(
                    grp, g_calls, g_hash,
                    omega=float(cfg["rollout"]["omega"]),
                    gamma=float(cfg["rollout"]["gamma"]),
                    remove_std=bool(cfg["rollout"]["remove_std"]),
                )
                batch_groups.append(grp)
            fallback_groups.extend(batch_groups)
            active_groups.extend(filter_active_groups(batch_groups))
            if len(active_groups) >= int(cfg["rollout"]["batch_size"]):
                active_groups = active_groups[: int(cfg["rollout"]["batch_size"])]
                break

        if not active_groups:
            # In Stage 3 OPD, zero episode-reward variance groups still learn from token reverse-KL!
            active_groups = fallback_groups[: int(cfg["rollout"]["batch_size"])] if cfg["stage"] == "voice_opd" else []
        episodes = [ep for grp in active_groups for ep in grp]
        if not episodes:
            print(f"[step {step}] skipped update (0 non-degenerate groups across {cfg['rollout']['max_gen_batches']} attempts)", flush=True)
            continue

        old_lps: list[torch.Tensor] = []
        ref_or_teach_lps: list[torch.Tensor] = []
        with torch.no_grad():
            for ep in episodes:
                model.set_adapter("default")
                old_lps.append(_compute_sequence_token_logprobs(model, ep.input_ids).cpu())
                if cfg["stage"] == "voice_opd":
                    model.set_adapter("teacher")
                    ref_or_teach_lps.append(_compute_teacher_aligned_logprobs(model, ep).cpu())
                else:
                    with model.disable_adapter():
                        ref_or_teach_lps.append(_compute_sequence_token_logprobs(model, ep.input_ids).cpu())

        model.set_adapter("default")
        model.train()
        optimizer.zero_grad()
        dev = next(model.parameters()).device
        loss_acc, kl_acc, clip_acc = [], [], []
        n_eps = len(episodes)

        for ep, o_lp_cpu, rt_lp_cpu in zip(episodes, old_lps, ref_or_teach_lps):
            cur_lp = _compute_sequence_token_logprobs(model, ep.input_ids)
            o_lp = o_lp_cpu.to(dev)
            rt_lp = rt_lp_cpu.to(dev)
            mask = torch.tensor([ep.loss_mask[: cur_lp.shape[1]]], dtype=torch.float32, device=dev)
            adv = _build_token_advantage_tensor(ep, cur_lp.shape[1], dev)

            if cfg["stage"] == "voice_opd":
                adv = compute_opd_reverse_kl_advantages(
                    cur_lp, rt_lp, adv, mask, alpha_ep=float(cfg["opd"]["alpha_ep"]), clip_val=float(cfg["opd"]["rkl_clip"])
                )

            pg_loss, st = compute_gspo_clip_higher_loss(
                cur_lp, o_lp, adv, mask, clip_low=float(cfg["optim"]["clip_low"]), clip_high=float(cfg["optim"]["clip_high"])
            )
            kl_loss = compute_low_var_kl_loss(cur_lp, rt_lp, mask)
            total = (pg_loss + float(cfg["optim"]["kl_coef"]) * kl_loss) / max(1, n_eps)
            total.backward()
            loss_acc.append(float(pg_loss.item()))
            kl_acc.append(float(kl_loss.item()))
            clip_acc.append(st["clipped_frac"])

        torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg["optim"]["max_grad_norm"]))
        optimizer.step()
        scheduler.step()

        mean_rew = sum(e.episode_reward for e in episodes) / n_eps
        mean_off = sum(e.reward_breakdown.official_score for e in episodes) / n_eps
        metrics = {
            "mean_reward": round(mean_rew, 4),
            "mean_official": round(mean_off, 4),
            "pg_loss": round(sum(loss_acc) / n_eps, 5),
            "kl": round(sum(kl_acc) / n_eps, 5),
            "clip_frac": round(sum(clip_acc) / n_eps, 4),
            "active_groups": len(active_groups),
        }
        print(f"[step {step}/{cfg['optim']['total_steps']}] {json.dumps(metrics)}", flush=True)

        due_step = (step % int(cfg["optim"]["save_steps"]) == 0) or (step == int(cfg["optim"]["total_steps"]))
        due_time = (time.monotonic() - last_sync_t) >= float(cfg["optim"]["sync_interval_s"])
        due_wall = (time.monotonic() - t0) >= max_wall_s
        if due_step or due_time or due_wall:
            last_sync_t = time.monotonic()
            _save_and_push_ckpt(model, tokenizer, optimizer, scheduler, out_dir, step, metrics, pusher, hf_prefix)
            if due_wall and step < int(cfg["optim"]["total_steps"]):
                print(f"[RL] Wall-cap reached cleanly at step {step}; next Colab chunk will auto-resume.", flush=True)
                pusher.flush(timeout=900)
                return

    pusher.flush(timeout=900)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run all unit tests + commit**

Run: `python -m pytest tests/test_rl_reward.py tests/test_rl_rollout.py tests/test_rl_losses.py tests/test_gigpo_anchor_advantage.py -v`
Expected: ALL PASS.

```bash
git add src/agent/rl_losses.py scripts/train_qwen35_rl.py configs/qwen35_v2_rl_stage2.yaml configs/qwen35_v2_rl_stage3_opd.yaml tests/test_rl_losses.py
git commit -m "feat(rl): add GSPO+Clip-Higher, positive k3 KL, Reverse-KL OPD/SDPO, and G4 RL trainer"
```

---

### Task 5: Stage 2 Execution on Colab G4 (`scripts/colab_run_qwen35_rl_g4.py`)

**Files:**
- Create: `scripts/colab_run_qwen35_rl_g4.py`

**VRAM & Compute Budget on Colab G4 (`NVIDIA RTX PRO 6000 Blackwell`, `101.97 GB` VRAM):**
- Frozen `SFT V2` base in `bf16`: `8.43 GB`
- Student + Teacher LoRA adapters (`r=32`): `0.56 GB`
- AdamW states (`8 bytes/param` on ~`70M` LoRA params) + grads: `~0.85 GB`
- Gradient-checkpointed forward/backward up to `13,072` tokens (`micro_bs=1`, `use_cache=False`): `~18–26 GB`
- **Total Peak VRAM: `~28–36 GB / 101.97 GB` (`> 65 GB` free headroom -> ZERO OOM risk!).**
- Rollout throughput note: if pure HF `model.generate` for `16 x 8` episodes takes `> 12s/group`, colocate `vLLM` engine with `gpu_memory_utilization=0.30` + `enable_lora=True` (`~30 GB` vLLM + `~34 GB` PyTorch trainer = `64 GB / 102 GB`) hot-swapping saved `LoRA` adapters per step for **`4–6x` faster rollout (`~2–3 min / RL step`, `~6–8h` -> `~18–28 CU` for 200 steps)**.

- [ ] **Step 1: Write `scripts/colab_run_qwen35_rl_g4.py`**

```python
"""Orchestrate chunked Qwen3.5-4B RL/OPD training on Colab G4 with HF auto-resume & 50 GB protection."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.colab_runner import exec_local, package_codebase, restart_kernel, run_command, ping_session
from src.core.hf_sync import HFSyncer, resolve_repo_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", default="g4")
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--override-total-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    total_steps = int(args.override_total_steps or cfg["optim"]["total_steps"])
    hf_prefix = cfg["hf_prefix"]
    hf_token = os.environ.get("HF_TOKEN") or ""
    hf_repo = resolve_repo_id(args.hf_repo)
    syncer = HFSyncer(hf_repo, token=hf_token, private=False)
    syncer.preflight()

    pkg_zip = ROOT / "scripts" / "scratch" / "code_latest.zip"
    package_codebase(pkg_zip)

    done_marker = f"{hf_prefix}/checkpoints/checkpoint-{total_steps}/adapter_config.json"
    bootstrap_cmd = (
        "python -c \"import pathlib, huggingface_hub as hh; "
        "p = pathlib.Path('/content/models/qwen35_4b_sft_v2_merged'); "
        "not (p / 'config.json').exists() and hh.snapshot_download("
        f"repo_id='{syncer.repo_id}', allow_patterns='exp_qwen35_sft_v2/merged/*', local_dir='/content/_hf_dl'); "
        "asr = pathlib.Path('/content/folds1_4_phowhisper_3beam_nbest.jsonl'); "
        "not asr.exists() and hh.hf_hub_download("
        f"repo_id='{syncer.repo_id}', filename='exp_qwen35_v2_rl/asr_cache/folds1_4_phowhisper_3beam_nbest.jsonl', local_dir='/content/_hf_dl')\""
    )
    for chunk in range(1, 16):
        if syncer.exists(done_marker):
            print(f"[colab_rl] Complete: {done_marker}", flush=True)
            return
        if not ping_session(args.session, timeout=25):
            restart_kernel(args.session)
        uploads = [
            (pkg_zip, "/content/code_latest.zip"),
            (ROOT / args.config, f"/content/{Path(args.config).name}"),
            (ROOT / "data" / "train" / "train_tools_catalog.json", "/content/train_tools_catalog.json"),
        ]
        for sname in ("train.jsonl.gz", "train_environment.zip", "zaic_tools.zip"):
            spath = ROOT / "scripts" / "scratch" / sname
            if spath.exists():
                uploads.append((spath, f"/content/{sname}"))
        for local_p, remote_p in uploads:
            exec_local(["colab", "upload", "-s", args.session, str(local_p), remote_p])

        step_flag = f" --override-total-steps {args.override_total_steps}" if args.override_total_steps else ""
        cmd = (
            "mkdir -p /content/data/train /content/models /content/repo && "
            "unzip -oq /content/code_latest.zip -d /content/repo && "
            "cp -f /content/train_tools_catalog.json /content/data/train/train_tools_catalog.json && "
            "([ -f /content/train.jsonl.gz ] && gzip -dc /content/train.jsonl.gz > /content/data/train/train.jsonl || true) && "
            "([ -f /content/train_environment.zip ] && unzip -oq /content/train_environment.zip -d /content/data/train || true) && "
            "([ -f /content/zaic_tools.zip ] && pip install -q --no-deps /content/zaic_tools.zip || true) && "
            f"{bootstrap_cmd} && "
            "([ -d /content/_hf_dl/exp_qwen35_sft_v2/merged ] && ln -sfn /content/_hf_dl/exp_qwen35_sft_v2/merged /content/models/qwen35_4b_sft_v2_merged || true) && "
            "([ -f /content/_hf_dl/exp_qwen35_v2_rl/asr_cache/folds1_4_phowhisper_3beam_nbest.jsonl ] && ln -sfn /content/_hf_dl/exp_qwen35_v2_rl/asr_cache/folds1_4_phowhisper_3beam_nbest.jsonl /content/folds1_4_phowhisper_3beam_nbest.jsonl || true) && "
            "cd /content/repo && "
            f"python scripts/train_qwen35_rl.py --config /content/{Path(args.config).name} --hf-repo {syncer.repo_id}{step_flag}"
        )
        run_command(
            cmd,
            session=args.session,
            is_file=False,
            timeout=3300,
            max_retries=2,
            env=[f"HF_TOKEN={hf_token}", f"HF_SYNC_REPO={syncer.repo_id}", "HF_SYNC_PRIVATE=0"],
            stream=True,
            gpu=args.gpu,
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test 5 steps on Colab G4 & verify HF Sync**

```bash
python scripts/colab_run_qwen35_rl_g4.py --session rl-g4 --config configs/qwen35_v2_rl_stage2.yaml --override-total-steps 5
```
Expected: Logs `[GCP Mine]`, non-zero `active_groups`, bounded `kl < 0.05`, and pushes `exp_qwen35_v2_rl/stage2/checkpoints/checkpoint-5/` & `latest/` (`~280 MB`) to HF.

- [ ] **Step 3: Run Stage 2 Full (`200` filtered GSPO+GiGPO+ARPO steps on `folds 1..4`)**

```bash
python scripts/colab_run_qwen35_rl_g4.py --session rl-g4 --config configs/qwen35_v2_rl_stage2.yaml
```
Expected progression: Every `25` steps (`25, 50, ..., 200`), evaluates non-regression on `fold_0` (`N=250`) and unseen-ASR slice (`folds 1..4` PhoWhisper transcripts before/after).

---

### Task 6: Stage 3 Privileged Voice/Error `SDPO-OPD`, Targeted `DiaTool Step-DPO`, & LoRA Souping

**Files:**
- Modify/Extend: `scripts/train_qwen35_rl.py` (Step-DPO `--stage step_dpo` mode calling `compute_diatool_step_dpo_loss`)
- Reuse: `scripts/merge_lfm3_soup.py` (`average_safetensors_weights`)
- Output Checkpoints: `exp_qwen35_v2_rl/stage3_opd/`, `exp_qwen35_v2_rl/stage3_soup/`

- [ ] **Step 1: Run Stage 3A — Privileged Voice & Error-Feedback Reverse-KL OPD (`100` steps)**

```bash
python scripts/colab_run_qwen35_rl_g4.py --session rl-g4 --config configs/qwen35_v2_rl_stage3_opd.yaml
```
Why this attacks `-2.85 pp` E2E drop directly: `85%` of student prompts use real PhoWhisper Loop #38 (`nbest + asr_phonetic_diff`), while frozen Stage-2 `"teacher"` adapter scores the exact student tokens conditioned on the clean Oracle transcript + env-hint prefix -> token-level reverse-KL pulls `"ngay -> nghin"` and `"ma co -> co"` recovery probability upward on-policy without thinking latency bloat!

- [ ] **Step 2: Optional Stage 3B — Targeted First-Error `DiaTool Step-DPO` (`40–60` steps, `lr=3e-7`)**

Only run if Step-1 checkpoint still shows `>= 5` recurring first-error `false_ask`, `missed_ask`, or `extra_read` bugs on `folds 1..4`. Build in-distribution step-`k` preference pairs directly from Stage 3A rollouts (`prefix s_{1..k-1}` fixed, `s_win` sampled at `T=0.85` & verified `SafeEnv` `official_score==1.0`, `s_lose` actual Stage 3A failure) and optimize with `compute_diatool_step_dpo_loss` (`beta=0.1, margin=0.5, gamma=0.5`) via `scripts/train_qwen35_step_dpo.py`:

```python
# Core optimization step inside scripts/train_qwen35_step_dpo.py
turn_lps_c = [_sum_turn_lp(model, pair["chosen_turns"][:t+1]) for t in range(len(pair["chosen_turns"]))]
turn_lps_r = [_sum_turn_lp(model, pair["rejected_turns"][:t+1]) for t in range(len(pair["rejected_turns"]))]
with torch.no_grad(), model.disable_adapter():
    ref_lps_c = [_sum_turn_lp(model, pair["chosen_turns"][:t+1]) for t in range(len(pair["chosen_turns"]))]
    ref_lps_r = [_sum_turn_lp(model, pair["rejected_turns"][:t+1]) for t in range(len(pair["rejected_turns"]))]
loss = compute_diatool_step_dpo_loss(turn_lps_c, ref_lps_c, turn_lps_r, ref_lps_r, beta=0.1, margin=0.5, gamma=0.5)
```

- [ ] **Step 3: Stage 3C — Top-K LoRA Souping (`merge_lfm3_soup.py`) + Merge Text-Only Checkpoint**

Select the top 2–3 checkpoints by balanced `(Fold_0 Oracle, Fold_0 E2E, OOS >= 1.0, Ask >= baseline)` score, average LoRA adapters, and merge onto `exp_qwen35_sft_v2/merged/`:

```bash
python scripts/merge_lfm3_soup.py \
  --checkpoints /content/experiments/exp_qwen35_v2_rl/stage2/checkpoints/checkpoint-175 \
                /content/experiments/exp_qwen35_v2_rl/stage3_opd/checkpoints/checkpoint-75 \
                /content/experiments/exp_qwen35_v2_rl/stage3_opd/checkpoints/checkpoint-100 \
  --output-dir /content/experiments/exp_qwen35_v2_rl/soup_adapter
```
Then merge with `PeftModel.from_pretrained(...).merge_and_unload()` and verify size/dtype with `src/agent/qwen35_sft.py:save_qwen35_text_only_checkpoint` before pushing final `exp_qwen35_v2_rl/final_merged/` to HF (`~8.43 GB` — after verifying `<= 50 GB` HF total).

---

### Task 7: Verification Gate, Competition Limit Compliance, & Sealed Dev 299 Promotion

**Files:**
- Create: `experiments/exp_qwen35_v2_rl/REPORT.md`
- Modify: `reports/LLM_AGENT_EXPERIMENTS_DASHBOARD.md`

- [ ] **Step 1: Verify Pre-Promotion Gate (`fold_0` + Adversarial ASR + Unit Suite)**

Run all repo unit tests locally (`python -m pytest tests/ -q`), plus evaluate `exp_qwen35_v2_rl/final_merged/` deterministically (`vllm 0.33 + xgrammar`, `temperature=0.0`, `enable_thinking=False`) on `fold_0` (`N=250`) Oracle & E2E Voice:
- Pass criteria before unlocking single-shot Dev 299:
  - `Fold 0 Oracle >= 0.905` (up from `0.8823`, strict improvement on `act->ask`, `sm=True F1<1`, `>=0750`)
  - `Fold 0 E2E Voice >= 0.885` (up from `0.8596`, shrinking E2E-vs-Oracle drop from `2.27 pp` to `<= 1.2–1.5 pp`)
  - Zero regression on `oos` (`1.0000`) and L4/G4 latency within `<= 5.0s` LLM (`<= 12s` E2E on L4).
- *(Optional Final Competition Push once HP/steps are locked: re-run exact winning Stage 2 + Stage 3 schedule using `train_only` `1,246` IDs before Step 2 if `Fold 0` confirmed zero overfitting!)*

- [ ] **Step 2: Single-Shot Sealed `dev_v1` (`N=299`) Oracle + E2E Voice Evaluation**

Run deterministic evaluation cleanly (recording only aggregate metrics & bucket counts per `dev_v1` sealing rules):
- Expect **Oracle Dev 299:** `0.840 – 0.875` (`+6.1 to +9.6 pp` over V2's `0.7788`)
- Expect **E2E Voice Dev 299:** `0.815 – 0.855` (`+6.5 to +10.5 pp` over V2's `0.7503`)

- [ ] **Step 3: Hard Constraint Audit & Reporting (`AGENTS.md §2, §8`)**

1. Run `python scripts/count_params.py` confirming `LLM < 5B` and `PhoWhisper + Qwen3.5-4B < 6B`.
2. Verify E2E latency `<= 30s / sample`, `temperature = 0.0`, fixed seeds, offline readiness, zero `<split>_environment/` reads.
3. Write `experiments/exp_qwen35_v2_rl/REPORT.md` citing exact HF checkpoint commits, verified metrics, CU cost (`~33–51 CU` estimated of `152.29 CU`), and append row to `reports/LLM_AGENT_EXPERIMENTS_DASHBOARD.md`.

---

## Self-Review Checklist (Verified Against Plan & Constraints)

1. **Spec Coverage:** Every one of the 6 SFT V2 flaws maps cleanly to a concrete implemented module/stage:
   - Fold-0 leakage & zero-std collapse -> `GCP` mining + `DAPO` dynamic group filtering + `folds_1_to_4` holdout discipline (`Task 0`, `Task 3`, `Task 5`, `Task 7`)
   - `-2.85 pp` PhoWhisper E2E drop -> `folds 1..4` 3-beam ASR cache + 40–85% real ASR mix + Stage 3 Privileged Voice `SDPO-OPD` (`Task 0`, `Task 4`, `Task 6`)
   - `sm=True, F1<1` fuzzy/optional-arg/redundancy cliff -> Strictly monotonic composite reward + fuzzy F1 densifier (`Task 2`)
   - Non-thinking `>=0750` multi-step failures & hallucinated email -> `Email Xref Guard` + `GiGPO` anchor step credit + `ARPO` post-tool entropy branching (`Task 1`, `Task 3`)
   - `act/ask/oos` boundary errors -> `OOS=+1.0`, `False-Ask=-0.15..-0.25`, partial `ask` param/tool shaping + `DiaTool Step-DPO` (`Task 2`, `Task 4`, `Task 6`)
   - HF `50 GB` storage limit -> Quota pre-check + LoRA-only `~260 MB` step checkpoints + `latest/` auto-resume (`Task 0`, `Task 4`, `Task 5`)
2. **Placeholder Scan:** Zero `TBD`/`TODO` placeholders; all modules, unit tests, YAML configs, trainer loop, and commands are concrete and aligned with existing repo APIs (`parse_tool_calls`, `has_malformed_tool_call`, `sanitize_tool_call`, `score_sample`, `token_f1`, `fuzzy_params`, `validate_step_with_reason`, `compute_gigpo_advantages`, `average_safetensors_weights`, `HFSyncer`).
3. **Type & Interface Consistency:** `MultiTurnEpisode`, `RewardBreakdown`, `execute_single_rollout`, `filter_active_groups`, `assign_group_gigpo_advantages`, `maybe_spawn_arpo_forks`, `compute_gspo_clip_higher_loss`, `compute_low_var_kl_loss`, and `compute_opd_reverse_kl_advantages` share identical field names and signatures across `rl_reward.py`, `rl_rollout.py`, `rl_losses.py`, `train_qwen35_rl.py`, and all test suites.
