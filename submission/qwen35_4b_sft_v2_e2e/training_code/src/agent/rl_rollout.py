"""Multi-turn SafeEnv rollout collector with assistant-only loss masking, GiGPO grouping,
ARPO post-tool entropy branching, ASR transcript mixing, and DAPO zero-std filtering."""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agent.history import build_messages, truncate_tool_result
from src.agent.lfm3_gigpo import EpisodeRollout, TrajectoryStepInfo, compute_gigpo_advantages
from src.agent.rl_reward import (
    RewardBreakdown,
    canonical_state_anchor,
    compute_episode_reward,
    compute_step_format_penalty,
)
from src.agent.tool_caller import has_malformed_tool_call, parse_tool_calls, sanitize_tool_call
from src.agent.tool_selector import select_tools
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
                txt = str(hist_map[str(idx)])
                turn.audio_text = txt
                turn.text = txt
    return s_copy, asr_row.get("asr_phonetic_diff")


def compute_assistant_loss_mask_positions(
    tokenizer: Any,
    segments: list[tuple[str, str]],
) -> tuple[list[int], list[int], list[int]]:
    """Convert ('prompt'|'tool'|'assistant'|'forced_assistant', text) segments into (ids, loss_mask, step_idx_per_token)."""
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
        if kind in ("assistant", "forced_assistant"):
            step_counter += 1
    return all_ids, all_mask, all_steps


def _safe_tool_call_dict(call_obj: Any) -> dict[str, Any]:
    return {"name": call_obj.name, "arguments": dict(call_obj.arguments)}


@dataclass
class RolloutRequest:
    sample: Sample
    gold_calls: list[dict[str, Any]]
    gold_hash: str
    clean_sample: Sample | None = None
    asr_phonetic_diff: str | None = None
    use_asr: bool = False
    forced_prefix_calls: list[dict[str, Any]] | None = None
    prefilled_step0: tuple[str, list[int] | None, float, float] | None = None


@dataclass
class _RolloutState:
    sample: Sample
    catalog: dict[str, dict[str, Any]]
    split: str
    root: str
    gold_calls: list[dict[str, Any]]
    gold_hash: str
    gold_names: list[str]
    env: SafeEnv
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    prompt_token_len: int
    teacher_prompt_ids: list[int]
    teacher_prompt_len: int
    segments: list[tuple[str, str]]
    use_asr: bool
    max_steps: int
    max_prompt_tokens: int
    max_seq_tokens: int
    forced_prefix_calls: list[dict[str, Any]] | None = None
    prefilled_step0: tuple[str, list[int] | None, float, float] | None = None
    steps: list[TrajectoryStepInfo] = field(default_factory=list)
    step_entropies: list[float] = field(default_factory=list)
    first_step_calls: list[dict[str, Any]] = field(default_factory=list)
    invalid_steps: int = 0
    extra_reads: int = 0
    malformed_terminal: bool = False
    done: bool = False


def _format_forced_prefix(forced_prefix_calls: list[dict[str, Any]]) -> str:
    return "".join(
        f"<tool_call>\n<function={c['name']}>\n"
        + "".join(
            f"<parameter={k}>\n{json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}\n</parameter>\n"
            for k, v in (c.get("arguments") or {}).items()
        )
        + "</function>\n</tool_call>"
        for c in forced_prefix_calls
    )


def _init_rollout_state(
    req: RolloutRequest,
    catalog: dict[str, dict[str, Any]],
    split: str,
    root: str,
    tokenizer: Any,
    *,
    max_steps: int = 8,
    max_prompt_tokens: int = 15200,
    max_seq_tokens: int = 16384,
) -> _RolloutState:
    sample = req.sample
    env = SafeEnv(split, sample.id, root)
    messages = build_messages(
        sample,
        current_text=sample.audio_text,
        asr_phonetic_diff=req.asr_phonetic_diff,
        prompt_style="qwen35",
        max_prompt_tokens=max_prompt_tokens,
    )
    tools = select_tools(sample, catalog)

    initial_prompt_text = tokenizer.apply_chat_template(
        messages, tools=tools or None, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_token_len = len(tokenizer.encode(initial_prompt_text, add_special_tokens=False))
    if req.clean_sample is not None and req.use_asr:
        clean_msgs = build_messages(
            req.clean_sample,
            current_text=req.clean_sample.audio_text,
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

    return _RolloutState(
        sample=sample,
        catalog=catalog,
        split=split,
        root=root,
        gold_calls=req.gold_calls,
        gold_hash=req.gold_hash,
        gold_names=[str(c.get("name") or "") for c in req.gold_calls],
        env=env,
        messages=messages,
        tools=tools,
        prompt_token_len=prompt_token_len,
        teacher_prompt_ids=teacher_prompt_ids,
        teacher_prompt_len=teacher_prompt_len,
        segments=[("prompt", initial_prompt_text)],
        use_asr=req.use_asr,
        max_steps=max_steps,
        max_prompt_tokens=max_prompt_tokens,
        max_seq_tokens=max_seq_tokens,
        forced_prefix_calls=req.forced_prefix_calls,
        prefilled_step0=req.prefilled_step0,
    )


def _advance_rollout_step(
    st: _RolloutState,
    step_idx: int,
    raw_text: str,
    step_ent: float,
    tokenizer: Any,
    *,
    is_forced: bool = False,
) -> None:
    parsed = parse_tool_calls(raw_text, st.catalog)
    parsed = [sanitize_tool_call(c, st.sample, st.sample.audio_text) for c in parsed]
    parsed_dicts = [_safe_tool_call_dict(c) for c in parsed]
    st.step_entropies.append(float(step_ent))

    is_terminal_text = not parsed
    if has_malformed_tool_call(raw_text, st.catalog) and not parsed:
        st.malformed_terminal = True
    pen, valid_step = compute_step_format_penalty(
        raw_text, parsed_dicts, st.catalog, is_last_turn=is_terminal_text
    )
    if not valid_step:
        st.invalid_steps += 1

    # Supervise <|im_end|> on assistant turn; keep trailing delimiter '\n' in unmasked 'tool' segment
    st.segments.append(("forced_assistant" if is_forced else "assistant", f"{raw_text}<|im_end|>"))

    if not parsed:
        anchor = canonical_state_anchor(st.sample.id, step_idx, [], [], st.malformed_terminal)
        st.steps.append(TrajectoryStepInfo(step_idx=step_idx, anchor_state=anchor, step_reward=pen))
        st.done = True
        return

    step_results: list[Any] = []
    st.messages.append({"role": "assistant", "content": raw_text})
    ask_stop = False
    tool_resp_msgs: list[dict[str, Any]] = []
    for c in parsed:
        if c.name not in st.gold_names and _is_lookup_tool(c.name):
            st.extra_reads += 1
        if c.name == ASK_MISSING_INFO:
            val_ask = st.env.validate(c.name, c.arguments)
            ok_ask = bool(val_ask.get("ok", False)) if isinstance(val_ask, dict) else bool(val_ask)
            err_ask = val_ask.get("error") if isinstance(val_ask, dict) else str(val_ask)
            if ok_ask:
                st.env.call(c.name, c.arguments)
                step_results.append({"asked": True})
                ask_stop = True
                break
            st.invalid_steps += 1
            res = {"error": err_ask}
            step_results.append(res)
            resp_msg = {"role": "tool", "name": c.name, "content": truncate_tool_result(res, max_chars=1500)}
            st.messages.append(resp_msg)
            tool_resp_msgs.append(resp_msg)
            continue
        val = st.env.validate(c.name, c.arguments)
        ok = bool(val.get("ok", False)) if isinstance(val, dict) else bool(val)
        err = val.get("error") if isinstance(val, dict) else str(val)
        res = st.env.call(c.name, c.arguments) if ok else {"error": err}
        step_results.append(res)
        resp_msg = {"role": "tool", "name": c.name, "content": truncate_tool_result(res, max_chars=1500)}
        st.messages.append(resp_msg)
        tool_resp_msgs.append(resp_msg)

    if step_idx == 0:
        st.first_step_calls = list(parsed_dicts)

    anchor = canonical_state_anchor(st.sample.id, step_idx, parsed_dicts, step_results, not valid_step)
    st.steps.append(TrajectoryStepInfo(step_idx=step_idx, anchor_state=anchor, step_reward=pen))

    if ask_stop:
        st.done = True
        return

    if tool_resp_msgs:
        before_obs = tokenizer.apply_chat_template(
            st.messages[: -len(tool_resp_msgs)],
            tools=st.tools or None,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        after_obs = tokenizer.apply_chat_template(
            st.messages,
            tools=st.tools or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        obs_delta = after_obs[len(before_obs) :] if after_obs.startswith(before_obs) else after_obs
        if before_obs.endswith("\n") and not obs_delta.startswith("\n"):
            obs_delta = "\n" + obs_delta
        st.segments.append(("tool", obs_delta))

        approx_chars = sum(len(t) for _, t in st.segments)
        if approx_chars > 38000:
            cur_toks = len(tokenizer.encode("".join(t for _, t in st.segments), add_special_tokens=False))
            if cur_toks >= st.max_seq_tokens - 420:
                st.done = True


def _finalize_rollout_episode(st: _RolloutState, tokenizer: Any) -> MultiTurnEpisode:
    pred_calls = st.env.get_tool_calls()
    pred_hash = st.env.get_state_hash()
    bd = compute_episode_reward(
        pred_calls,
        pred_hash,
        st.gold_calls,
        st.gold_hash,
        st.catalog,
        invalid_steps=st.invalid_steps,
        extra_reads=st.extra_reads,
        malformed_terminal=st.malformed_terminal,
    )
    if st.steps:
        st.steps[-1].step_reward += bd.episode_reward

    input_ids, loss_mask, step_ids = compute_assistant_loss_mask_positions(tokenizer, st.segments)
    return MultiTurnEpisode(
        sample_id=st.sample.id,
        use_asr=st.use_asr,
        prompt_token_len=st.prompt_token_len,
        input_ids=input_ids,
        loss_mask=loss_mask,
        step_ids_per_token=step_ids,
        teacher_prompt_ids=st.teacher_prompt_ids,
        teacher_prompt_len=st.teacher_prompt_len,
        steps=st.steps,
        step_entropies=st.step_entropies,
        first_step_calls=st.first_step_calls,
        reward_breakdown=bd,
        episode_reward=bd.episode_reward,
    )


def execute_batched_rollouts(
    requests: list[RolloutRequest],
    catalog: dict[str, dict[str, Any]],
    split: str,
    root: str,
    batch_generate_fn: Callable[
        [list[tuple[list[dict[str, Any]], int, list[dict[str, Any]]]]],
        list[tuple[str, list[int] | None, float, float]],
    ],
    tokenizer: Any,
    *,
    max_steps: int = 8,
    max_prompt_tokens: int = 15200,
    max_seq_tokens: int = 16384,
) -> list[MultiTurnEpisode]:
    """Execute N live multi-turn rollouts turn-by-turn so GPU inference stays batched across active episodes."""
    if not requests:
        return []
    states = [
        _init_rollout_state(
            req,
            catalog,
            split,
            root,
            tokenizer,
            max_steps=max_steps,
            max_prompt_tokens=max_prompt_tokens,
            max_seq_tokens=max_seq_tokens,
        )
        for req in requests
    ]
    for step_idx in range(max_steps):
        pending_indices: list[int] = []
        pending_reqs: list[tuple[list[dict[str, Any]], int, list[dict[str, Any]]]] = []
        for idx, st in enumerate(states):
            if st.done:
                continue
            if step_idx == 0 and st.forced_prefix_calls:
                _advance_rollout_step(
                    st, step_idx, _format_forced_prefix(st.forced_prefix_calls), 0.0, tokenizer, is_forced=True
                )
            elif step_idx == 0 and st.prefilled_step0 is not None:
                raw_text, _tok_ids, _sum_lp, step_ent = st.prefilled_step0
                _advance_rollout_step(st, step_idx, raw_text, float(step_ent), tokenizer, is_forced=False)
            else:
                pending_indices.append(idx)
                pending_reqs.append((st.messages, step_idx, st.tools))

        if pending_reqs:
            resps = batch_generate_fn(pending_reqs)
            for st_idx, (raw_text, _tok_ids, _sum_lp, step_ent) in zip(pending_indices, resps):
                _advance_rollout_step(states[st_idx], step_idx, raw_text, float(step_ent), tokenizer, is_forced=False)

        if all(st.done for st in states):
            break

    return [_finalize_rollout_episode(st, tokenizer) for st in states]


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
    req = RolloutRequest(
        sample=sample,
        gold_calls=gold_calls,
        gold_hash=gold_hash,
        clean_sample=clean_sample,
        asr_phonetic_diff=asr_phonetic_diff,
        use_asr=use_asr,
        forced_prefix_calls=forced_prefix_calls,
    )
    return execute_batched_rollouts(
        [req],
        catalog,
        split,
        root,
        lambda batch: [generate_step_fn(msgs, sidx, tools) for msgs, sidx, tools in batch],
        tokenizer,
        max_steps=max_steps,
        max_prompt_tokens=max_prompt_tokens,
    )[0]


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


def select_arpo_anchor(
    group: list[MultiTurnEpisode],
    *,
    entropy_thresh: float = 0.45,
) -> MultiTurnEpisode | None:
    """Select the highest-entropy post-read Turn-2 episode to fork from, or None."""
    candidates = [
        ep
        for ep in group
        if ep.first_step_calls
        and any(_is_lookup_tool(c["name"]) for c in ep.first_step_calls)
        and len(ep.step_entropies) >= 2
        and ep.step_entropies[1] >= entropy_thresh
    ]
    return max(candidates, key=lambda e: e.step_entropies[1]) if candidates else None


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
    max_steps: int = 8,
    max_prompt_tokens: int = 15200,
) -> list[MultiTurnEpisode]:
    """ARPO-inspired post-tool branch sampling when Turn 2 shows high entropy or stalled group."""
    anchor_ep = select_arpo_anchor(group, entropy_thresh=entropy_thresh)
    if anchor_ep is None:
        return group
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
            max_steps=max_steps,
            max_prompt_tokens=max_prompt_tokens,
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
