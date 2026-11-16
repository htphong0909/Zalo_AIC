"""Step-wise verifiable reward function for LFM3 GRPO inheriting lfm3_grpo_with_trl.ipynb and SafeEnv."""
from __future__ import annotations

import json
from typing import Any, Callable

from src.agent.lfm3_scorer import arg_overlap_f1, calls_match, tool_name_f1
from src.agent.tool_caller import coerce_arguments, has_malformed_tool_call, parse_tool_calls
from src.env.safe_env import SafeEnv


def extract_predicted_calls(
    completion_ids: Any,
    prompt: str,
    tools: list[dict[str, Any]],
    tokenizer: Any | None = None,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Parse completion token IDs or text into a list of {'name', 'arguments'} dicts and format validity flag."""
    clean_ids = completion_ids
    if hasattr(clean_ids, "tolist"):
        clean_ids = clean_ids.tolist()
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        if isinstance(clean_ids, list):
            clean_ids = [tok for tok in clean_ids if tok != tokenizer.pad_token_id]

    if tokenizer is not None and not isinstance(clean_ids, str) and hasattr(tokenizer, "parse_response"):
        try:
            message = tokenizer.parse_response(clean_ids, prefix=prompt, tools=tools or None)
            raw_calls = [call["function"] for call in (message.get("tool_calls") or []) if "function" in call]
            if raw_calls:
                coerced = [
                    {
                        "name": c["name"],
                        "arguments": coerce_arguments(c["name"], c.get("arguments") or {}, catalog),
                    }
                    for c in raw_calls
                ]
                return coerced, True
            # If parse_response succeeded and raw_calls is empty, it's a valid text/OOS/stop response
            return [], True
        except Exception:
            pass

    if isinstance(clean_ids, str):
        text = clean_ids
    elif isinstance(clean_ids, list) and clean_ids and isinstance(clean_ids[0], dict):
        text = str(clean_ids[0].get("content") or "")
    elif tokenizer is not None:
        text = tokenizer.decode(clean_ids, skip_special_tokens=False)
    else:
        text = str(clean_ids)

    if has_malformed_tool_call(text, catalog):
        return [], False

    parsed = parse_tool_calls(text, catalog)
    return [{"name": c.name, "arguments": c.arguments} for c in parsed], True


def verify_step_in_env(
    sample_id: str,
    split: str,
    root: str,
    prefix_calls: list[dict[str, Any]],
    pred_calls: list[dict[str, Any]],
    *,
    gold_state_hash: str | None = None,
) -> tuple[bool, bool]:
    """Replay prefix_calls + pred_calls in SafeEnv, returning (all_env_valid, state_matched)."""
    if not pred_calls and not gold_state_hash:
        return True, True
    try:
        env = SafeEnv(split, sample_id, root)
        for pc in prefix_calls:
            env.call(pc["name"], pc.get("arguments") or {})
        for c in pred_calls:
            c_name = c.get("name", "")
            c_args = c.get("arguments") or {}
            if not isinstance(c_name, str) or not c_name or not isinstance(c_args, dict):
                return False, False
            val = env.validate(c_name, c_args)
            if not val.get("ok", False):
                return False, False
            res = env.call(c_name, c_args)
            if isinstance(res, dict) and res.get("ok") is False:
                return False, False
        state_ok = True
        if isinstance(gold_state_hash, str) and gold_state_hash:
            state_ok = env.get_state_hash() == gold_state_hash
        return True, state_ok
    except Exception:
        return False, False


def validate_step_in_env(
    sample_id: str,
    split: str,
    root: str,
    prefix_calls: list[dict[str, Any]],
    pred_calls: list[dict[str, Any]],
) -> bool:
    """Replay prefix_calls in SafeEnv and verify all pred_calls validate and execute cleanly."""
    if not pred_calls:
        return True
    env_ok, _ = verify_step_in_env(sample_id, split, root, prefix_calls, pred_calls)
    return env_ok


def compute_step_reward(
    gold_calls: list[dict[str, Any]],
    pred_calls: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    *,
    is_valid_format: bool = True,
    all_env_valid: bool = True,
    state_match: bool = True,
) -> float:
    """Compute multi-tier verifiable step reward in [0.0, 1.0] gated on state_match."""
    if not gold_calls:
        return 1.0 if (not pred_calls and is_valid_format and state_match) else 0.0

    if not pred_calls or not is_valid_format:
        return 0.0

    if all_env_valid and state_match and calls_match(gold_calls, pred_calls, catalog):
        return 1.0

    name_score = 0.2 * tool_name_f1(gold_calls, pred_calls)
    arg_score = 0.3 * arg_overlap_f1(gold_calls, pred_calls, catalog)
    env_score = 0.1 if all_env_valid else 0.0
    cap = 0.6 if state_match else 0.45
    return min(cap, name_score + arg_score + env_score)


def create_lfm3_reward_func(
    catalog: dict[str, dict[str, Any]],
    tokenizer: Any | None = None,
    *,
    split: str = "train",
    root: str = "data",
    verify_env: bool = True,
) -> Callable[..., list[float]]:
    """Create a TRL GRPOTrainer-compatible reward function mirroring tool_call_reward in lfm3_grpo_with_trl.ipynb."""

    def tool_call_reward(
        prompts: list[Any],
        completions: list[Any] | None = None,
        completion_ids: list[Any] | None = None,
        calls: list[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        comp_items = completion_ids if completion_ids is not None else (completions or [])
        gold_col = calls or kwargs.get("gold_calls_list") or []
        sample_ids = kwargs.get("sample_id") or [None] * len(prompts)
        prefix_col = kwargs.get("prefix_calls") or ["[]"] * len(prompts)
        tools_col = kwargs.get("tools") or ["[]"] * len(prompts)
        gold_hash_col = kwargs.get("step_gold_state_hash") or kwargs.get("gold_state_hash") or [None] * len(prompts)
        terminal_col = kwargs.get("is_terminal") or [False] * len(prompts)

        rewards: list[float] = []
        for prompt, comp_item, gold_raw, sid, pref_raw, tools_raw, g_hash, is_term in zip(
            prompts, comp_items, gold_col, sample_ids, prefix_col, tools_col, gold_hash_col, terminal_col
        ):
            gold_calls = json.loads(gold_raw) if isinstance(gold_raw, str) else list(gold_raw or [])
            prefix_calls = json.loads(pref_raw) if isinstance(pref_raw, str) else list(pref_raw or [])
            step_tools = json.loads(tools_raw) if isinstance(tools_raw, str) else list(tools_raw or [])

            pred_calls, valid_fmt = extract_predicted_calls(
                comp_item,
                prompt=prompt if isinstance(prompt, str) else "",
                tools=step_tools,
                tokenizer=tokenizer,
                catalog=catalog,
            )

            env_ok = True
            state_ok = True
            check_hash = str(g_hash) if g_hash else None
            if verify_env and sid and (pred_calls or check_hash):
                env_ok, state_ok = verify_step_in_env(
                    str(sid),
                    split,
                    root,
                    prefix_calls,
                    pred_calls,
                    gold_state_hash=check_hash,
                )

            r = compute_step_reward(
                gold_calls,
                pred_calls,
                catalog,
                is_valid_format=valid_fmt,
                all_env_valid=env_ok,
                state_match=state_ok,
            )
            rewards.append(float(r))
        return rewards

    return tool_call_reward
