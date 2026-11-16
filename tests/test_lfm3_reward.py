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


def test_validate_step_in_env_returns_false_on_exception_or_invalid_call():
    from src.agent.lfm3_reward import validate_step_in_env

    # Nonexistent sample_id raises inside SafeEnv; must NOT be swallowed as True!
    ok = validate_step_in_env(
        sample_id="nonexistent_sample_999999",
        split="train",
        root="data",
        prefix_calls=[],
        pred_calls=[{"name": "weather.get_forecast", "arguments": {"date": "2026-11-20"}}],
    )
    assert ok is False


def test_compute_step_reward_gates_on_state_match():
    cat = _catalog()
    gold = [{"name": "weather.get_forecast", "arguments": {"date": "2026-11-20"}}]
    pred_exact = [{"name": "weather.get_forecast", "arguments": {"date": "2026-11-20"}}]

    r_bad_state = compute_step_reward(
        gold,
        pred_exact,
        cat,
        is_valid_format=True,
        all_env_valid=True,
        state_match=False,
    )
    assert r_bad_state < 1.0
    assert compute_step_reward([], [], cat, is_valid_format=True, all_env_valid=True, state_match=False) == 0.0

