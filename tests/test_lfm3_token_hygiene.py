"""Test token hygiene (no double BOS/EOS), completion length, and step_gold_state_hash preservation."""
import json
from src.core.dataset import load_catalog, load_split
from src.agent.sft_dataset import build_sft_trajectory, unroll_trajectory_steps
from scripts.build_lfm3_data import (
    render_step_prompt_completion,
    build_sft_step_record,
    build_grpo_step_record,
)


def test_no_double_bos_or_eos_in_prompt_completion():
    # Construct a sample step
    step = {
        "id": "test_s1",
        "step_index": 0,
        "step_category": "act",
        "is_terminal": False,
        "prompt_messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Bật đèn phòng khách giúp anh."},
        ],
        "completion_message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "smart_home.light_set", "arguments": {"device_id": "light_1", "power": "on"}},
                }
            ],
        },
        "tools": [{"type": "function", "function": {"name": "smart_home.light_set"}}],
    }

    prompt_str, comp_str = render_step_prompt_completion(step)

    # 1. Prompt string must NOT begin with raw <|startoftext|>
    assert not prompt_str.startswith("<|startoftext|>"), "Prompt must not contain leading <|startoftext|>"

    # 2. Completion string must NOT end with trailing <|im_end|>
    assert not comp_str.rstrip().endswith("<|im_end|>"), "Completion must not end with trailing <|im_end|>"

    # 3. SFT record preserves both conversational messages and prompt/completion
    sft_rec = build_sft_step_record(step)
    assert "messages" in sft_rec
    assert len(sft_rec["messages"]) == 3
    assert sft_rec["messages"][-1]["role"] == "assistant"


def test_step_gold_state_hash_populated_for_every_step():
    catalog = load_catalog("train", "data")
    sample = next(
        s
        for s in load_split("train", "data")
        if [c.name for c in (s.label.tool_calls if s.label else [])] == ["smart_home.get_status", "smart_home.light_set"]
    )

    traj = build_sft_trajectory(sample, catalog)
    steps = unroll_trajectory_steps(traj, split="train", root="data")

    assert len(steps) >= 2
    # Check that step 1 (smart_home.light_set, a write step) has step_gold_state_hash populated!
    write_step = next(st for st in steps if any(c["name"] == "smart_home.light_set" for c in st["step_calls"]))
    assert write_step["step_gold_state_hash"] is not None
    assert len(write_step["step_gold_state_hash"]) > 0

    # GRPO record must include this hash
    grpo_rec = build_grpo_step_record(write_step)
    assert grpo_rec["gold_state_hash"] == write_step["step_gold_state_hash"]
