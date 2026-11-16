import json
import pytest
from src.core.schemas import Label, Sample, ToolCall, Turn
from src.agent.sft_dataset import build_sft_trajectory, unroll_trajectory_steps
from scripts.build_lfm3_data import (
    build_grpo_step_record,
    build_sft_step_record,
    format_lfm3_completion_xml,
    oversample_grpo_records,
    simulate_ctc_transcript,
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


def test_simulate_ctc_transcript_converts_digits_and_punctuation():
    raw = "Đặt báo thức 5h30 sáng mai, chuyển 2,8 triệu và nhắn SMS nhé!"
    sim = simulate_ctc_transcript(raw)
    assert sim == sim.lower()
    assert "," not in sim and "!" not in sim
    assert "năm rưỡi" in sim
    assert "hai triệu tám" in sim
    assert "a ét mờ ét" in sim


def test_fallback_render_prompt_includes_tools():
    catalog = _mock_catalog()
    sample = Sample(
        id="train_9003",
        audio="audio/train_9003.wav",
        audio_text="Tìm liên hệ Bố",
        context={"installed_apps": {"contacts": {"tools": ["contacts.search"]}}},
        label=Label(
            tool_calls=[ToolCall(name="contacts.search", arguments={"query": "Bố"})],
            tool_results=[{"ok": True}],
        ),
    )
    traj = build_sft_trajectory(sample, catalog)
    steps = unroll_trajectory_steps(traj)
    sft_rec = build_sft_step_record(steps[0], tokenizer=None)
    assert "Tìm liên hệ" in sft_rec["prompt"]
    assert "parameters" in sft_rec["prompt"]


def test_filter_training_tools_preserves_history_apps_and_oos_apps():
    import random
    from scripts.build_lfm3_data import _filter_training_tools
    from src.agent.tool_selector import compact_tool_spec

    catalog = {
        **_mock_catalog(),
        "weather.get_forecast": {"name": "weather.get_forecast", "parameters": {"properties": {}}},
        "alarm.create": {"name": "alarm.create", "parameters": {"properties": {}}},
        "banking.get_balance": {"name": "banking.get_balance", "parameters": {"properties": {}}},
        "calendar.list_events": {"name": "calendar.list_events", "parameters": {"properties": {}}},
        "notes.list_notes": {"name": "notes.list_notes", "parameters": {"properties": {}}},
    }
    full_tools = [compact_tool_spec(v) for v in catalog.values()]

    # OOS sample with history calling weather.get_forecast must keep ALL installed apps (not prune to 3!)
    oos_sample = Sample(
        id="train_9004",
        audio="audio/train_9004.wav",
        audio_text="Đặt vé máy bay đi Đà Nẵng",
        context={
            "installed_apps": {
                "contacts": {},
                "sms": {},
                "weather": {},
                "alarm": {},
                "banking": {},
                "calendar": {},
                "notes": {},
            }
        },
        history=[
            Turn(
                role="assistant",
                tool_calls=[ToolCall(name="weather.get_forecast", arguments={})],
                tool_results=[{"ok": True}],
            )
        ],
        label=Label(tool_calls=[], tool_results=[]),
    )
    kept = _filter_training_tools(oos_sample, full_tools, max_distractor_apps=3, rng=random.Random(42))
    kept_names = {t["function"]["name"] for t in kept}
    assert "weather.get_forecast" in kept_names
    assert len(kept_names) == len(full_tools)


def test_build_dpo_pairs_and_hammer_masking():
    import random
    from scripts.build_lfm3_data import apply_hammer_masking, build_dpo_pairs_for_step

    catalog = _mock_catalog()
    act_sample = Sample(
        id="train_9005",
        audio="audio/train_9005.wav",
        audio_text="Gửi tin nhắn cho c_01 nội dung xin chào",
        context={
            "installed_apps": {
                "contacts": {"tools": ["contacts.search"]},
                "sms": {"tools": ["sms.send"]},
            }
        },
        label=Label(
            tool_calls=[ToolCall(name="sms.send", arguments={"contact_id": "c_01", "content": "xin chào"})],
            tool_results=[{"ok": True}],
        ),
    )
    traj = build_sft_trajectory(act_sample, catalog)
    steps = unroll_trajectory_steps(traj)

    # 1. DPO pairs for act step (chosen = sms.send, rejected = false sys.ask_missing_info)
    act_pairs = build_dpo_pairs_for_step(steps[0], tokenizer=None, rng=random.Random(7))
    assert len(act_pairs) >= 1
    assert "sms.send" in act_pairs[0]["chosen"]
    assert "sys.ask_missing_info" in act_pairs[0]["rejected"]

    # 2. DPO pairs for final_stop step (chosen = text stop, rejected = hallucinated tool_call)
    stop_pairs = build_dpo_pairs_for_step(steps[1], tokenizer=None, rng=random.Random(7))
    assert len(stop_pairs) >= 1
    assert "<tool_call>" not in stop_pairs[0]["chosen"]
    assert "<tool_call>" in stop_pairs[0]["rejected"]

    # 3. Hammer masking remaps function and parameter names consistently across tools & completion
    masked_step = apply_hammer_masking(steps[0], rng=random.Random(11))
    masked_tool_names = {t["function"]["name"] for t in masked_step["tools"]}
    assert "sms.send" not in masked_tool_names
    masked_call = masked_step["step_calls"][0]
    assert masked_call["name"] in masked_tool_names
    assert set(masked_call["arguments"].keys()) != {"contact_id", "content"}
    assert len(masked_call["arguments"]) == 2


