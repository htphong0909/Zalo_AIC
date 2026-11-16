from __future__ import annotations

from src.agent.history import build_messages, truncate_tool_result
from src.agent.prompts import build_system_prompt
from src.core.dataset import load_split
from src.core.schemas import Sample, ToolCall, Turn


def test_build_system_prompt_contains_context_and_rules():
    sample = load_split("train", "data")[0]
    prompt = build_system_prompt(sample)

    assert sample.context["current_time"] in prompt
    assert sample.context["timezone"] in prompt
    assert "sys.ask_missing_info" in prompt
    assert "installed_apps" in prompt or any(app in prompt for app in sample.context["installed_apps"])
    assert '<function name="..."><param name="...">...</param></function>' in prompt

    prompt_q35 = build_system_prompt(sample, prompt_style="qwen35")
    assert "<tool_call><function=tên_tool><parameter=tham_số>giá_trị</parameter></function></tool_call>" in prompt_q35
    assert '<function name="...">' not in prompt_q35


def test_build_messages_renders_history_and_current_turn():
    sample = next(s for s in load_split("train", "data") if s.history)
    msgs = build_messages(sample)

    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == sample.audio_text
    assert any(m["role"] == "tool" for m in msgs)


def test_build_messages_truncates_when_budget_is_small():
    turns = []
    for i in range(6):
        turns.append(Turn(role="user", audio_text=f"Câu hỏi lịch sử {i} " * 20))
        turns.append(
            Turn(
                role="assistant",
                text=f"Trả lời {i}",
                tool_calls=[ToolCall(name="alarm.list", arguments={})],
                tool_results=[{"ok": True, "alarms": [{"id": f"a_{j}", "blob": "x" * 200} for j in range(10)]}],
            )
        )
    sample = Sample(
        id="test_trunc",
        audio="audio/test.wav",
        audio_text="Câu hỏi hiện tại",
        context={
            "current_time": "2026-11-13T12:00:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
            "location": {"lat": 10.0, "lng": 106.0},
            "user_profile": {"name": "Khoa", "city": "TP.HCM"},
            "installed_apps": {"alarm": {"tools": ["alarm.list"]}},
        },
        history=turns,
    )
    msgs = build_messages(sample, max_prompt_tokens=900)
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == "Câu hỏi hiện tại"
    assert len(msgs) < 2 + len(turns) + 6
    assert len(truncate_tool_result({"data": "a" * 5000}, max_chars=400)) <= 420
