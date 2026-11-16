"""Unit tests for BoundaryVerifier."""
from src.agent.boundary_verifier import BoundaryVerifier
from src.core.schemas import ToolCall


def test_boundary_verifier_removes_hallucinated_optional_arg():
    verifier = BoundaryVerifier()

    # User never mentioned repeat
    call_with_repeat = ToolCall(name="alarm.create", arguments={"time": "06:30", "repeat": "daily"})
    verified = verifier.verify_optional_args(call_with_repeat, user_utterance="Đặt báo thức 6 rưỡi sáng mai giúp anh")
    assert "repeat" not in verified.arguments
    assert verified.arguments == {"time": "06:30"}

    # User explicitly mentioned repeat
    verified_keep = verifier.verify_optional_args(call_with_repeat, user_utterance="Đặt báo thức 6 rưỡi lặp lại mỗi ngày")
    assert "repeat" in verified_keep.arguments


def test_boundary_verifier_detects_ambiguous_duplicate_entities():
    verifier = BoundaryVerifier()
    entities = [
        {"id": "c_1", "name": "Nguyễn Văn Hùng", "phone": "0912345678"},
        {"id": "c_2", "name": "Nguyễn Văn Hùng", "phone": "0987654321"},
    ]
    should_ask = verifier.should_ask_clarification(
        user_utterance="Gọi cho anh Hùng",
        retrieved_entities=entities,
        target_tool="phone.call",
        required_id_param="contact_id",
    )
    assert should_ask is True


def test_boundary_verifier_find_missing_prereq_read():
    verifier = BoundaryVerifier()

    ask_note = ToolCall(name="sys.ask_missing_info", arguments={"tool_name": "notes.append", "params": ["note_id"]})
    prereq = verifier.find_missing_prereq_read(
        ask_note,
        seen_reads=set(),
        available_tools={"notes.list", "notes.append"},
    )
    assert prereq == ToolCall(name="notes.list", arguments={})

    # Already read -> no duplicate prereq needed
    assert (
        verifier.find_missing_prereq_read(
            ask_note,
            seen_reads={("notes.list", "{}")},
            available_tools={"notes.list", "notes.append"},
        )
        is None
    )

    ask_scene = ToolCall(
        name="sys.ask_missing_info", arguments={"tool_name": "smart_home.scene_run", "params": ["scene_id"]}
    )
    assert verifier.find_missing_prereq_read(
        ask_scene,
        seen_reads=set(),
        available_tools={"smart_home.get_status", "smart_home.scene_run"},
    ) == ToolCall(name="smart_home.get_status", arguments={})

    # Direct phone ask stays intact (contacts.search not blindly forced when no target name)
    ask_phone = ToolCall(name="sys.ask_missing_info", arguments={"tool_name": "phone.call", "params": ["contact_id"]})
    assert (
        verifier.find_missing_prereq_read(
            ask_phone,
            seen_reads=set(),
            available_tools={"contacts.search", "phone.call"},
        )
        is None
    )


def test_boundary_verifier_lottery_today_default_date_stripping():
    verifier = BoundaryVerifier()

    call = ToolCall(name="lottery.get_results", arguments={"province": "TP.HCM", "date": "2026-12-06"})
    stripped = verifier.verify_optional_args(
        call,
        user_utterance="Dò kết quả xổ số TP.HCM chiều nay giúp anh",
        current_date="2026-12-06",
    )
    assert stripped.arguments == {"province": "TP.HCM"}

    kept = verifier.verify_optional_args(
        ToolCall(name="lottery.get_results", arguments={"province": "TP.HCM", "date": "2026-12-05"}),
        user_utterance="Xem kết quả xổ số TP.HCM hôm qua ngày 2026-12-05",
        current_date="2026-12-06",
    )
    assert kept.arguments == {"province": "TP.HCM", "date": "2026-12-05"}
