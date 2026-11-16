from __future__ import annotations

import json
from pathlib import Path

from src.agent.sft_dataset import build_sft_trajectory, group_calls_into_steps
from src.core.dataset import load_catalog, load_split


def test_group_calls_splits_read_before_write():
    catalog = load_catalog("train", "data")
    sample = next(
        s
        for s in load_split("train", "data")
        if [c.name for c in s.label.tool_calls] == ["smart_home.get_status", "smart_home.light_set"]
    )
    steps = group_calls_into_steps(sample.label.tool_calls, sample.label.tool_results, catalog)
    assert len(steps) == 2
    assert [c.name for c, _ in steps[0]] == ["smart_home.get_status"]
    assert [c.name for c, _ in steps[1]] == ["smart_home.light_set"]


def test_build_sft_trajectory_structure_and_leak_free():
    catalog = load_catalog("train", "data")
    dev_ids = set(json.loads(Path("configs/splits/dev_v1.json").read_text(encoding="utf-8"))["dev_ids"])
    train_only = [s for s in load_split("train", "data") if s.id not in dev_ids]
    assert len(train_only) == 1246

    act_sample = next(s for s in train_only if len(s.label.tool_calls) >= 2)
    traj = build_sft_trajectory(act_sample, catalog)
    assert traj["id"] not in dev_ids
    assert traj["messages"][0]["role"] == "system"
    assert any(m["role"] == "assistant" and "tool_calls" in m for m in traj["messages"])
    assert traj["messages"][-1]["role"] == "assistant"
    assert "tool_calls" not in traj["messages"][-1]

    ask_sample = next(
        s for s in train_only if any(c.name == "sys.ask_missing_info" for c in s.label.tool_calls)
    )
    ask_traj = build_sft_trajectory(ask_sample, catalog)
    assert ask_traj["messages"][-1]["role"] == "assistant"
    assert ask_traj["messages"][-1]["tool_calls"][0]["function"]["name"] == "sys.ask_missing_info"


def test_group_calls_keeps_independent_reads_parallel_but_splits_dependent_reads():
    from src.core.schemas import ToolCall

    cat = {
        "banking.get_balance": {"name": "banking.get_balance", "kind": "read", "parameters": {"properties": {}}},
        "contacts.search": {
            "name": "contacts.search",
            "kind": "read",
            "parameters": {"properties": {"query": {"type": "string"}}},
        },
        "chat_ott.list_chats": {"name": "chat_ott.list_chats", "kind": "read", "parameters": {"properties": {}}},
        "chat_ott.read_chat": {
            "name": "chat_ott.read_chat",
            "kind": "read",
            "parameters": {"properties": {"chat_id": {"type": "string", "x-ref": "chat_ott.chats"}}},
        },
    }

    # Independent reads: contacts.search(query="Hùng") does NOT depend on banking.get_balance()
    indep_calls = [
        ToolCall(name="banking.get_balance", arguments={}),
        ToolCall(name="contacts.search", arguments={"query": "Hùng"}),
    ]
    indep_results = [
        {"ok": True, "accounts": [{"id": "acc_01", "balance": 5000000}]},
        {"ok": True, "contacts": [{"id": "c_01", "name": "Hùng"}]},
    ]
    indep_steps = group_calls_into_steps(indep_calls, indep_results, cat)
    assert len(indep_steps) == 1
    assert [c.name for c, _ in indep_steps[0]] == ["banking.get_balance", "contacts.search"]

    # Dependent reads: chat_ott.read_chat(chat_id="chat_09") depends on chat_ott.list_chats() result
    dep_calls = [
        ToolCall(name="chat_ott.list_chats", arguments={}),
        ToolCall(name="chat_ott.read_chat", arguments={"chat_id": "chat_09"}),
    ]
    dep_results = [
        {"ok": True, "chats": [{"id": "chat_09", "name": "Nhóm Gia Đình"}]},
        {"ok": True, "messages": []},
    ]
    dep_steps = group_calls_into_steps(dep_calls, dep_results, cat)
    assert len(dep_steps) == 2
    assert [c.name for c, _ in dep_steps[0]] == ["chat_ott.list_chats"]
    assert [c.name for c, _ in dep_steps[1]] == ["chat_ott.read_chat"]

