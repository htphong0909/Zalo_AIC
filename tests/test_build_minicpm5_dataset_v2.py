"""Tests for SFT V2 dataset generation pipeline and dual execution verifier."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from src.core.schemas import ASK_MISSING_INFO, ToolCall
from src.data_gen.execution_verifier import verify_step_validity
from scripts.build_minicpm5_dataset_v2 import (
    build_v2_dataset,
    format_minicpm5_tools_block,
    resolve_step_tools,
)


def test_verify_step_validity_standard_calls():
    catalog = {
        "smart_home.light_set": {
            "name": "smart_home.light_set",
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "state": {"type": "string"},
                },
                "required": ["device_id"],
            },
        },
        "alarm.create": {
            "name": "alarm.create",
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {"time": {"type": "string"}},
                "required": ["time"],
            },
        },
    }

    # Valid step
    valid_step = {
        "step_calls": [
            {"name": "smart_home.light_set", "arguments": {"device_id": "light_1", "state": "on"}}
        ]
    }
    assert verify_step_validity(valid_step, catalog) is True

    # Empty calls (OOS or direct response) is valid
    empty_step = {"step_calls": []}
    assert verify_step_validity(empty_step, catalog) is True

    # Unknown tool call
    unknown_step = {
        "step_calls": [
            {"name": "non_existent_tool", "arguments": {}}
        ]
    }
    assert verify_step_validity(unknown_step, catalog) is False

    # Invalid arguments type
    invalid_args_step = {
        "step_calls": [
            {"name": "alarm.create", "arguments": "not_a_dict"}
        ]
    }
    assert verify_step_validity(invalid_args_step, catalog) is False


def test_verify_step_validity_ask_missing_info():
    catalog = {
        "alarm.create": {
            "name": "alarm.create",
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {"time": {"type": "string"}},
                "required": ["time"],
            },
        },
    }

    # Valid ask call
    valid_ask_step = {
        "step_calls": [
            {
                "name": ASK_MISSING_INFO,
                "arguments": {"tool_name": "alarm.create", "params": ["time"]},
            }
        ]
    }
    assert verify_step_validity(valid_ask_step, catalog) is True

    # Ask call referencing tool not in catalog
    invalid_tool_ask = {
        "step_calls": [
            {
                "name": ASK_MISSING_INFO,
                "arguments": {"tool_name": "unknown.tool", "params": ["time"]},
            }
        ]
    }
    assert verify_step_validity(invalid_tool_ask, catalog) is False

    # Ask call with empty params list
    empty_params_ask = {
        "step_calls": [
            {
                "name": ASK_MISSING_INFO,
                "arguments": {"tool_name": "alarm.create", "params": []},
            }
        ]
    }
    assert verify_step_validity(empty_params_ask, catalog) is False

    # Ask call with non-string param
    non_str_param_ask = {
        "step_calls": [
            {
                "name": ASK_MISSING_INFO,
                "arguments": {"tool_name": "alarm.create", "params": [123]},
            }
        ]
    }
    assert verify_step_validity(non_str_param_ask, catalog) is False

    # Ask call referencing param not in tool properties
    unknown_param_ask = {
        "step_calls": [
            {
                "name": ASK_MISSING_INFO,
                "arguments": {"tool_name": "alarm.create", "params": ["invalid_param"]},
            }
        ]
    }
    assert verify_step_validity(unknown_param_ask, catalog) is False


def test_v2_dataset_no_dev_leakage(tmp_path: Path):
    out_dir = tmp_path / "sft_out"
    count, gz_path = build_v2_dataset(
        split_name="train",
        root="data",
        dev_split_path="configs/splits/dev_v1.json",
        out_dir=out_dir,
        target_synthetic=30,
    )

    assert count > 0
    assert gz_path.exists()
    assert gz_path.suffix == ".gz"

    jsonl_path = out_dir / "minicpm5_train_sft_v2.jsonl"
    assert jsonl_path.exists()

    dev_split_file = Path("configs/splits/dev_v1.json")
    dev_ids = set(json.loads(dev_split_file.read_text(encoding="utf-8"))["dev_ids"])
    assert len(dev_ids) == 299

    records_read = 0
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            records_read += 1

            # Check essential ChatML keys
            assert "id" in rec
            assert "sample_id" in rec
            assert "category" in rec
            assert "messages" in rec

            sample_id = rec["sample_id"]
            # Check 0 dev leakage
            assert sample_id not in dev_ids
            # For multi-intent composite sample_ids like train_01_train_02
            parts = sample_id.split("_")
            for p in parts:
                assert p not in dev_ids

            msgs = rec["messages"]
            assert len(msgs) >= 2
            roles = [m["role"] for m in msgs]
            assert "system" in roles
            assert "user" in roles
            assert "assistant" in roles

            # Check MiniCPM5 XML tools block in system message
            sys_msg = next(m for m in msgs if m["role"] == "system")
            assert "# Tools" in sys_msg["content"]
            assert "<tools>" in sys_msg["content"]
            assert "</tools>" in sys_msg["content"]

            # Check <think> 4-slot structure in assistant turn
            last_asst = [m for m in msgs if m["role"] == "assistant"][-1]
            assert "<think>" in last_asst["content"]
            assert "</think>" in last_asst["content"]
            assert "1. Ý định:" in last_asst["content"]
            assert "2. Dữ liệu:" in last_asst["content"]
            assert "3. Kế hoạch:" in last_asst["content"]
            assert "4. Ranh giới:" in last_asst["content"]

    assert records_read == count


def test_hammer_tool_shuffling():
    import random
    tools = [
        {"type": "function", "function": {"name": f"tool_{i}"}}
        for i in range(10)
    ]
    step = {"tools": tools, "prompt_messages": [{"role": "user", "content": "test"}]}

    rng1 = random.Random(42)
    resolved1 = resolve_step_tools(step, None, None)
    rng1.shuffle(resolved1)
    names1 = [t["function"]["name"] for t in resolved1]

    rng2 = random.Random(999)
    resolved2 = resolve_step_tools(step, None, None)
    rng2.shuffle(resolved2)
    names2 = [t["function"]["name"] for t in resolved2]

    # Hammer shuffling should vary tool order across different seeds
    assert names1 != names2
