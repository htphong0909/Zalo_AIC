"""Test that sanitize_tool_call satisfies the Zero-Gold-Corruption Invariant on 100% of train_only gold labels."""
import json
from pathlib import Path
from src.core.dataset import load_split
from src.agent.tool_caller import sanitize_tool_call


def test_zero_gold_corruption_on_all_train_only():
    dev_ids = set(json.loads(Path("configs/splits/dev_v1.json").read_text(encoding="utf-8"))["dev_ids"])
    train_only = [s for s in load_split("train", "data") if s.id not in dev_ids]

    total_gold_calls = 0
    corrupted_calls = []

    for sample in train_only:
        if not sample.label or not sample.label.tool_calls:
            continue
        for gold_call in sample.label.tool_calls:
            total_gold_calls += 1
            sanitized = sanitize_tool_call(gold_call, sample)
            if sanitized.name != gold_call.name or sanitized.arguments != gold_call.arguments:
                corrupted_calls.append({
                    "sample_id": sample.id,
                    "tool": gold_call.name,
                    "gold": gold_call.arguments,
                    "sanitized": sanitized.arguments,
                })

    assert total_gold_calls == 2854, f"Expected 2854 gold calls in train_only, got {total_gold_calls}"
    assert len(corrupted_calls) == 0, (
        f"Critical regression: sanitize_tool_call corrupted {len(corrupted_calls)} gold calls! "
        f"First few corruptions: {corrupted_calls[:5]}"
    )
