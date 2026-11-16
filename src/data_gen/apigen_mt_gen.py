"""APIGen-MT (NeurIPS 2025) Multi-Intent Chaining & Horizon-Aware Generator.

Adapted from Salesforce APIGen-MT (NeurIPS 2025) and NVIDIA Tool-N1 (ICLR 2026):
references/tool_calling_repos/06_xLAM_APIGen-MT_NeurIPS2025
references/tool_calling_repos/01_Tool-N1_ICLR2026/data_process/distill_data.py

Combines compatible independent actions into verified multi-intent multi-turn trajectories
with natural Vietnamese connectors (" đồng thời ", ", rồi ", " và ", ", với lại ").
Enforces horizon-aware <think> reasoning traces with explicit remaining sub-goals to prevent
premature stopping and under-generation.
"""
from __future__ import annotations

import copy
import json
import random
from typing import Any

from src.core.schemas import ASK_MISSING_INFO, Label, Sample, ToolCall
from src.agent.sft_dataset import group_calls_into_steps
from src.core.schemas import ASK_MISSING_INFO, Sample, ToolCall
from src.data_gen.distill_think import synthesize_deterministic_think
from src.data_gen.prompt_utils import (
    build_assistant_history_message,
    build_sample_prefix_messages,
    call_to_dict,
    serialize_observation,
)

# Natural Vietnamese conversational multi-intent connectors
_CONNECTORS = [" đồng thời ", ", rồi ", " và ", ", với lại "]


def render_minicpm5_function_call_xml(name: str, args: dict[str, Any]) -> str:
    """Render a tool call into MiniCPM5 native XML: <function name="..."><param name="...">...</param></function>."""
    if not args:
        return f'<function name="{name}"></function>'
    params_str = "".join(f'<param name="{k}">{v}</param>' for k, v in args.items())
    return f'<function name="{name}">{params_str}</function>'


def _extract_tool_calls(sample: Sample) -> list[dict[str, Any]]:
    """Extract and normalize valid tool calls from a sample, excluding ASK_MISSING_INFO and OOS."""
    raw_calls: list[Any] = []
    if isinstance(sample.label, Label):
        raw_calls = sample.label.tool_calls
    elif isinstance(sample.label, dict):
        raw_calls = sample.label.get("tool_calls", [])

    valid_calls: list[dict[str, Any]] = []
    for c in raw_calls:
        if isinstance(c, dict):
            name = c.get("name") or ""
            args = dict(c.get("arguments") or {})
        elif hasattr(c, "name"):
            name = getattr(c, "name", "")
            args = dict(getattr(c, "arguments", {}) or {})
        else:
            continue

        if not name or name == ASK_MISSING_INFO:
            continue
        valid_calls.append({"name": name, "arguments": args})

    return valid_calls


def generate_apigen_multi_intent_samples(
    train_samples: list[Sample],
    catalog: dict[str, Any],
    count: int = 600,
    seed: int = 42,
    include_step_1: bool = False,
) -> list[dict[str, Any]]:
    """Synthesize multi-intent chaining trajectories from pairs of single-intent train samples.

    Formulation:
    - Sample pairs of compatible single-intent train samples (s1, s2).
    - Join user utterances using natural Vietnamese connectors (' đồng thời ', ', rồi ', ' và ', ', với lại ').
    - Create Step 0 which calls Intent 1's tools and explicitly lists Intent 2's tools in `remaining_subgoals`.
    - Synthesize horizon-aware 4-slot <think> trace with Slot 4 showing:
      "Quyết định: Đủ điều kiện thực thi (Act). Mục tiêu còn lại: [{subgoals}]. Chưa kết thúc phiên."
    - Optionally (when include_step_1=True), generate Step 1 where Intent 1 has executed and returned observation,
      and Intent 2's tools are called with `remaining_subgoals: []`.
    """
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []

    valid_samples: list[Sample] = []
    for s in train_samples:
        if not s.audio_text or not s.audio_text.strip():
            continue
        calls = _extract_tool_calls(s)
        if calls:
            valid_samples.append(s)

    if len(valid_samples) < 2:
        return results

    seen_pairs: set[tuple[str, str]] = set()
    max_possible_pairs = len(valid_samples) * (len(valid_samples) - 1)
    max_attempts = max(count * 20, 200)
    attempts = 0

    while len(results) < count and attempts < max_attempts:
        attempts += 1
        s1, s2 = rng.sample(valid_samples, 2)
        pair_key = (s1.id, s2.id)

        # Ensure pair uniqueness if not exhausted
        if len(seen_pairs) < max_possible_pairs:
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

        c1_list = _extract_tool_calls(s1)
        c2_list = _extract_tool_calls(s2)
        if not c1_list or not c2_list:
            continue

        conn = rng.choice(_CONNECTORS)
        t1 = (s1.audio_text or "").rstrip(".,!? \t\n")
        t2 = (s2.audio_text or "").strip()
        if t2 and t2[0].isupper() and (len(t2) == 1 or not t2[1].isupper()):
            t2 = t2[0].lower() + t2[1:]
        combined_text = f"{t1}{conn}{t2}"

        tool_names_2 = [c["name"] for c in c2_list]

        # Combine installed apps
        apps1 = set((s1.context or {}).get("installed_apps", {}).keys())
        apps2 = set((s2.context or {}).get("installed_apps", {}).keys())
        combined_apps = sorted(apps1 | apps2)
        if not combined_apps:
            inferred = set()
            for c in c1_list + c2_list:
                n = c["name"]
                if "." in n:
                    inferred.add(n.split(".")[0])
                elif n:
                    inferred.add(n)
            combined_apps = sorted(inferred)

        # Merge contexts
        merged_context = copy.deepcopy(s1.context or {})
        if "installed_apps" in merged_context and isinstance(merged_context["installed_apps"], dict):
            s2_apps = (s2.context or {}).get("installed_apps", {})
            if isinstance(s2_apps, dict):
                merged_context["installed_apps"].update(s2_apps)

        # Group dependent read->write calls cleanly so no single step mixes a read and a dependent write
        s1_res = list(getattr(s1.label, "tool_results", []) or []) if not isinstance(s1.label, dict) else list(s1.label.get("tool_results") or [])
        s2_res = list(getattr(s2.label, "tool_results", []) or []) if not isinstance(s2.label, dict) else list(s2.label.get("tool_results") or [])
        steps_1 = group_calls_into_steps([ToolCall(**c) for c in c1_list], s1_res, catalog)
        steps_2 = group_calls_into_steps([ToolCall(**c) for c in c2_list], s2_res, catalog)
        c1_step0 = [call_to_dict(c) for c, _ in steps_1[0]] if steps_1 else c1_list
        c1_rest = [call_to_dict(c) for grp in steps_1[1:] for c, _ in grp]
        c2_step0 = [call_to_dict(c) for c, _ in steps_2[0]] if steps_2 else c2_list
        c2_rest_names = [c.name for grp in steps_2[1:] for c, _ in grp]

        merged_sample = Sample(
            id=f"apigen_{s1.id}_{s2.id}",
            context=merged_context,
            history=list(getattr(s1, "history", []) or []),
            audio_text=combined_text,
        )

        # Step 0: User multi-intent request -> call Intent 1 Step-0 tools cleanly, keep remaining in subgoals
        prompt_msgs_s0 = build_sample_prefix_messages(merged_sample, catalog)

        step_0: dict[str, Any] = {
            "id": f"apigen_{s1.id}_{s2.id}_s0",
            "sample_id": f"{s1.id}_{s2.id}",
            "step_index": 0,
            "step_category": "act",
            "prompt_messages": prompt_msgs_s0,
            "step_calls": c1_step0,
            "completion_message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for c in c1_step0
                ],
            },
            "remaining_subgoals": [c["name"] for c in c1_rest] + tool_names_2,
            "tools": combined_apps,
            "sample_context": merged_context,
        }
        step_0["think"] = synthesize_deterministic_think(step_0, merged_context, catalog)
        results.append(step_0)

        # Optional Step 1: Tool 1 observation received -> execute Step 0 of Intent 2 (never mixing read+write)
        if include_step_1 and not c1_rest and len(results) < count:
            prompt_msgs_s1 = build_sample_prefix_messages(merged_sample, catalog)
            prompt_msgs_s1.append(build_assistant_history_message(c1_step0))
            for idx_c, c in enumerate(c1_step0):
                obs = s1_res[idx_c] if idx_c < len(s1_res) else {"ok": True}
                prompt_msgs_s1.append({
                    "role": "tool",
                    "name": c["name"],
                    "content": serialize_observation(obs),
                })

            step_1: dict[str, Any] = {
                "id": f"apigen_{s1.id}_{s2.id}_s1",
                "sample_id": f"{s1.id}_{s2.id}",
                "step_index": 1,
                "step_category": "act",
                "prompt_messages": prompt_msgs_s1,
                "step_calls": c2_step0,
                "completion_message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]},
                        }
                        for c in c2_step0
                    ],
                },
                "remaining_subgoals": c2_rest_names,
                "tools": combined_apps,
                "sample_context": merged_context,
            }
            step_1["think"] = synthesize_deterministic_think(step_1, merged_context, catalog)
            results.append(step_1)

    return results[:count]
