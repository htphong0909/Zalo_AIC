"""ToolSandbox (NAACL 2025) Milestone Anchor Binding Generator.

Adapted from Apple ML ToolSandbox (NAACL 2025) milestone dependency formulation:
references/tool_calling_repos/09_ToolSandbox_NAACL2025/tool_sandbox/
Generates multi-turn Read -> Write steps where Write parameters (like contact_id,
bill_id, event_id, device_id, etc.) are dynamically bound from prior Read observations.
Integrates with synthesize_deterministic_think to generate a 4-slot <think> trace
with Slot 2 grounding directly on the tool observation.
"""
from __future__ import annotations

import copy
import json
import random
from typing import Any

from src.core.schemas import ASK_MISSING_INFO, Sample, ToolCall
from src.data_gen.distill_think import synthesize_deterministic_think
from src.data_gen.prompt_utils import (
    build_assistant_history_message,
    build_sample_prefix_messages,
    call_to_dict,
    resolve_installed_apps,
    serialize_observation,
)

# Set of standard entity ID parameter names for tool-calling domains
_ID_PARAM_KEYS = frozenset(
    {
        "device_id",
        "contact_id",
        "event_id",
        "alarm_id",
        "bill_id",
        "beneficiary_id",
        "from_account_id",
        "restaurant_id",
        "item_id",
        "track_id",
        "chat_id",
        "course_id",
        "note_id",
        "appliance_id",
        "bike_id",
        "subscription_id",
        "reminder_id",
        "flight_id",
        "order_id",
        "booking_id",
        "channel",
        "id",
    }
)


def render_minicpm5_function_call_xml(name: str, args: dict[str, Any]) -> str:
    """Render a tool call into MiniCPM5 native XML: <function name="..."><param name="...">...</param></function>."""
    param_parts: list[str] = []
    for k, v in (args or {}).items():
        if isinstance(v, (dict, list, bool)):
            val_str = json.dumps(v, ensure_ascii=False)
        else:
            val_str = str(v)
        param_parts.append(f'<param name="{k}">{val_str}</param>')
    inner = "".join(param_parts)
    return f'<function name="{name}">{inner}</function>'


def _collect_scalar_values(obj: Any, out: set[str]) -> None:
    """Recursively collect scalar strings, numbers, etc. from nested tool result structures."""
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_scalar_values(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_scalar_values(item, out)
    elif isinstance(obj, (str, int, float)):
        val_str = str(obj).strip()
        if len(val_str) >= 2:
            out.add(val_str)


def _is_read_tool(name: str, catalog: dict[str, Any]) -> bool:
    """Determine whether tool is an information-gathering/lookup read tool."""
    kind = str((catalog.get(name) or {}).get("kind") or "")
    if kind == "read":
        return True
    return name.endswith((".search", ".list", ".get", ".get_status", ".guide"))


def _is_bound_anchor(
    write_name: str,
    write_args: dict[str, Any],
    read_res: Any,
    catalog: dict[str, Any],
) -> bool:
    """Verify that write tool consumes an entity ID or dynamic value anchored in read observation."""
    if not write_name or write_name == ASK_MISSING_INFO:
        return False

    w_kind = str((catalog.get(write_name) or {}).get("kind") or "")
    if w_kind == "read":
        return False

    obs_scalars: set[str] = set()
    _collect_scalar_values(read_res, obs_scalars)

    for k, v in (write_args or {}).items():
        # Direct value grounding: argument value is present in read observation scalars
        if isinstance(v, (str, int, float)) and str(v).strip() in obs_scalars:
            return True
        k_low = k.lower()
        if k_low.endswith("_id") or k_low == "id" or k_low in _ID_PARAM_KEYS:
            return True

    return False


def generate_anchor_binding_samples(
    train_samples: list[Sample],
    catalog: dict[str, Any],
    count: int = 400,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Synthesize multi-turn trajectory steps binding observation IDs to subsequent action tools.

    Formulation:
    - User message: original query
    - Assistant turn (Step 0): native XML function call for Read tool
    - Tool turn: JSON observation of Read tool results
    - Target assistant turn (Step 1): Write tool call dynamically consuming bound ID
    - Reasoning: 4-slot micro-<think> trace with Slot 2 grounding on the observation
    """
    rng = random.Random(seed)
    candidates: list[tuple[Sample, int, int, list[dict[str, Any]], list[Any], tuple[str, dict[str, Any]], Any]] = []

    for s in train_samples:
        if s.label is None:
            continue
        if isinstance(s.label, dict):
            calls = s.label.get("tool_calls", [])
            results_list = s.label.get("tool_results", [])
        elif hasattr(s.label, "tool_calls"):
            calls = getattr(s.label, "tool_calls", [])
            results_list = getattr(s.label, "tool_results", [])
        else:
            calls = []
            results_list = []

        if len(calls) < 2 or len(results_list) < 1:
            continue

        norm_calls = [call_to_dict(c) for c in calls]
        # Scan all Read -> Write dependency edges across the entire trajectory
        for j in range(1, len(norm_calls)):
            wc = norm_calls[j]
            w_name, w_args = wc["name"], wc["arguments"]
            for i in range(j):
                if i >= len(results_list):
                    break
                rc = norm_calls[i]
                r_name = rc["name"]
                read_res = results_list[i]
                if not _is_read_tool(r_name, catalog):
                    continue
                if _is_bound_anchor(w_name, w_args, read_res, catalog):
                    candidates.append((s, i, j, norm_calls, list(results_list), (w_name, w_args), read_res))
                    break

    rng.shuffle(candidates)

    results: list[dict[str, Any]] = []
    for s, _i_read, j_write, norm_calls, results_list, (w_name, w_args), _read_res in candidates:
        if len(results) >= count:
            break

        prompt_msgs = build_sample_prefix_messages(s, catalog)
        for k in range(j_write):
            prior_c = norm_calls[k]
            # Keep compact json_params=True on prior assistant XML so complex filters remain valid JSON
            prompt_msgs.append(build_assistant_history_message([prior_c], json_params=True))
            obs = results_list[k] if k < len(results_list) else {"ok": True}
            prompt_msgs.append({
                "role": "tool",
                "name": prior_c["name"],
                "content": serialize_observation(obs),
            })

        installed = resolve_installed_apps(s, [c["name"] for c in norm_calls])

        step_dict: dict[str, Any] = {
            "id": f"anchor_{s.id}_s{j_write}",
            "sample_id": s.id,
            "step_index": j_write,
            "step_category": "act",
            "prompt_messages": prompt_msgs,
            "step_calls": [{"name": w_name, "arguments": w_args}],
            "completion_message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": w_name, "arguments": w_args},
                    }
                ],
            },
            "tools": installed,
            "sample_context": s.context,
        }

        # Synthesize verified 4-slot <think> trace with Slot 2 grounding on the observation
        think_trace = synthesize_deterministic_think(step_dict, s.context, catalog)
        step_dict["think"] = think_trace

        results.append(step_dict)

    return results
