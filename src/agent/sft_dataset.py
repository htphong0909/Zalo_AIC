"""Build leak-free multi-step ReAct SFT trajectories from train samples (excluding dev split)."""
from __future__ import annotations

from typing import Any

from src.agent.history import build_messages, truncate_tool_result
from src.agent.tool_selector import select_tools
from src.core.schemas import ASK_MISSING_INFO, Sample, ToolCall
from src.eval.splits import stratum


def _collect_scalar_values(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_scalar_values(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_scalar_values(item, out)
    elif isinstance(obj, str) and len(obj.strip()) >= 2:
        out.add(obj.strip())


def _read_depends_on_previous(
    call: ToolCall,
    current_step: list[tuple[ToolCall, Any]],
    catalog: dict[str, dict[str, Any]],
) -> bool:
    if not call.arguments or not current_step:
        return False
    props = ((catalog.get(call.name) or {}).get("parameters") or {}).get("properties") or {}
    prior_values: set[str] = set()
    for _, prev_res in current_step:
        _collect_scalar_values(prev_res, prior_values)

    for k, v in call.arguments.items():
        if (props.get(k) or {}).get("x-ref"):
            return True
        if isinstance(v, str) and v.strip() in prior_values:
            return True
    return False


def group_calls_into_steps(
    calls: list[ToolCall],
    results: list[Any],
    catalog: dict[str, dict[str, Any]],
) -> list[list[tuple[ToolCall, Any]]]:
    """Partition a sample's gold tool_calls into sequential ReAct turns (parallel independent reads -> dependent read -> write/meta)."""
    if not calls:
        return []

    padded_results = list(results) + [{"ok": True}] * max(0, len(calls) - len(results))
    steps: list[list[tuple[ToolCall, Any]]] = []
    current_step: list[tuple[ToolCall, Any]] = []
    current_step_kinds: list[str] = []

    for call, res in zip(calls, padded_results):
        kind = str((catalog.get(call.name) or {}).get("kind") or "write")
        if not current_step:
            current_step.append((call, res))
            current_step_kinds.append(kind)
            continue

        prev_kind = current_step_kinds[-1]
        # Start a new ReAct turn when:
        # 1. Transitioning from read tool(s) to write/meta tool(s)
        # 2. Current call is a read tool whose x-ref / argument depends on an earlier read result in current_step
        # 3. Current call is sys.ask_missing_info or follows a meta call
        if (
            (prev_kind == "read" and kind != "read")
            or (prev_kind == "read" and kind == "read" and _read_depends_on_previous(call, current_step, catalog))
            or call.name == ASK_MISSING_INFO
            or prev_kind == "meta"
        ):
            steps.append(current_step)
            current_step = [(call, res)]
            current_step_kinds = [kind]
        else:
            current_step.append((call, res))
            current_step_kinds.append(kind)

    if current_step:
        steps.append(current_step)
    return steps


def build_sft_trajectory(
    sample: Sample,
    catalog: dict[str, dict[str, Any]],
    *,
    current_text: str | None = None,
    history_texts: dict[int, str] | None = None,
    max_prompt_tokens: int = 12000,
    max_result_chars: int = 1500,
    gold_state_hash: str | None = None,
) -> dict[str, Any]:
    """Construct a complete multi-step ChatML + tools trajectory matching run_react_loop."""
    messages = build_messages(
        sample,
        current_text=current_text,
        history_texts=history_texts,
        max_prompt_tokens=max_prompt_tokens,
        max_result_chars=max_result_chars,
    )
    tools = select_tools(sample, catalog)
    calls = list(sample.label.tool_calls) if sample.label else []
    results = list(sample.label.tool_results) if sample.label else []

    if not calls:
        messages.append(
            {
                "role": "assistant",
                "content": "Xin lỗi, thiết bị hiện chưa có ứng dụng phù hợp để thực hiện yêu cầu này.",
            }
        )
    else:
        steps = group_calls_into_steps(calls, results, catalog)
        ended_with_ask = False
        for step_items in steps:
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                        for c, _ in step_items
                    ],
                }
            )
            if any(c.name == ASK_MISSING_INFO for c, _ in step_items):
                ended_with_ask = True
                break
            for c, r in step_items:
                messages.append(
                    {
                        "role": "tool",
                        "name": c.name,
                        "content": truncate_tool_result(r, max_chars=max_result_chars),
                    }
                )
        if not ended_with_ask:
            messages.append(
                {
                    "role": "assistant",
                    "content": "Đã thực hiện xong yêu cầu của bạn.",
                }
            )

    return {
        "id": sample.id,
        "stratum": stratum(sample),
        "messages": messages,
        "tools": tools,
        "gold_state_hash": gold_state_hash,
    }


def _extract_msg_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tcs = msg.get("tool_calls") or []
    out: list[dict[str, Any]] = []
    for tc in raw_tcs:
        fn = tc.get("function") if isinstance(tc, dict) else None
        if isinstance(fn, dict) and "name" in fn:
            out.append({"name": fn["name"], "arguments": dict(fn.get("arguments") or {})})
    return out


def unroll_trajectory_steps(
    trajectory: dict[str, Any],
    split: str | None = None,
    root: str | None = None,
) -> list[dict[str, Any]]:
    """Decompose a multi-step ReAct trajectory into step-wise (prompt_messages, completion_message) turns."""
    messages: list[dict[str, Any]] = trajectory.get("messages") or []
    tools: list[dict[str, Any]] = trajectory.get("tools") or []
    sample_id = str(trajectory.get("id") or "")
    sample_stratum = str(trajectory.get("stratum") or "")
    gold_state_hash = trajectory.get("gold_state_hash")

    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not user_indices:
        return []
    last_user_idx = user_indices[-1]
    assistant_indices = [idx for idx in range(last_user_idx + 1, len(messages)) if messages[idx].get("role") == "assistant"]
    last_assistant_idx = assistant_indices[-1] if assistant_indices else -1

    env = None
    if split and root and sample_id:
        try:
            from src.env.safe_env import SafeEnv

            env = SafeEnv(split, sample_id, root)
        except Exception:
            env = None

    steps: list[dict[str, Any]] = []
    prefix_calls: list[dict[str, Any]] = []
    step_index = 0

    for idx in assistant_indices:
        msg = messages[idx]
        step_calls = _extract_msg_calls(msg)
        if not step_calls:
            step_category = "oos" if step_index == 0 else "final_stop"
        elif any(c["name"] == ASK_MISSING_INFO for c in step_calls):
            step_category = "ask"
        else:
            step_category = "act"

        step_hash = None
        if env is not None:
            for sc in step_calls:
                try:
                    env.call(sc["name"], sc.get("arguments", {}))
                except Exception:
                    pass
            try:
                step_hash = env.get_state_hash()
            except Exception:
                step_hash = None
        if step_hash is None and (idx == last_assistant_idx or step_calls):
            step_hash = gold_state_hash

        steps.append(
            {
                "id": sample_id,
                "stratum": sample_stratum,
                "step_index": step_index,
                "step_category": step_category,
                "is_terminal": idx == last_assistant_idx,
                "gold_state_hash": step_hash,
                "step_gold_state_hash": step_hash,
                "prompt_messages": list(messages[:idx]),
                "completion_message": dict(msg),
                "step_calls": step_calls,
                "prefix_calls": list(prefix_calls),
                "tools": tools,
            }
        )
        prefix_calls.extend(step_calls)
        step_index += 1

    return steps
