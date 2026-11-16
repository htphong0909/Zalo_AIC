"""Shared canonical prompt, history, tool-list, and XML helpers for synthetic data generators."""
from __future__ import annotations

import json
from typing import Any

from src.agent.history import build_messages, truncate_tool_result
from src.agent.tool_selector import select_tools
from src.core.schemas import ASK_MISSING_INFO, Sample, ToolCall


def call_to_dict(call: ToolCall | dict[str, Any]) -> dict[str, Any]:
    """Normalize a ToolCall or raw dict to ``{'name': ..., 'arguments': {...}}``."""
    if isinstance(call, dict):
        return {"name": str(call.get("name", "")), "arguments": dict(call.get("arguments") or {})}
    return {"name": call.name, "arguments": dict(call.arguments or {})}


def resolve_installed_apps(
    sample: Sample,
    fallback_tools: list[str] | None = None,
) -> list[str]:
    """Return all installed app names for ``sample``, inferring from tools only when absent."""
    apps = list((sample.context or {}).get("installed_apps", {}).keys())
    if apps:
        return apps
    inferred: list[str] = []
    for tname in fallback_tools or []:
        if not tname or tname == ASK_MISSING_INFO or "." not in tname:
            continue
        app = tname.split(".")[0]
        if app not in inferred:
            inferred.append(app)
    return inferred


def merge_sample_contexts(s1: Sample, s2: Sample) -> dict[str, Any]:
    """Merge runtime context from two samples while preserving time/location/profile."""
    ctx1 = dict(s1.context or {})
    ctx2 = dict(s2.context or {})
    apps1 = dict(ctx1.get("installed_apps") or {})
    apps2 = dict(ctx2.get("installed_apps") or {})
    merged_apps = dict(apps1)
    for k, v in apps2.items():
        if k not in merged_apps:
            merged_apps[k] = v
    merged_ctx = dict(ctx1)
    merged_ctx["installed_apps"] = merged_apps
    return merged_ctx


def build_sample_prefix_messages(
    sample: Sample,
    catalog: dict[str, Any] | None = None,
    *,
    user_text: str | None = None,
    max_result_chars: int = 1500,
) -> list[dict[str, Any]]:
    """Build canonical [system, ...history..., user] prompt messages via build_messages."""
    _ = catalog
    return build_messages(
        sample,
        current_text=user_text,
        max_result_chars=max_result_chars,
    )


def build_assistant_history_message(
    calls: list[ToolCall | dict[str, Any]],
    *,
    think: str = "",
    json_params: bool = False,
) -> dict[str, Any]:
    """Create a structured assistant message holding both ``tool_calls`` and rendered XML."""
    norm_calls = [call_to_dict(c) for c in calls]
    xml = render_function_xml(norm_calls, python_repr=not json_params)
    content = f"{think}\n\n{xml}" if think else xml
    return {"role": "assistant", "content": content, "tool_calls": norm_calls}


def render_function_xml(calls: list[dict[str, Any]], *, python_repr: bool = True) -> str:
    """Serialize tool calls into native MiniCPM5 <function name=...> XML tags."""
    blocks: list[str] = []
    for c in calls:
        name = str(c.get("name", ""))
        args = c.get("arguments") or {}
        if not args:
            blocks.append(f'<function name="{name}"></function>')
            continue
        params_xml: list[str] = []
        for k, v in args.items():
            if isinstance(v, str):
                val_str = v
            elif python_repr:
                val_str = repr(v)
            else:
                val_str = json.dumps(v, ensure_ascii=False)
            params_xml.append(f'<param name="{k}">{val_str}</param>')
        blocks.append(f'<function name="{name}">{"".join(params_xml)}</function>')
    return "\n".join(blocks)


def resolve_sample_tool_specs(sample: Sample, catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve OpenAI-format tool specs matching inference select_tools()."""
    if not catalog:
        return []
    return select_tools(sample, catalog)


def serialize_observation(result: Any, *, max_chars: int = 1500) -> str:
    """Serialize a tool observation cleanly via truncate_tool_result."""
    return truncate_tool_result(result, max_chars=max_chars)
