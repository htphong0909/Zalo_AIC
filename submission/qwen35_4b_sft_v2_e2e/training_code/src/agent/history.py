"""Render conversation history and current turn into chat messages within token budget."""
from __future__ import annotations

import json
from typing import Any

from src.agent.prompts import build_system_prompt
from src.core.schemas import Sample, Turn


def truncate_tool_result(result: Any, max_chars: int = 1500) -> str:
    text = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15] + '..."truncated"}'


def _estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, len(serialized) // 3)


def _render_turn_group(
    idx: int,
    turn: Turn,
    history_texts: dict[int, str] | None,
    max_result_chars: int,
) -> list[dict[str, Any]]:
    if turn.role == "user":
        content = (history_texts or {}).get(idx)
        if content is None:
            content = turn.audio_text or turn.text or ""
        return [{"role": "user", "content": content}]

    if turn.role == "assistant":
        if not turn.tool_calls:
            return [{"role": "assistant", "content": turn.text or ""}]
        msgs: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": turn.text or "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    }
                    for c in turn.tool_calls
                ],
            }
        ]
        for i, call in enumerate(turn.tool_calls):
            res = turn.tool_results[i] if i < len(turn.tool_results) else {"ok": True}
            msgs.append(
                {
                    "role": "tool",
                    "name": call.name,
                    "content": truncate_tool_result(res, max_chars=max_result_chars),
                }
            )
        return msgs

    return []


def build_messages(
    sample: Sample,
    *,
    current_text: str | None = None,
    history_texts: dict[int, str] | None = None,
    asr_phonetic_diff: str | None = None,
    prompt_style: str = "v1",
    max_prompt_tokens: int = 12000,
    max_result_chars: int = 1500,
) -> list[dict[str, Any]]:
    system_msg = {"role": "system", "content": build_system_prompt(sample, prompt_style=prompt_style)}
    user_text = current_text if current_text is not None else (sample.audio_text or "")
    if asr_phonetic_diff:
        user_text = f"{user_text}\n{asr_phonetic_diff}"
    current_msg = {"role": "user", "content": user_text}

    groups = [
        _render_turn_group(i, t, history_texts, max_result_chars)
        for i, t in enumerate(sample.history)
    ]
    flat_history = [m for g in groups for m in g]
    candidate = [system_msg, *flat_history, current_msg]

    if _estimate_messages_tokens(candidate) > max_prompt_tokens and sample.history:
        groups = [
            _render_turn_group(i, t, history_texts, 300)
            for i, t in enumerate(sample.history)
        ]
        while groups:
            flat_history = [m for g in groups for m in g]
            candidate = [system_msg, *flat_history, current_msg]
            if _estimate_messages_tokens(candidate) <= max_prompt_tokens:
                break
            groups.pop(0)
            if groups and groups[0] and groups[0][0]["role"] == "assistant":
                groups.pop(0)
        flat_history = [m for g in groups for m in g]
        candidate = [system_msg, *flat_history, current_msg]

    return candidate
