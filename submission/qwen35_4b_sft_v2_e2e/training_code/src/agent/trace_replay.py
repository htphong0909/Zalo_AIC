"""Replay recorded agent traces through SafeEnv under different GuardrailConfig settings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.boundary_verifier import BoundaryVerifier
from src.agent.react_loop import _apply_ask_guard, _collect_ambiguous_from_result
from src.agent.tool_caller import sanitize_tool_call
from src.core.config import GuardrailConfig
from src.core.schemas import ASK_MISSING_INFO, Sample, ToolCall
from src.env.safe_env import SafeEnv


def replay_trace_with_guards(
    sample: Sample,
    trace_row: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    root: str | Path = "data",
    *,
    split: str = "train",
    current_text: str | None = None,
    guardrails: GuardrailConfig | None = None,
) -> dict[str, Any]:
    """Re-execute a recorded trace's tool calls through a fresh SafeEnv with configurable guardrails."""
    guards = guardrails if guardrails is not None else GuardrailConfig()
    env = SafeEnv(split, sample.id, root)
    text_low = (current_text if current_text is not None else (sample.audio_text or "")).lower()
    boundary_verifier = BoundaryVerifier()

    ambig_contact_ids: set[str] = set()
    ambig_phones: set[str] = set()
    ambig_emails: set[str] = set()
    ambig_chat_ids: set[str] = set()

    steps = trace_row.get("trace") or []
    for step_entry in steps:
        if step_entry.get("event") != "executed":
            continue
        raw_calls = step_entry.get("calls") or []
        asked_missing = False
        for raw_c in raw_calls:
            c = ToolCall(name=raw_c["name"], arguments=dict(raw_c.get("arguments") or {}))
            if guards.enable_arg_sanitizer:
                c = sanitize_tool_call(c, sample, current_text=current_text)
            if guards.enable_boundary_verifier:
                c = boundary_verifier.verify_optional_args(c, current_text if current_text is not None else (sample.audio_text or ""))
            if guards.enable_ask_guard:
                c = _apply_ask_guard(
                    c,
                    text_low,
                    ambig_contact_ids,
                    ambig_phones,
                    ambig_emails,
                    ambig_chat_ids,
                )
            val = env.validate(c.name, c.arguments)
            if not val.get("ok", False):
                continue
            res = env.call(c.name, c.arguments)
            _collect_ambiguous_from_result(
                c.name,
                res,
                text_low,
                ambig_contact_ids,
                ambig_phones,
                ambig_emails,
                ambig_chat_ids,
            )
            if c.name == ASK_MISSING_INFO:
                asked_missing = True
                break
        if asked_missing:
            break

    return {
        "id": sample.id,
        "tool_calls": env.get_tool_calls(),
        "state_hash": env.get_state_hash(),
    }
