"""Clean Execution Verifier filtering candidate tool-calling steps against JSON Schema & constraints."""
from __future__ import annotations

from typing import Any

from src.core.schemas import ASK_MISSING_INFO


def _is_present(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, str) and val.strip() == "":
        return False
    return True


def validate_step_with_reason(
    step: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> tuple[bool, str]:
    """Validate tool names, required fields, x-one-of alternatives, enums, and ask_missing_info."""
    step_calls = step.get("step_calls")
    if step_calls is None:
        return False, "missing step_calls"
    if not isinstance(step_calls, list):
        return False, "step_calls must be a list"
    if len(step_calls) == 0:
        return True, "valid empty stop"

    for c in step_calls:
        if not isinstance(c, dict):
            return False, "tool call entry must be dict"
        name = c.get("name")
        args = c.get("arguments")
        if not isinstance(name, str) or not name:
            return False, "missing tool name"
        if not isinstance(args, dict):
            return False, f"arguments for {name} must be dict"

        if name == ASK_MISSING_INFO:
            target_tool = args.get("tool_name")
            params = args.get("params")
            if not isinstance(target_tool, str) or not target_tool or target_tool not in catalog:
                return False, f"ask_missing_info target tool_name invalid: {target_tool!r}"
            if not isinstance(params, list) or not params or not all(isinstance(p, str) and p for p in params):
                return False, f"ask_missing_info params invalid: {params!r}"
            target_props = ((catalog[target_tool].get("parameters") or {}).get("properties") or {})
            for p in params:
                if target_props and p not in target_props:
                    return False, f"ask_missing_info param '{p}' not in {target_tool} properties"
            continue

        if name not in catalog:
            return False, f"unknown tool '{name}'"

        params_spec = catalog[name].get("parameters") or {}
        props = params_spec.get("properties") or {}
        required = params_spec.get("required") or []

        for req in required:
            if req not in args or not _is_present(args[req]):
                return False, f"missing required parameter '{req}' for {name}"

        one_of = params_spec.get("x-one-of") or []
        if one_of:
            satisfied = any(
                all(k in args and _is_present(args[k]) for k in grp)
                for grp in one_of
            )
            if not satisfied:
                return False, f"missing x-one-of parameter group {one_of} for {name}"

        for k, v in args.items():
            if k in props and isinstance(props[k], dict):
                enum_vals = props[k].get("enum")
                if enum_vals is not None and v not in enum_vals:
                    return False, f"invalid enum value {v!r} for {name}.{k} (expected {enum_vals})"

    return True, "ok"


def verify_step_validity(step: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> bool:
    """Boolean wrapper around validate_step_with_reason for strict step filtering."""
    ok, _ = validate_step_with_reason(step, catalog)
    return ok
