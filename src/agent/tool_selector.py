"""Filter tool catalog to sample's installed apps and compact schemas for chat templates."""
from __future__ import annotations

import json
from typing import Any

from src.core.schemas import ASK_MISSING_INFO, Sample

_FORMAT_HINTS = {
    "date": "YYYY-MM-DD",
    "date-time": "ISO-8601 (ví dụ 2026-11-16T07:00:00+07:00)",
    "time": "HH:MM",
}

_KEEP_PROP_KEYS = (
    "type",
    "enum",
    "format",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "items",
)


def _compact_property(prop_spec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in _KEEP_PROP_KEYS:
        if k in prop_spec:
            out[k] = prop_spec[k]
    desc = (prop_spec.get("description") or "").strip()
    fmt = prop_spec.get("format")
    if fmt in _FORMAT_HINTS and _FORMAT_HINTS[fmt].split()[0] not in desc:
        hint = f"Dạng {_FORMAT_HINTS[fmt]}"
        desc = f"{desc} ({hint})" if desc else hint
    xref = prop_spec.get("x-ref")
    if isinstance(xref, str) and xref.strip():
        xref_tag = f"[x-ref: {xref.strip()}]"
        if xref_tag not in desc:
            desc = f"{desc} {xref_tag}".strip()
    if prop_spec.get("x-match") == "fuzzy" and "[x-match: fuzzy]" not in desc:
        desc = f"{desc} [x-match: fuzzy]".strip()
    if desc:
        out["description"] = desc
    return out


def compact_tool_spec(spec: dict[str, Any]) -> dict[str, Any]:
    params = spec.get("parameters") or {}
    props = params.get("properties") or {}
    req = list(params.get("required") or [])
    one_of = params.get("x-one-of") or []

    desc = (spec.get("description") or "").strip()
    if one_of:
        groups_str = " hoặc ".join("+".join(g) for g in one_of)
        suffix = f"(Bắt buộc có ít nhất một trong: {groups_str})"
        desc = f"{desc} {suffix}".strip()

    compacted_params: dict[str, Any] = {
        "type": "object",
        "properties": {k: _compact_property(v) for k, v in props.items()},
    }
    if req:
        compacted_params["required"] = req

    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": desc,
            "parameters": compacted_params,
        },
    }


def select_tools(sample: Sample, catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = sample.allowed_tools
    ordered_names = [name for name in catalog if name in allowed and name != ASK_MISSING_INFO]
    if ASK_MISSING_INFO in catalog and ASK_MISSING_INFO in allowed:
        ordered_names.append(ASK_MISSING_INFO)
    return [compact_tool_spec(catalog[name]) for name in ordered_names]


def estimate_tools_tokens(tools: list[dict[str, Any]]) -> int:
    serialized = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    return max(1, len(serialized) // 3)
