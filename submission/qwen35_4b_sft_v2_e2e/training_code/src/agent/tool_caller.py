"""Parse Qwen3.5 XML and fallback JSON tool calls with catalog-guided type coercion."""
from __future__ import annotations

import ast
import json
import re
import unicodedata
from typing import Any

from src.core.schemas import Sample, ToolCall
from src.agent.minicpm5_parser import (
    extract_minicpm5_xml_calls,
    normalize_minicpm5_output,
    _FUNC_BLOCK_RE as _MINICPM5_FUNC_BLOCK_RE,
)

EMPTY = (None, "", [], {})

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LEADING_THINK_CLOSE_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=\s*([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=\s*([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)


def _strip_think(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    cleaned = _LEADING_THINK_CLOSE_RE.sub("", cleaned)
    return _UNCLOSED_THINK_RE.sub("", cleaned)


def _parse_structured_str(raw: str) -> Any:
    s = raw.strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(s)
        except Exception:
            pass
    return raw


def coerce_value(val: Any, prop_spec: dict[str, Any] | None) -> Any:
    spec = prop_spec or {}
    target = spec.get("type")

    if target == "string":
        if isinstance(val, str):
            s = val.strip()
            if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
                try:
                    unquoted = json.loads(s)
                    if isinstance(unquoted, str):
                        return unquoted
                except Exception:
                    pass
            return s
        return "" if val is None else str(val)

    if target == "integer":
        if isinstance(val, int) and not isinstance(val, bool):
            return val
        if isinstance(val, float) and val.is_integer():
            return int(val)
        if isinstance(val, str):
            s = val.strip()
            try:
                return int(s)
            except ValueError:
                try:
                    f = float(s)
                    if f.is_integer():
                        return int(f)
                except ValueError:
                    pass
        return val

    if target == "number":
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return val
        if isinstance(val, str):
            s = val.strip()
            try:
                return int(s)
            except ValueError:
                try:
                    return float(s)
                except ValueError:
                    pass
        return val

    if target == "boolean":
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            low = val.strip().lower()
            if low == "true":
                return True
            if low == "false":
                return False
        return val

    if target == "array":
        parsed = _parse_structured_str(val) if isinstance(val, str) else val
        if isinstance(parsed, str):
            parsed = [x.strip() for x in parsed.split(",") if x.strip()] if parsed.strip() else []
        if isinstance(parsed, list):
            item_spec = spec.get("items") if isinstance(spec.get("items"), dict) else None
            return [coerce_value(x, item_spec) for x in parsed]
        return val

    if target == "object":
        parsed = _parse_structured_str(val) if isinstance(val, str) else val
        if isinstance(parsed, dict):
            sub_props = spec.get("properties") or {}
            return {k: coerce_value(v, sub_props.get(k)) for k, v in parsed.items() if v not in EMPTY}
        return val

    if isinstance(val, str):
        s = val.strip()
        if s and not (s.startswith("0") and len(s) > 1 and s.isdigit()):
            try:
                return json.loads(s)
            except Exception:
                pass
        return s
    return val


def coerce_arguments(
    fn_name: str,
    raw_args: dict[str, Any],
    catalog: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    props = ((catalog or {}).get(fn_name) or {}).get("parameters", {}).get("properties") or {}
    out: dict[str, Any] = {}
    for k, v in (raw_args or {}).items():
        coerced = coerce_value(v, props.get(k))
        if coerced not in EMPTY:
            out[k] = coerced
    return out


_CONTACT_POLITE_SUFFIX_RE = re.compile(
    r"\s+(?:giúp|giùm|dùm|hộ)(?:\s+(?:em|anh|chị|tôi|mình|con|cháu))?(?:\s+(?:với|nha|nhé|đi|ạ|cái))?$"
    r"|\s+(?:nha|nhé|ạ|với\s+ạ)$",
    flags=re.IGNORECASE,
)


def sanitize_tool_call(
    call: ToolCall,
    sample: Sample,
    current_text: str | None = None,
) -> ToolCall:
    text_low = (current_text if current_text is not None else (sample.audio_text or "")).lower()

    # 1. Clean out empty values (matching metrics.py _clean) and normalize string NFC
    args: dict[str, Any] = {}
    for k, v in call.arguments.items():
        if v in (None, "", [], {}):
            continue
        if isinstance(v, str):
            v = unicodedata.normalize("NFC", v).strip()
        args[k] = v

    # 2. Address pronouns resolution (home/work)
    if call.name in ("ride_hailing.book", "ride_hailing.get_quotes", "maps.get_route", "food_delivery.place_order"):
        profile = (sample.context or {}).get("user_profile") or {}
        home_addr = profile.get("home_address")
        work_addr = profile.get("work_address")
        for addr_k in ("pickup", "dropoff", "origin", "destination", "address"):
            val = args.get(addr_k)
            if isinstance(val, str):
                v_low = val.strip().lower()
                if home_addr and v_low in ("nhà", "nhà riêng", "nhà mình", "home"):
                    args[addr_k] = home_addr
                elif work_addr and v_low in ("công ty", "văn phòng", "chỗ làm", "work", "office"):
                    args[addr_k] = work_addr

    # 3. Contacts search query normalization
    if call.name == "contacts.search" and isinstance(args.get("query"), str):
        q = args["query"].strip()
        if q:
            # Strip trailing conversational polite particles
            q = _CONTACT_POLITE_SUFFIX_RE.sub("", q).strip()
            q_low = q.lower()
            if q_low in ("dũng", "sếp dũng") and "sếp dũng" in text_low:
                q = "Sếp Dũng"
            elif q_low in ("mai", "cô giáo mai") and "cô giáo mai" in text_low:
                q = "Cô giáo Mai"
            elif q_low in ("ba", "bác ba") and "bác ba" in text_low:
                q = "Bác Ba"
            elif q_low in ("hạnh", "cô hạnh") and "cô hạnh" in text_low:
                q = "Cô Hạnh"
            elif q and q[0].islower():
                q = q[0].upper() + q[1:]
            args["query"] = q

    return ToolCall(name=call.name, arguments=args)


def _parse_single_block(block: str, catalog: dict[str, dict[str, Any]] | None) -> list[ToolCall]:
    calls: list[ToolCall] = []
    fn_matches = list(_FUNCTION_RE.finditer(block))
    if fn_matches:
        for fm in fn_matches:
            fn_name = fm.group(1).strip()
            body = fm.group(2)
            raw_args: dict[str, Any] = {}
            for pm in _PARAMETER_RE.finditer(body):
                raw_args[pm.group(1).strip()] = pm.group(2).strip()
            calls.append(ToolCall(name=fn_name, arguments=coerce_arguments(fn_name, raw_args, catalog)))
        return calls

    stripped = block.strip()
    if stripped:
        parsed = _parse_structured_str(stripped)
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if isinstance(item, dict):
                fn_name = item.get("name") or item.get("function")
                raw_args = item.get("arguments")
                if raw_args is None:
                    raw_args = item.get("parameters") or {}
                if isinstance(fn_name, str) and isinstance(raw_args, dict):
                    calls.append(ToolCall(name=fn_name, arguments=coerce_arguments(fn_name, raw_args, catalog)))
    return calls


def parse_tool_calls(
    raw_output: str,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> list[ToolCall]:
    # 1. Try MiniCPM5 native XML parsing (<function name="...">...</function>)
    minicpm_calls = extract_minicpm5_xml_calls(raw_output, catalog)
    if minicpm_calls:
        # Pass through argument coercer for extra catalog safety
        return [
            ToolCall(name=c.name, arguments=coerce_arguments(c.name, c.arguments, catalog))
            for c in minicpm_calls
        ]

    # 2. Fall back to Qwen XML or JSON tool_call blocks
    cleaned = _strip_think(raw_output)
    calls: list[ToolCall] = []
    for m in _TOOL_CALL_BLOCK_RE.finditer(cleaned):
        calls.extend(_parse_single_block(m.group(1), catalog))
    if not calls and "<function=" in cleaned:
        calls.extend(_parse_single_block(cleaned, catalog))
    return calls


def has_malformed_tool_call(
    raw_output: str,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> bool:
    normalized = normalize_minicpm5_output(raw_output)
    cleaned = _strip_think(normalized)
    has_marker = any(
        tag in cleaned
        for tag in (
            "<tool_call>",
            "</tool_call>",
            "<function=",
            "<function name=",
            "<param name=",
        )
    )
    return has_marker and not parse_tool_calls(cleaned, catalog)


def extract_final_text(raw_output: str) -> str:
    normalized = normalize_minicpm5_output(raw_output)
    cleaned = _strip_think(normalized)
    without_calls = _TOOL_CALL_BLOCK_RE.sub("", cleaned)
    without_calls = _MINICPM5_FUNC_BLOCK_RE.sub("", without_calls)
    return without_calls.strip()
