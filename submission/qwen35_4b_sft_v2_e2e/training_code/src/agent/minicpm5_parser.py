"""MiniCPM5 XML Tool Parser for agentic function calling.

Adapted and optimized from OpenBMB MiniCPM (branch minicpm5-2b):
tool_parsers/minicpm5xml_tool_parser.py
"""
from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from src.core.schemas import ToolCall

logger = logging.getLogger(__name__)

# SentencePiece/BPE tokenizers may emit \u0120 (Ġ) instead of ASCII space
_TOKENIZER_SPACE = "\u0120"

_FUNC_BLOCK_RE = re.compile(r"<function\b[^>]*>.*?</function>", re.DOTALL | re.IGNORECASE)
_FUNC_NAME_RE = re.compile(r"<function\s+name=['\"]([^'\"]+)['\"][^>]*>", re.IGNORECASE)
_PARAM_WITH_NAME_RE = re.compile(
    r"<param\s+name=['\"]([^'\"]+)['\"]>([\s\S]*?)</param>", re.DOTALL | re.IGNORECASE
)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LEADING_THINK_CLOSE_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)


def normalize_minicpm5_output(text: str) -> str:
    """Normalize model output by fixing tokenizer artifacts and collapsed tag names."""
    if not text:
        return ""
    normalized = text.replace(_TOKENIZER_SPACE, " ")
    # Fix collapsed tags like <functionname="foo"> or <paramname="bar">
    normalized = re.sub(r"<\s*function\s*name\s*=", "<function name=", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*param\s*name\s*=", "<param name=", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*/\s*function\s*>", "</function>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*/\s*param\s*>", "</param>", normalized, flags=re.IGNORECASE)
    return normalized


def _strip_think(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    cleaned = _LEADING_THINK_CLOSE_RE.sub("", cleaned)
    return _UNCLOSED_THINK_RE.sub("", cleaned)


def _convert_param_value(raw_val: str, expected_type: str | None = None) -> Any:
    """Convert raw XML param text into Python types guided by schema or heuristic parsing."""
    val = raw_val
    # Unpack CDATA wrapper if present
    if val.startswith("<![CDATA[") and val.endswith("]]>"):
        val = val[len("<![CDATA[") : -len("]]>")]

    if expected_type == "string":
        # Keep string raw without parsing numbers or booleans
        return val.strip()

    # If expected_type is explicit non-string or unknown, try json/ast parsing
    stripped = val.strip()
    if expected_type == "boolean":
        low = stripped.lower()
        if low == "true":
            return True
        if low == "false":
            return False

    if expected_type == "integer":
        try:
            return int(stripped)
        except ValueError:
            pass

    if expected_type == "number":
        try:
            f = float(stripped)
            return int(f) if f.is_integer() else f
        except ValueError:
            pass

    # Generic JSON/literal eval attempt for array/object/number/boolean
    if stripped.startswith(("{", "[", '"')) or stripped in ("true", "false", "null") or stripped.isdigit():
        try:
            return json.loads(stripped)
        except Exception:
            try:
                return ast.literal_eval(stripped)
            except Exception:
                pass

    return stripped


def extract_minicpm5_xml_calls(
    raw_output: str,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> list[ToolCall]:
    """Extract tool calls from MiniCPM5 XML format:

    <function name="tool_name">
      <param name="param_name">param_value</param>
    </function>
    """
    normalized = normalize_minicpm5_output(raw_output)
    cleaned = _strip_think(normalized)

    calls: list[ToolCall] = []
    # Match each <function ...>...</function> block
    for block_match in _FUNC_BLOCK_RE.finditer(cleaned):
        block = block_match.group(0)
        name_match = _FUNC_NAME_RE.search(block)
        if not name_match:
            continue
        func_name = name_match.group(1).strip()
        if not func_name:
            continue

        tool_spec = (catalog or {}).get(func_name, {})
        param_specs = (
            tool_spec.get("parameters", {}).get("properties", {})
            if isinstance(tool_spec, dict)
            else {}
        )

        args: dict[str, Any] = {}
        for pm in _PARAM_WITH_NAME_RE.finditer(block):
            param_name = pm.group(1).strip()
            raw_val = pm.group(2)
            expected_type = (
                param_specs.get(param_name, {}).get("type")
                if isinstance(param_specs, dict) and param_name in param_specs
                else None
            )
            converted = _convert_param_value(raw_val, expected_type)
            args[param_name] = converted

        calls.append(ToolCall(name=func_name, arguments=args))

    return calls
