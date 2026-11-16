from __future__ import annotations

from src.agent.tool_selector import estimate_tools_tokens, select_tools
from src.core.dataset import load_catalog, load_split
from src.core.schemas import ASK_MISSING_INFO


def test_select_tools_filters_to_installed_apps_and_ask_missing_info():
    catalog = load_catalog("train", "data")
    sample = load_split("train", "data")[0]

    tools = select_tools(sample, catalog)
    names = [t["function"]["name"] for t in tools]

    assert set(names) == sample.allowed_tools
    assert ASK_MISSING_INFO in names
    assert all(t["type"] == "function" for t in tools)
    assert estimate_tools_tokens(tools) > 100


def test_select_tools_surfaces_x_one_of_and_format_hints():
    catalog = load_catalog("train", "data")
    sample = next(s for s in load_split("train", "data") if "contacts.update" in s.allowed_tools)

    tools = {t["function"]["name"]: t["function"] for t in select_tools(sample, catalog)}
    update_fn = tools["contacts.update"]
    assert "phone" in update_fn["description"] and "email" in update_fn["description"]

    sample_cal = next(s for s in load_split("train", "data") if "weather.get_forecast" in s.allowed_tools)
    cal_tools = {t["function"]["name"]: t["function"] for t in select_tools(sample_cal, catalog)}
    date_prop = cal_tools["weather.get_forecast"]["parameters"]["properties"]["date"]
    assert "YYYY-MM-DD" in date_prop.get("description", "")


def test_compact_tool_spec_surfaces_xref_and_fuzzy_hints():
    from src.agent.tool_selector import compact_tool_spec

    spec = {
        "name": "banking.transfer",
        "description": "Chuyển khoản ngân hàng",
        "parameters": {
            "type": "object",
            "properties": {
                "beneficiary_id": {
                    "type": "string",
                    "description": "Mã người thụ hưởng",
                    "x-ref": "banking.beneficiaries",
                },
                "note": {
                    "type": "string",
                    "description": "Nội dung chuyển khoản",
                    "x-match": "fuzzy",
                },
            },
            "required": ["beneficiary_id"],
        },
    }
    compacted = compact_tool_spec(spec)
    props = compacted["function"]["parameters"]["properties"]
    assert "x-ref" not in props["beneficiary_id"]
    assert "x-match" not in props["note"]
    assert "[x-ref: banking.beneficiaries]" in props["beneficiary_id"]["description"]
    assert "[x-match: fuzzy]" in props["note"]["description"]

