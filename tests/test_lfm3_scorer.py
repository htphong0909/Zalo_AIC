import pytest
from src.agent.lfm3_scorer import (
    arg_overlap_f1,
    calls_match,
    norm,
    same_value,
    tool_name_f1,
)


def _sample_catalog() -> dict[str, dict]:
    return {
        "banking.transfer": {
            "name": "banking.transfer",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_account_id": {"type": "string", "x-ref": "banking.accounts"},
                    "beneficiary_id": {"type": "string", "x-ref": "banking.beneficiaries"},
                    "amount": {"type": "integer"},
                    "note": {"type": "string", "x-match": "fuzzy"},
                },
                "required": ["from_account_id", "beneficiary_id", "amount"],
            },
        },
        "ewallet.transfer": {
            "name": "ewallet.transfer",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_id": {"type": "string"},
                    "amount": {"type": "integer"},
                },
                "required": ["amount"],
            },
        },
        "alarm.create": {
            "name": "alarm.create",
            "parameters": {
                "type": "object",
                "properties": {"time": {"type": "string", "format": "date-time"}},
                "required": ["time"],
            },
        },
    }


def test_norm_vietnamese_and_accents():
    assert norm("Phạm Hùng") == norm("pham hung")
    assert norm("  Hà   Nội! ") == "ha noi"


def test_same_value_kinds():
    assert same_value("number", 2800000, 2800000.0) is True
    assert same_value("bool", True, True) is True
    assert same_value("bool", True, "true") is False
    assert same_value("fuzzy", "chuyển tiền nhà tháng này", "chuyển tiền nhà tháng này nhé") is True
    assert same_value("datetime", "2026-11-20T06:00:00+07:00", "2026-11-20T06:00:00+07:00") is True


def test_calls_match_order_invariant_and_fuzzy():
    cat = _sample_catalog()
    gold = [
        {
            "name": "banking.transfer",
            "arguments": {
                "from_account_id": "acc_01",
                "beneficiary_id": "ben_03",
                "amount": 2800000,
                "note": "tiền nhà tháng 11",
            },
        },
        {"name": "ewallet.transfer", "arguments": {"amount": 200000, "contact_id": "c_04"}},
    ]
    pred_same = [
        {"name": "ewallet.transfer", "arguments": {"amount": 200000, "contact_id": "c_04"}},
        {
            "name": "banking.transfer",
            "arguments": {
                "from_account_id": "acc_01",
                "beneficiary_id": "ben_03",
                "amount": 2800000,
                "note": "tiền nhà tháng 11 nhé",
            },
        },
    ]
    pred_wrong_amount = [
        {
            "name": "banking.transfer",
            "arguments": {"from_account_id": "acc_01", "beneficiary_id": "ben_03", "amount": 3800000},
        }
    ]
    assert calls_match(gold, pred_same, cat) is True
    assert calls_match(gold, pred_wrong_amount, cat) is False


def test_tool_name_and_arg_overlap_f1():
    cat = _sample_catalog()
    gold = [{"name": "alarm.create", "arguments": {"time": "2026-11-21T06:00:00+07:00"}}]
    pred_right_tool_wrong_arg = [{"name": "alarm.create", "arguments": {"time": "2026-11-21T09:00:00+07:00"}}]
    assert tool_name_f1(gold, pred_right_tool_wrong_arg) == 1.0
    assert arg_overlap_f1(gold, pred_right_tool_wrong_arg, cat) == 0.0
    assert arg_overlap_f1(gold, gold, cat) == 1.0


def test_calls_match_strictly_matches_official_metrics_on_defaults_accents_enums_and_list_order():
    from src.eval.metrics import count_matches

    cat = {
        "alarm.create": {
            "name": "alarm.create",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"type": "string", "format": "time"},
                    "repeat": {"type": "string", "enum": ["none", "daily"], "default": "none"},
                },
                "required": ["time"],
            },
        },
        "contacts.search": {
            "name": "contacts.search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        "bill_payment.pay": {
            "name": "bill_payment.pay",
            "parameters": {
                "type": "object",
                "properties": {
                    "bill_id": {"type": "string"},
                    "source": {"type": "string", "enum": ["bank", "ewallet"]},
                },
                "required": ["bill_id", "source"],
            },
        },
        "sys.ask_missing_info": {
            "name": "sys.ask_missing_info",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "params": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["tool_name", "params"],
            },
        },
    }

    # 1. Hallucinated schema default ("repeat": "none") when gold omits optional arg -> official metric rejects!
    gold_alarm = [{"name": "alarm.create", "arguments": {"time": "06:00"}}]
    pred_hallucinated_default = [{"name": "alarm.create", "arguments": {"time": "06:00", "repeat": "none"}}]
    assert count_matches(pred_hallucinated_default, gold_alarm, cat) == 0
    assert calls_match(gold_alarm, pred_hallucinated_default, cat) is False

    # 2. Stripped Vietnamese diacritics on non-fuzzy field -> official metric rejects!
    gold_search = [{"name": "contacts.search", "arguments": {"query": "Phạm Hùng"}}]
    pred_unaccented = [{"name": "contacts.search", "arguments": {"query": "pham hung"}}]
    assert count_matches(pred_unaccented, gold_search, cat) == 0
    assert calls_match(gold_search, pred_unaccented, cat) is False

    # 3. Wrong enum casing ("BANK" vs "bank") -> official metric rejects!
    gold_pay = [{"name": "bill_payment.pay", "arguments": {"bill_id": "b_01", "source": "bank"}}]
    pred_upper_enum = [{"name": "bill_payment.pay", "arguments": {"bill_id": "b_01", "source": "BANK"}}]
    assert count_matches(pred_upper_enum, gold_pay, cat) == 0
    assert calls_match(gold_pay, pred_upper_enum, cat) is False

    # 4. Different array order on non-fuzzy params list -> official metric rejects!
    gold_ask = [
        {"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["contact_id", "content"]}}
    ]
    pred_reordered_ask = [
        {"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["content", "contact_id"]}}
    ]
    assert count_matches(pred_reordered_ask, gold_ask, cat) == 0
    assert calls_match(gold_ask, pred_reordered_ask, cat) is False

