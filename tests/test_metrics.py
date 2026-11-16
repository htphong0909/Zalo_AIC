import unicodedata

import pytest

from src.eval.metrics import calls_match, count_matches, score_sample, token_f1

CATALOG = {
    "sms.send": {
        "parameters": {
            "properties": {
                "contact_id": {"type": "string"},
                "content": {"type": "string", "x-match": "fuzzy"},
            }
        }
    },
    "smart_home.light_set": {
        "parameters": {
            "properties": {
                "device_id": {"type": "string"},
                "power": {"type": "string"},
            }
        }
    },
}


def sms(content, contact="c_01"):
    return {"name": "sms.send", "arguments": {"contact_id": contact, "content": content}}


def test_token_f1_basic_and_nfc():
    assert token_f1("gọi cho mẹ", "gọi lại cho mẹ") == pytest.approx(6 / 7)
    assert token_f1("Hương", unicodedata.normalize("NFD", "hương")) == 1.0
    assert token_f1("", "") == 1.0
    assert token_f1("a", "") == 0.0


def test_fuzzy_threshold():
    assert calls_match(sms("gọi cho mẹ"), sms("gọi lại cho mẹ"), CATALOG)
    assert not calls_match(sms("mẹ"), sms("gọi lại cho mẹ"), CATALOG)


def test_non_fuzzy_must_be_identical_and_empty_args_ignored():
    gold = {"name": "smart_home.light_set", "arguments": {"device_id": "light_kitchen", "power": "off"}}
    assert calls_match(
        {
            "name": "smart_home.light_set",
            "arguments": {"device_id": "light_kitchen", "power": "off", "color": ""},
        },
        gold,
        CATALOG,
    )
    assert not calls_match(
        {"name": "smart_home.light_set", "arguments": {"device_id": "light_living", "power": "off"}},
        gold,
        CATALOG,
    )
    assert not calls_match(sms("x", "c_02"), sms("x", "c_01"), CATALOG)


def test_matching_is_one_to_one_and_maximal():
    gold = [sms("a b c"), sms("a b d")]
    preds = [sms("a b c d"), sms("a c")]  # greedy would pair pred0-gold0 and leave pred1 unmatched
    assert count_matches(preds, gold, CATALOG) == 2
    assert count_matches([sms("a b c"), sms("a b c")], [sms("a b c")], CATALOG) == 1


def test_score_sample_rules():
    g = [sms("xin chào")]
    assert score_sample([], "h", [], "h", CATALOG).score == 1.0
    assert score_sample(g, "h", [], "h", CATALOG).score == 0.0
    assert score_sample(g, "h", g, "h", CATALOG).score == 1.0
    mismatch = score_sample(g, "h1", g, "h2", CATALOG)
    assert mismatch.score == 0.0 and mismatch.f1 == 1.0
    extra = score_sample(g + [sms("khác hẳn", "c_09")], "h", g, "h", CATALOG)
    assert extra.precision == 0.5 and extra.recall == 1.0
    assert extra.score == pytest.approx(2 / 3)
