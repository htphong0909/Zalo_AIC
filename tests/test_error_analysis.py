from __future__ import annotations

from src.core.dataset import load_catalog, load_split
from src.eval.error_analysis import analyze_errors
from src.eval.evaluator import evaluate


def test_analyze_errors_categorizes_key_and_value_mismatches():
    catalog = load_catalog("train", "data")
    samples = load_split("train", "data")[:5]

    preds = {}
    for s in samples:
        if s.label and s.label.tool_calls:
            first = s.label.tool_calls[0].model_dump()
            # Introduce an extra key and a wrong value on purpose
            bad_args = dict(first["arguments"])
            if bad_args:
                k0 = next(iter(bad_args))
                bad_args[k0] = "__wrong_value__"
            bad_args["__extra_key__"] = "unexpected"
            preds[s.id] = {
                "id": s.id,
                "tool_calls": [{"name": first["name"], "arguments": bad_args}],
                "state_hash": "bad_hash",
            }
        else:
            preds[s.id] = {"id": s.id, "tool_calls": [], "state_hash": ""}

    results = evaluate(preds, samples, catalog, "data")
    report = analyze_errors(preds, samples, results, catalog)
    assert "failure_mode_counts" in report
    assert "top_extra_arg_keys" in report
    assert "top_value_mismatches" in report
    assert report["failure_mode_counts"].get("arg_key_mismatch", 0) >= 1
