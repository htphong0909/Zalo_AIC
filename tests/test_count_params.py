from scripts.count_params import check_budget, summarize

LIMITS = {"llm_max_params": 5_000_000_000, "total_max_params": 6_000_000_000}


def test_summarize_totals_and_groups():
    tensors = {"model.layers.0.w": 10, "model.layers.1.w": 20, "visual.blocks.0.w": 5}
    out = summarize(tensors, exclude_prefixes=("visual.",), depth=2)
    assert out["total"] == 35
    assert out["excluded"] == 5
    assert out["kept"] == 30
    assert out["groups"] == {"model.layers": 30, "visual.blocks": 5}


def test_summarize_without_exclusion_keeps_everything():
    out = summarize({"a.b": 3, "c.d": 4})
    assert out["excluded"] == 0 and out["kept"] == 7


def test_check_budget_passes_within_limits():
    assert check_budget(900_000_000, 4_200_000_000, LIMITS) == []


def test_check_budget_flags_llm_and_total():
    errors = check_budget(2_100_000_000, 5_000_000_000, LIMITS)
    assert any("LLM" in e for e in errors)
    assert any("total" in e for e in errors)
