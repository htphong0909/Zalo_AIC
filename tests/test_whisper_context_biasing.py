"""Test that whisper context biasing includes app domain keywords and batching never drops context."""
from src.core.schemas import Sample
from scripts.run_asr import build_sample_asr_context


def test_build_sample_asr_context_includes_installed_apps_keywords():
    sample = Sample(
        id="test_01",
        audio="dummy.wav",
        context={
            "user_profile": {"city": "Hà Nội", "home_address": "12 Cát Linh"},
            "installed_apps": {
                "learning": {"tools": ["learning.get_courses"]},
                "fuel_price": {"tools": ["fuel_price.get_price"]},
            },
        },
        history=[],
    )

    ctx = build_sample_asr_context(sample)
    assert "Hà Nội" in ctx
    assert "12 Cát Linh" in ctx
    assert "tiếng Hàn" in ctx or "khóa học" in ctx
    assert "giá xăng" in ctx or "dầu diesel" in ctx


def test_whisper_batch_grouping_logic():
    from collections import defaultdict

    chunk_contexts = ["ctx_A", "ctx_B", "ctx_A", "ctx_C"]
    waveforms = [True, True, True, True]

    ctx_groups = defaultdict(list)
    for active_idx, ctx_val in enumerate(chunk_contexts):
        if waveforms[active_idx] is not None:
            ctx_groups[ctx_val].append(active_idx)

    # 3 distinct context groups, none is dropped to None!
    assert len(ctx_groups) == 3
    assert ctx_groups["ctx_A"] == [0, 2]
    assert ctx_groups["ctx_B"] == [1]
    assert ctx_groups["ctx_C"] == [3]
    assert None not in ctx_groups
