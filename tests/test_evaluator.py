from pathlib import Path

import pytest

from src.core.dataset import load_catalog, load_split
from src.eval.evaluator import evaluate, summarize_results
from src.eval.gold import replay_gold


@pytest.fixture(scope="module")
def subset(data_root: Path):
    return load_split("train", data_root)[:40], load_catalog("train", data_root)


def test_gold_predictions_score_one(subset, data_root: Path):
    samples, catalog = subset
    preds = {
        s.id: {
            "tool_calls": [c.model_dump() for c in s.label.tool_calls],
            "state_hash": replay_gold(s, data_root).state_hash,
        }
        for s in samples
    }
    summary = summarize_results(evaluate(preds, samples, catalog, data_root), samples)
    assert summary["final_score"] == pytest.approx(1.0)


def test_empty_predictions_score_equals_oos_share(subset, data_root: Path):
    samples, catalog = subset
    summary = summarize_results(evaluate({}, samples, catalog, data_root), samples)
    oos = sum(1 for s in samples if not s.label.tool_calls) / len(samples)
    assert summary["final_score"] == pytest.approx(oos)
