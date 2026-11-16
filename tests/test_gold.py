from pathlib import Path

import pytest

from src.core.dataset import load_split
from src.eval.gold import replay_gold


def test_replay_requires_label(data_root: Path):
    s = load_split("train", data_root)[0].model_copy(update={"label": None})
    with pytest.raises(ValueError):
        replay_gold(s, data_root)


def test_replay_first_50_train_samples(data_root: Path):
    for sample in load_split("train", data_root)[:50]:
        g = replay_gold(sample, data_root)
        assert len(g.state_hash) == 64
        assert all(r.get("ok") for r in g.results), sample.id
        assert g.changed_ok, sample.id
