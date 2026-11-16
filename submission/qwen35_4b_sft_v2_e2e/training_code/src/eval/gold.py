"""EVALUATION ONLY (train split): replay gold calls to obtain the gold state hash. Never import from inference code."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.schemas import Sample
from src.env.safe_env import load_toolenv_factory


@dataclass(frozen=True)
class GoldReplay:
    state_hash: str
    results: list[dict]
    changed_ok: bool  # get_state()[app] == label.final_state[app] for every app in final_state


def replay_gold(sample: Sample, root: str | Path) -> GoldReplay:
    if sample.label is None:
        raise ValueError(f"{sample.id} has no label; gold replay is train-only")
    env = load_toolenv_factory("train", root)("train", sample.id, root=str(root))
    results = [env.call(c.name, c.arguments) for c in sample.label.tool_calls]
    state = env.get_state()  # allowed: train split, evaluation only
    changed_ok = all(state.get(app) == st for app, st in sample.label.final_state.items())
    return GoldReplay(env.get_state_hash(), results, changed_ok)
