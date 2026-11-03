from collections import Counter

from src.core.schemas import Sample
from src.eval.splits import make_dev_split, stratum


def mk(i: int, n_calls: int, ask: bool = False, history: bool = False) -> Sample:
    calls = [{"name": "sys.ask_missing_info" if ask else "a.b", "arguments": {}}] * n_calls
    hist = [{"role": "user", "audio": "x.wav"}] if history else []
    return Sample.model_validate(
        {"id": f"s{i:04d}", "audio": "a.wav", "context": {}, "history": hist, "label": {"tool_calls": calls}}
    )


def test_stratum_labels():
    assert stratum(mk(0, 0)) == "oos|0|n"
    assert stratum(mk(1, 1, ask=True, history=True)) == "ask|1|h"
    assert stratum(mk(2, 3)) == "act|2-3|n"
    assert stratum(mk(3, 5)) == "act|4+|n"


def test_split_is_deterministic_and_proportional():
    samples = [mk(i, i % 5, history=bool(i % 2)) for i in range(1000)]
    a, b = make_dev_split(samples, 200, 7), make_dev_split(samples, 200, 7)
    assert a == b
    assert abs(len(a) - 200) <= 10
    dev = {s.id for s in samples if s.id in set(a)}
    by_stratum = Counter(stratum(s) for s in samples if s.id in dev)
    full = Counter(stratum(s) for s in samples)
    for key, n in full.items():
        assert abs(by_stratum[key] / len(a) - n / len(samples)) < 0.02
