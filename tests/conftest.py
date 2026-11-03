from pathlib import Path

import pytest

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def data_root() -> Path:
    if not (DATA_ROOT / "train" / "train.jsonl").is_file():
        pytest.skip("competition data not available under data/")
    return DATA_ROOT
