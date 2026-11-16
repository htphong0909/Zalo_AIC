"""Unit tests for train_only 5-fold stratification and strict zero-leakage guarantee."""
import json
from pathlib import Path
from src.core.dataset import load_split
from scripts.make_folds import generate_train_only_5folds


def test_train_only_5folds_no_leak_and_balanced(tmp_path):
    out_file = tmp_path / "train_only_5folds.json"
    data = generate_train_only_5folds(
        data_root="data",
        dev_split_path="configs/splits/dev_v1.json",
        output_path=str(out_file),
        num_folds=5,
        seed=20261108,
    )

    dev_ids = set(json.loads(Path("configs/splits/dev_v1.json").read_text(encoding="utf-8"))["dev_ids"])
    all_train = load_split("train", "data")
    train_only_ids = {s.id for s in all_train if s.id not in dev_ids}

    assert len(train_only_ids) == 1246, f"Expected 1246 train_only samples, got {len(train_only_ids)}"

    folds = data["folds"]
    assert len(folds) == 5

    all_fold_ids = set()
    for fold_name, f_ids in folds.items():
        # Check size balance (~249 per fold)
        assert 248 <= len(f_ids) <= 251, f"{fold_name} size out of balance: {len(f_ids)}"

        # Strict zero-leak into dev_v1
        leak = set(f_ids) & dev_ids
        assert len(leak) == 0, f"Critical leak into dev_v1 in {fold_name}: {leak}"

        # No overlap between folds
        assert len(all_fold_ids & set(f_ids)) == 0, f"Overlapping IDs between folds in {fold_name}"
        all_fold_ids.update(f_ids)

    # Union covers 100% of train_only
    assert all_fold_ids == train_only_ids, "Folds do not partition train_only completely"
