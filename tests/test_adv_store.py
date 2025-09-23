import os

from adv_store import ADVStore


def test_adv_store_resolves_dataset_from_later_candidate(tmp_path):
    dataset_name = "adv.json"

    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()

    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    dataset_path = valid_dir / dataset_name
    dataset_path.write_text("{}", encoding="utf-8")

    cfg = {
        "path": os.fspath(missing_dir),
        "extra": {"adv_path": os.fspath(valid_dir)},
        "dataset": dataset_name,
    }

    store = ADVStore(cfg)

    assert store.path == os.fspath(dataset_path)


def test_adv_store_skips_missing_file_candidate(tmp_path):
    missing_file = tmp_path / "missing.json"
    valid_file = tmp_path / "valid.json"
    valid_file.write_text("{}", encoding="utf-8")

    cfg = {
        "path": os.fspath(missing_file),
        "extra": {"adv_path": os.fspath(valid_file)},
    }

    store = ADVStore(cfg)

    assert store.path == os.fspath(valid_file)
