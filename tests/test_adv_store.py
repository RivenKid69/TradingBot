import os

from adv_store import ADVStore


def test_resolve_path_skips_missing_dataset(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    secondary = tmp_path / "secondary"
    secondary.mkdir()

    dataset_name = "foo"
    expected = secondary / dataset_name
    expected.touch()

    cfg = {
        "path": os.fspath(primary),
        "dataset": dataset_name,
        "extra": {
            "data_path": os.fspath(secondary),
        },
    }

    store = ADVStore(cfg)

    assert store.path == os.fspath(expected)
