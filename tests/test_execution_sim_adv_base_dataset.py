import json

from execution_sim import ExecutionSimulator


def test_load_adv_base_dataset_preserves_all_symbols(tmp_path):
    payload = {
        "BTCUSDT": 123.45,
        "ETHUSDT": 67.89,
    }
    dataset_path = tmp_path / "adv_base.json"
    dataset_path.write_text(json.dumps(payload))

    simulator = ExecutionSimulator.__new__(ExecutionSimulator)
    dataset, _ = ExecutionSimulator._load_adv_base_dataset(simulator, str(dataset_path))

    assert dataset == {"BTCUSDT": 123.45, "ETHUSDT": 67.89}
