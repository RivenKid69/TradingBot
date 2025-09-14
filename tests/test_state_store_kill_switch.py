import json
from state_store import kill_switch_counters, load, save


def test_kill_switch_counters_persisted(tmp_path):
    store = tmp_path / "store.json"
    ops = tmp_path / "ops.json"

    kill_switch_counters.clear()
    kill_switch_counters.update({"rest": 1, "ws": 2})
    save(store, ops)

    data = json.loads(ops.read_text())
    assert data["counters"] == {"rest": 1, "ws": 2}

    kill_switch_counters.clear()
    load(store, ops)
    assert kill_switch_counters == {"rest": 1, "ws": 2}

    main_data = json.loads(store.read_text())
    assert "kill_switch_counters" not in main_data
