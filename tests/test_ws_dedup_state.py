import json

import ws_dedup_state as sb


def setup_state(tmp_path):
    sb.PERSIST_PATH = tmp_path / "state.json"
    sb.STATE.clear()
    sb.ENABLED = True
    return sb.PERSIST_PATH


def test_load_state_reinit_on_corruption(tmp_path):
    p = setup_state(tmp_path)
    p.write_text("not-json")
    sb.load_state()
    assert sb.STATE == {}
    assert json.loads(p.read_text()) == {}


def test_update_flushes(tmp_path):
    p = setup_state(tmp_path)
    sb.update("BTC", 1000)
    assert json.loads(p.read_text()) == {"BTC": 1000}
