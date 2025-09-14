import json
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import ws_dedup_state as s


def test_load_update_and_flush(tmp_path, caplog):
    state_file = tmp_path / "state.json"

    # load_state should handle missing file
    with caplog.at_level(logging.INFO):
        s.load_state(state_file)
    assert s.STATE == {}
    assert "does not exist" in caplog.text

    # initial update persists immediately
    s.update("BTCUSDT", 1000, path=state_file)
    assert json.loads(state_file.read_text()) == {"BTCUSDT": 1000}
    assert s.should_skip("BTCUSDT", 1000)
    assert s.should_skip("BTCUSDT", 900)
    assert not s.should_skip("BTCUSDT", 1100)

    # update without auto flush
    s.update("ETHUSDT", 2000, path=state_file, auto_flush=False)
    # file not yet updated
    assert json.loads(state_file.read_text()) == {"BTCUSDT": 1000}
    # flush manually
    s.flush(state_file)
    assert json.loads(state_file.read_text()) == {
        "BTCUSDT": 1000,
        "ETHUSDT": 2000,
    }

    # reload from disk
    s.STATE.clear()
    with caplog.at_level(logging.INFO):
        s.load_state(state_file)
    assert s.STATE == {"BTCUSDT": 1000, "ETHUSDT": 2000}
    assert "Loaded 2 symbols" in caplog.text


def test_periodic_flush(tmp_path):
    state_file = tmp_path / "state.json"
    s.init(enabled=True, persist_path=state_file, flush_interval_s=0.1)
    try:
        s.update("BTCUSDT", 1234, auto_flush=False)
        time.sleep(0.25)
    finally:
        s.shutdown()
    assert json.loads(state_file.read_text()) == {"BTCUSDT": 1234}
