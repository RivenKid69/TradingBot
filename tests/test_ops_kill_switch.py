import json, pathlib, sys

# Ensure stdlib logging is used instead of local logging.py
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_orig_sys_path = list(sys.path)
sys.path = [p for p in sys.path if p not in ("", str(REPO_ROOT))]
import logging as std_logging  # type: ignore
sys.modules["logging"] = std_logging
sys.path = _orig_sys_path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services import ops_kill_switch


def test_flag_file_trips_safe_mode(tmp_path):
    flag = tmp_path / "flag"
    state = tmp_path / "state.json"
    cfg = {"flag_path": str(flag), "state_path": str(state)}
    ops_kill_switch.init(cfg)
    assert not ops_kill_switch.tripped()
    flag.write_text("1")
    ops_kill_switch.init(cfg)
    assert ops_kill_switch.tripped()
    ops_kill_switch.manual_reset()
    assert not ops_kill_switch.tripped()
    assert not flag.exists()


def test_tick_persists_state(tmp_path):
    flag = tmp_path / "flag"
    state = tmp_path / "state.json"
    cfg = {"flag_path": str(flag), "state_path": str(state)}
    ops_kill_switch.init(cfg)
    ops_kill_switch.record_error("rest")
    ops_kill_switch.tick()
    data = json.loads(state.read_text())
    assert data["counters"]["rest"] == 1
    ops_kill_switch.manual_reset()
