import importlib.util
import pathlib
import sys
import json
import time
import numpy as np

BASE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

spec = importlib.util.spec_from_file_location("utils_time", BASE / "utils_time.py")
utils_time = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_time)

watch_seasonality_file = utils_time.watch_seasonality_file
HOURS_IN_WEEK = utils_time.HOURS_IN_WEEK


def test_watch_seasonality_file_retains_last_good(tmp_path, capsys):
    path = tmp_path / "mult.json"
    good = {"latency": [1.0] * HOURS_IN_WEEK}
    path.write_text(json.dumps(good))

    calls = []
    watch_seasonality_file(str(path), lambda d: calls.append(d), poll_interval=0.05)
    time.sleep(0.1)
    assert len(calls) >= 1

    path.write_text("{ bad json")
    time.sleep(0.2)
    captured = capsys.readouterr()

    assert len(calls) >= 2
    assert np.allclose(calls[-1]["latency"], np.ones(HOURS_IN_WEEK))
    assert "Failed to reload seasonality multipliers" in captured.err

    # Restore valid file to avoid noisy background logs
    path.write_text(json.dumps(good))
    time.sleep(0.1)
