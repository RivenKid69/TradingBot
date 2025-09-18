import json
import importlib.util
import pathlib
import sys
import numpy as np
import pytest

BASE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

spec_utils = importlib.util.spec_from_file_location("utils_time", BASE / "utils_time.py")
utils_time = importlib.util.module_from_spec(spec_utils)
sys.modules["utils_time"] = utils_time
spec_utils.loader.exec_module(utils_time)

load_seasonality = utils_time.load_seasonality
load_hourly_seasonality = utils_time.load_hourly_seasonality
HOURS_IN_WEEK = utils_time.HOURS_IN_WEEK
SEASONALITY_MULT_MIN = utils_time.SEASONALITY_MULT_MIN
SEASONALITY_MULT_MAX = utils_time.SEASONALITY_MULT_MAX

def _arr(v):
    return [float(v)] * HOURS_IN_WEEK


def test_load_seasonality_sample_file():
    path = BASE / "configs" / "liquidity_latency_seasonality.sample.json"
    res = load_seasonality(str(path))
    assert np.allclose(res["liquidity"], 1.0)
    assert np.allclose(res["latency"], 1.0)
    assert np.allclose(res["spread"], 1.0)


def test_load_seasonality_basic(tmp_path):
    data = {"liquidity": _arr(1.0), "latency": _arr(2.0)}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data))
    res = load_seasonality(str(p))
    assert np.allclose(res["liquidity"], 1.0)
    assert np.allclose(res["latency"], 2.0)


def test_load_seasonality_nested(tmp_path):
    data = {"BTCUSDT": {"spread": _arr(3.0)}}
    p = tmp_path / "nested.json"
    p.write_text(json.dumps(data))
    res = load_seasonality(str(p))
    assert np.allclose(res["spread"], 3.0)


def test_load_seasonality_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_seasonality(str(tmp_path / "missing.json"))


def test_load_seasonality_bad_length(tmp_path):
    data = {"liquidity": [1.0]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_seasonality(str(p))


def test_seasonality_negative_values(tmp_path):
    p = tmp_path / "neg.json"
    p.write_text(json.dumps({"liquidity": [-1.0] * HOURS_IN_WEEK}))
    with pytest.raises(ValueError):
        load_seasonality(str(p))
    with pytest.raises(ValueError):
        load_hourly_seasonality(str(p), "liquidity")


def test_seasonality_clamping(tmp_path):
    data = {
        "liquidity": [0.01] * HOURS_IN_WEEK,
        "latency": [20.0] * HOURS_IN_WEEK,
    }
    p = tmp_path / "clamp.json"
    p.write_text(json.dumps(data))
    res = load_seasonality(str(p))
    assert res["liquidity"].min() == SEASONALITY_MULT_MIN
    assert res["latency"].max() == SEASONALITY_MULT_MAX


def test_hourly_seasonality_clamping(tmp_path):
    p = tmp_path / "liq.json"
    p.write_text(json.dumps({"liquidity": [0.01] * HOURS_IN_WEEK}))
    arr = load_hourly_seasonality(str(p), "liquidity")
    assert arr.min() == SEASONALITY_MULT_MIN

    p2 = tmp_path / "lat.json"
    p2.write_text(json.dumps({"latency": [20.0] * HOURS_IN_WEEK}))
    arr2 = load_hourly_seasonality(str(p2), "latency")
    assert arr2.max() == SEASONALITY_MULT_MAX


def test_hourly_seasonality_mapping(tmp_path):
    mapping = {str(i): 1.0 for i in range(HOURS_IN_WEEK)}
    p = tmp_path / "map.json"
    p.write_text(json.dumps({"latency": mapping}))
    arr = load_hourly_seasonality(str(p), "latency")
    assert arr.shape[0] == HOURS_IN_WEEK
    assert np.allclose(arr, 1.0)
    data = load_seasonality(str(p))
    assert np.allclose(data["latency"], 1.0)

    day_mapping = {str(i): 2.0 for i in range(7)}
    p2 = tmp_path / "map_day.json"
    p2.write_text(json.dumps({"latency": day_mapping}))
    arr_day = load_hourly_seasonality(str(p2), "latency")
    assert arr_day.shape[0] == 7
    assert np.allclose(arr_day, 2.0)


def test_load_seasonality_daily(tmp_path):
    p = tmp_path / "day.json"
    p.write_text(json.dumps({"liquidity": [1.0] * 7}))
    res = load_seasonality(str(p))
    assert len(res["liquidity"]) == 7
    arr = load_hourly_seasonality(str(p), "liquidity")
    assert len(arr) == 7
