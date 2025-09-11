import json
import numpy as np
import pytest

from utils_time import load_seasonality, HOURS_IN_WEEK

def _arr(v):
    return [float(v)] * HOURS_IN_WEEK


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
