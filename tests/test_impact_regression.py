import json
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

base = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(base))
spec = importlib.util.spec_from_file_location("slippage", base / "slippage.py")
slip_mod = importlib.util.module_from_spec(spec)
sys.modules["slippage"] = slip_mod
spec.loader.exec_module(slip_mod)

SlippageConfig = slip_mod.SlippageConfig
model_curve = slip_mod.model_curve


def test_impact_participation_regression():
    data_path = pathlib.Path(__file__).resolve().parent / "data" / "impact_benchmark.json"
    with open(data_path, "r", encoding="utf-8") as f:
        bench = json.load(f)
    participations = np.array(bench["participation"], dtype=float)
    cfg = SlippageConfig(k=0.8, default_spread_bps=2.0, min_half_spread_bps=0.1)
    impact = model_curve(participations, cfg=cfg, spread_bps=cfg.default_spread_bps)
    assert impact == pytest.approx(np.array(bench["impact_bps"]))
