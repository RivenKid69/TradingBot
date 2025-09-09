import numpy as np
import pandas as pd
import pytest
import os
import sys

sys.path.append(os.getcwd())

from no_trade import compute_no_trade_mask, estimate_block_ratio
from no_trade_config import get_no_trade_config


def test_blocked_share_matches_legacy_config():
    cfg = get_no_trade_config("configs/legacy_sandbox.yaml")
    ts = np.arange(0, 24 * 60, dtype=np.int64) * 60_000
    df = pd.DataFrame({"ts_ms": ts})
    mask = compute_no_trade_mask(df, sandbox_yaml_path="configs/legacy_sandbox.yaml")
    est = estimate_block_ratio(df, cfg)
    expected = 28 / 1440
    assert est == pytest.approx(expected, abs=1e-6)
    assert mask.mean() == pytest.approx(expected, abs=1e-6)
