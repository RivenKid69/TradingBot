import pandas as pd

from no_trade_config import get_no_trade_config
from no_trade import get_no_trade_config as nt_get
from trading_patchnew import TradingEnv


def test_consumers_load_identical_configs():
    path = "configs/legacy_sandbox.yaml"
    cfg = get_no_trade_config(path)
    assert nt_get(path) == cfg

    env = TradingEnv(pd.DataFrame({"ts_ms": [0]}), sandbox_config=path)
    assert env._no_trade_cfg == cfg
