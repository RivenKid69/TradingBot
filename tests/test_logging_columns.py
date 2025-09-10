import importlib.util
import pathlib
import sys

import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent.parent

spec_exec = importlib.util.spec_from_file_location("execution_sim", BASE / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec_exec)
sys.modules["execution_sim"] = exec_mod
spec_exec.loader.exec_module(exec_mod)

spec_log = importlib.util.spec_from_file_location("logging", BASE / "logging.py")
log_mod = importlib.util.module_from_spec(spec_log)
sys.modules["logging"] = log_mod
spec_log.loader.exec_module(log_mod)

ActionType = exec_mod.ActionType
ExecutionSimulator = exec_mod.ExecutionSimulator
LogWriter = log_mod.LogWriter
LogConfig = log_mod.LogConfig


def test_logging_extended_columns(tmp_path):
    trades_path = tmp_path / "trades.csv"
    reports_path = tmp_path / "reports.csv"
    sim = ExecutionSimulator()
    sim.set_market_snapshot(bid=100.0, ask=101.0, liquidity=1.0)
    class Proto:
        def __init__(self):
            self.action_type = ActionType.MARKET
            self.volume_frac = 1.0
            self.ttl_steps = 5
            self.tif = "IOC"
    proto = Proto()
    sim.submit(proto)
    rep = sim.pop_ready(ref_price=100.5)
    log = LogWriter(LogConfig(trades_path=str(trades_path), reports_path=str(reports_path), flush_every=1))
    log.append(rep, symbol="BTCUSDT", ts_ms=0)
    log.flush()
    df = pd.read_csv(trades_path)
    for col in ["slippage_bps", "spread_bps", "latency_ms", "tif", "ttl_steps"]:
        assert col in df.columns
    row = df.iloc[0]
    assert row["tif"] == "IOC"
    assert row["ttl_steps"] == 5
