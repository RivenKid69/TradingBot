import importlib.util
import pathlib
import sys

import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE := pathlib.Path(__file__).resolve().parent.parent))

from services.metrics import calculate_metrics

spec_exec = importlib.util.spec_from_file_location("execution_sim", BASE / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec_exec)
sys.modules["execution_sim"] = exec_mod
spec_exec.loader.exec_module(exec_mod)

spec_log = importlib.util.spec_from_file_location("sim_logging", BASE / "sim_logging.py")
log_mod = importlib.util.module_from_spec(spec_log)
sys.modules["sim_logging"] = log_mod
spec_log.loader.exec_module(log_mod)

ActionType = exec_mod.ActionType
ExecutionSimulator = exec_mod.ExecutionSimulator
LogWriter = log_mod.LogWriter
LogConfig = log_mod.LogConfig

def test_execution_profile_logging_and_metrics(tmp_path):
    trades_path = tmp_path / "trades.csv"
    reports_path = tmp_path / "reports.csv"
    sim = ExecutionSimulator(execution_profile="MKT_OPEN_NEXT_H1")
    sim.set_market_snapshot(bid=100.0, ask=101.0, liquidity=1.0)
    setattr(sim, "_last_market_regime", "BULL")

    class Proto:
        def __init__(self):
            self.action_type = ActionType.MARKET
            self.volume_frac = 1.0
            self.ttl_steps = 5
            self.tif = "IOC"

    proto = Proto()
    sim.submit(proto)
    sim.pop_ready(ref_price=100.5)
    rep = sim.pop_ready(ref_price=100.5)
    log = LogWriter(LogConfig(trades_path=str(trades_path), reports_path=str(reports_path), flush_every=1))
    log.append(rep, symbol="BTCUSDT", ts_ms=0)
    log.flush()
    df = pd.read_csv(trades_path)
    assert "execution_profile" in df.columns
    assert set(df["execution_profile"]) == {"MKT_OPEN_NEXT_H1"}
    assert "market_regime" in df.columns
    assert set(df["market_regime"].dropna()) == {"BULL"}

    trades = pd.DataFrame({
        "ts_ms": [1, 2, 3, 4],
        "pnl": [1.0, -0.5, 2.0, -1.0],
        "side": ["BUY", "SELL", "BUY", "SELL"],
        "qty": [1, 1, 1, 1],
        "execution_profile": ["A", "A", "B", "B"],
    })
    equity = pd.DataFrame({
        "ts_ms": [1, 2, 3, 4],
        "equity": [1.0, 0.5, 2.0, 1.0],
        "execution_profile": ["A", "A", "B", "B"],
    })
    metrics = calculate_metrics(trades, equity)
    assert set(metrics.keys()) == {"A", "B"}
    assert metrics["A"]["trades"]["n_trades"] == 2
    assert metrics["B"]["trades"]["n_trades"] == 2
