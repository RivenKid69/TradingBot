import importlib.util
import pathlib
import sys

import pandas as pd
import pytest

BASE = pathlib.Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# load execution_sim module dynamically
spec = importlib.util.spec_from_file_location("execution_sim", BASE / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec)
sys.modules["execution_sim"] = exec_mod
spec.loader.exec_module(exec_mod)

from aggregate_exec_logs import recompute_pnl

ActionProto = exec_mod.ActionProto
ActionType = exec_mod.ActionType
ExecutionSimulator = exec_mod.ExecutionSimulator


def _trades_to_df(trades):
    return pd.DataFrame(
        {
            "ts": list(range(len(trades))),
            "side": [t.side for t in trades],
            "price": [t.price for t in trades],
            "quantity": [t.qty for t in trades],
        }
    )


def _reports_to_df(reports):
    rows = []
    for i, r in enumerate(reports):
        rows.append(
            {
                "ts_ms": i,
                "bid": r.bid,
                "ask": r.ask,
                # first report keeps mtm_price, second drops to test bid/ask fallback
                "mtm_price": r.mtm_price if i == 0 else float("nan"),
                "realized_pnl": r.realized_pnl,
                "unrealized_pnl": r.unrealized_pnl,
            }
        )
    return pd.DataFrame(rows)


def test_recompute_pnl_uses_reports_data():
    sim = ExecutionSimulator()
    sim.set_market_snapshot(bid=100.0, ask=101.0)
    trades_log = []
    reports = []

    sim.submit(ActionProto(action_type=ActionType.MARKET, volume_frac=1.0))
    rep1 = sim.pop_ready(ref_price=100.5)
    trades_log.extend(rep1.trades)
    reports.append(rep1)

    sim.set_market_snapshot(bid=102.0, ask=103.0)
    sim.submit(ActionProto(action_type=ActionType.MARKET, volume_frac=-1.0))
    rep2 = sim.pop_ready(ref_price=102.5)
    trades_log.extend(rep2.trades)
    reports.append(rep2)

    trades_df = _trades_to_df(trades_log)
    reports_df = _reports_to_df(reports)

    recomputed = recompute_pnl(trades_df, reports_df)
    expected = reports_df["realized_pnl"] + reports_df["unrealized_pnl"]

    assert list(recomputed) == pytest.approx(list(expected), abs=1e-9)
