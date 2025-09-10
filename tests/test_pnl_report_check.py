import importlib.util
import pathlib
import sys

import pytest

BASE = pathlib.Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

spec = importlib.util.spec_from_file_location("execution_sim", BASE / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec)
sys.modules["execution_sim"] = exec_mod
spec.loader.exec_module(exec_mod)

ActionProto = exec_mod.ActionProto
ActionType = exec_mod.ActionType
ExecutionSimulator = exec_mod.ExecutionSimulator


def _recompute_total(trades, bid, ask, mtm_price):
    pos = 0.0
    avg = None
    realized = 0.0
    for tr in trades:
        price = tr.price
        qty = tr.qty
        if tr.side == "BUY":
            if pos < 0.0:
                close_qty = min(qty, -pos)
                if avg is not None:
                    realized += (avg - price) * close_qty
                pos += close_qty
                qty -= close_qty
                if qty > 0.0:
                    pos += qty
                    avg = price
                elif pos == 0.0:
                    avg = None
            else:
                new_pos = pos + qty
                avg = (avg * pos + price * qty) / new_pos if pos > 0.0 and avg is not None else price
                pos = new_pos
        else:  # SELL
            if pos > 0.0:
                close_qty = min(qty, pos)
                if avg is not None:
                    realized += (price - avg) * close_qty
                pos -= close_qty
                qty -= close_qty
                if qty > 0.0:
                    pos -= qty
                    avg = price
                elif pos == 0.0:
                    avg = None
            else:
                new_pos = pos - qty
                avg = (avg * (-pos) + price * qty) / (-new_pos) if pos < 0.0 and avg is not None else price
                pos = new_pos
    mark_p = mtm_price or None
    if mark_p is None:
        if pos > 0.0:
            mark_p = bid
        elif pos < 0.0:
            mark_p = ask
        elif bid and ask:
            mark_p = (bid + ask) / 2.0
    unrealized = 0.0
    if mark_p is not None and avg is not None and pos != 0.0:
        if pos > 0.0:
            unrealized = (mark_p - avg) * pos
        else:
            unrealized = (avg - mark_p) * (-pos)
    return realized + unrealized


def test_pnl_report_recompute_matches() -> None:
    sim = ExecutionSimulator()
    sim.set_market_snapshot(bid=100.0, ask=101.0)
    trades_log = []

    # Buy 1 unit
    sim.submit(ActionProto(action_type=ActionType.MARKET, volume_frac=1.0))
    rep1 = sim.pop_ready(ref_price=100.5)
    trades_log.extend(rep1.trades)
    total1 = _recompute_total(trades_log, rep1.bid, rep1.ask, rep1.mtm_price)
    assert rep1.realized_pnl + rep1.unrealized_pnl == pytest.approx(total1, abs=1e-9)

    # Sell 1 unit after price moves higher
    sim.set_market_snapshot(bid=102.0, ask=103.0)
    sim.submit(ActionProto(action_type=ActionType.MARKET, volume_frac=-1.0))
    rep2 = sim.pop_ready(ref_price=102.5)
    trades_log.extend(rep2.trades)
    total2 = _recompute_total(trades_log, rep2.bid, rep2.ask, rep2.mtm_price)
    assert rep2.realized_pnl + rep2.unrealized_pnl == pytest.approx(total2, abs=1e-9)
