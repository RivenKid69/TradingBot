import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("execution_sim", ROOT / "execution_sim.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
ExecutionSimulator = module.ExecutionSimulator
ActionProto = module.ActionProto
ActionType = module.ActionType

def recompute_total(trades, bid, ask, mtm_price):
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
                else:
                    if pos == 0.0:
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
                else:
                    if pos == 0.0:
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

def main() -> None:
    sim = ExecutionSimulator()
    class DummyExec:
        def plan_market(self, now_ts_ms, side, target_qty, snapshot):
            return [type("C", (), {"ts_offset_ms": 0, "qty": target_qty, "liquidity_hint": None})()]
    sim._executor = DummyExec()
    sim.set_market_snapshot(bid=100.0, ask=101.0)

    trades_log = []

    sim.submit(ActionProto(action_type=ActionType.MARKET, volume_frac=1.0))
    report1 = sim.pop_ready(ref_price=100.5)
    trades_log.extend(report1.trades)
    total1 = recompute_total(trades_log, report1.bid, report1.ask, report1.mark_price)
    assert abs(report1.realized_pnl + report1.unrealized_pnl - total1) < 1e-9

    sim.set_market_snapshot(bid=102.0, ask=103.0)
    sim.submit(ActionProto(action_type=ActionType.MARKET, volume_frac=-1.0))
    report2 = sim.pop_ready(ref_price=102.5)
    trades_log.extend(report2.trades)
    total2 = recompute_total(trades_log, report2.bid, report2.ask, report2.mark_price)
    assert abs(report2.realized_pnl + report2.unrealized_pnl - total2) < 1e-9

if __name__ == "__main__":
    main()
