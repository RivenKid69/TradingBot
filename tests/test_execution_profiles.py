import importlib.util
import pathlib
import sys

import pytest

# Dynamically import modules from repository root
# Ensure repository root is on the import path so that execution_sim can
# resolve its sibling modules (execution_algos, etc.).
base = pathlib.Path(__file__).resolve().parent.parent
if str(base) not in sys.path:
    sys.path.append(str(base))

spec_exec = importlib.util.spec_from_file_location("execution_sim", base / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec_exec)
sys.modules["execution_sim"] = exec_mod
spec_exec.loader.exec_module(exec_mod)

import execution_algos as algos_mod

ActionProto = exec_mod.ActionProto
ActionType = exec_mod.ActionType
ExecutionSimulator = exec_mod.ExecutionSimulator
VWAPExecutor = algos_mod.VWAPExecutor
MarketOpenH1Executor = algos_mod.MarketOpenH1Executor
MidOffsetLimitExecutor = algos_mod.MidOffsetLimitExecutor


class DummyQuantizer:
    def quantize_qty(self, symbol, qty):
        return float(qty)

    def quantize_price(self, symbol, price):
        return float(price)

    def clamp_notional(self, symbol, ref_price, qty):
        return float(qty)

    def check_percent_price_by_side(self, symbol, side, price, ref_price):
        return True


class DummyFees:
    def compute(self, *, side, price, qty, liquidity):
        return float(price) * float(qty) * 0.001


from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class DummyRiskEvent:
    ts_ms: int
    code: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


class DummyRisk:
    def __init__(self):
        self.events = []
        self.paused_until_ms = 0

    def pre_trade_adjust(self, *, ts_ms, side, intended_qty, price, position_qty):
        return float(intended_qty)

    def pop_events(self):
        ev = list(self.events)
        self.events.clear()
        return ev

    def can_send_order(self, ts_ms):
        return True

    def on_new_order(self, ts_ms):
        pass

    def on_mark(self, ts_ms, equity):
        pass

    def _emit(self, ts_ms, code, message, **data):
        self.events.append(DummyRiskEvent(ts_ms=int(ts_ms), code=str(code), message=str(message), data=dict(data)))


@pytest.fixture
def base_sim():
    sim = ExecutionSimulator()
    sim.set_symbol("BTCUSDT")
    sim.set_quantizer(DummyQuantizer())
    sim.fees = DummyFees()
    sim.risk = DummyRisk()
    return sim


def test_mkt_open_next_h1_profile(base_sim):
    sim = base_sim
    sim.set_execution_profile("MKT_OPEN_NEXT_H1")
    sim.set_next_open_price(105.0)
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)
    rep = sim.run_step(
        ts=1_800_000,
        ref_price=100.0,
        bid=99.0,
        ask=101.0,
        vol_factor=1.0,
        liquidity=1.0,
        actions=[(ActionType.MARKET, proto)],
    )
    assert len(rep.trades) == 1
    trade = rep.trades[0]
    assert trade.ts == 3_600_000
    assert trade.price == pytest.approx(105.0)
    assert rep.fee_total == pytest.approx(trade.price * trade.qty * 0.001)
    assert rep.position_qty == pytest.approx(trade.qty)


def test_vwap_current_h1_profile(base_sim):
    sim = ExecutionSimulator(execution_config={"algo": "VWAP"}, execution_profile="VWAP_CURRENT_H1")
    sim.set_symbol("BTCUSDT")
    sim.set_quantizer(DummyQuantizer())
    sim.fees = DummyFees()
    sim.risk = DummyRisk()
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=2.0)
    rep = sim.run_step(
        ts=1_800_000,
        ref_price=200.0,
        bid=199.0,
        ask=201.0,
        vol_factor=1.0,
        liquidity=2.0,
        actions=[(ActionType.MARKET, proto)],
    )
    assert len(rep.trades) == 1
    trade = rep.trades[0]
    assert trade.ts == 3_600_000
    assert trade.price == pytest.approx(200.0)
    assert rep.fee_total == pytest.approx(trade.price * trade.qty * 0.001)
    assert rep.position_qty == pytest.approx(trade.qty)


def test_limit_mid_bps_profile(base_sim):
    sim = base_sim
    sim.set_execution_profile("LIMIT_MID_BPS")
    sim.set_market_snapshot(bid=99.0, ask=101.0, liquidity=5.0)
    executor = MidOffsetLimitExecutor(offset_bps=200)
    built = executor.build_action(side="BUY", qty=1.0, snapshot={"mid": 100.0})

    class _Proto:
        def __init__(self, price):
            self.action_type = ActionType.LIMIT
            self.volume_frac = 1.0
            self.abs_price = price
            self.ttl_steps = 0
            self.tif = "GTC"

    proto = _Proto(built["abs_price"])
    sim.submit(proto)
    rep = sim.pop_ready(ref_price=100.0)
    assert len(rep.trades) == 1
    trade = rep.trades[0]
    assert trade.price == pytest.approx(101.0)
    assert rep.fee_total == pytest.approx(trade.price * trade.qty * 0.001)
    assert rep.position_qty == pytest.approx(trade.qty)


def test_executor_switching():
    sim = ExecutionSimulator(execution_config={"algo": "VWAP"}, execution_profile="VWAP_CURRENT_H1")
    assert isinstance(sim._executor, VWAPExecutor)
    sim.set_execution_profile("MKT_OPEN_NEXT_H1")
    assert isinstance(sim._executor, MarketOpenH1Executor)
