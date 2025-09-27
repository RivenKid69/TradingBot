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
from compat_shims import sim_report_dict_to_core_exec_reports

ActionProto = exec_mod.ActionProto
ActionType = exec_mod.ActionType
ExecutionSimulator = exec_mod.ExecutionSimulator
VWAPExecutor = algos_mod.VWAPExecutor
MarketOpenH1Executor = algos_mod.MarketOpenH1Executor


class DummyQuantizer:
    def quantize_qty(self, symbol, qty):
        return float(qty)

    def quantize_price(self, symbol, price):
        return float(price)

    def clamp_notional(self, symbol, ref_price, qty):
        return float(qty)

    def check_percent_price_by_side(self, symbol, side, price, ref_price):
        return True

    def attach_to(self, sim, *, strict=True, enforce_percent_price_by_side=True):
        sim.quantizer = self
        setattr(sim, "enforce_ppbs", enforce_percent_price_by_side)
        setattr(sim, "strict_filters", strict)


class DummyFees:
    def compute(self, *, side, price, qty, liquidity, **_):
        return float(price) * float(qty) * 0.001

    def attach_to(self, sim):
        sim.fees = self

    def get_fee_bps(self, symbol, is_maker):
        return 0.0 if is_maker else 10.0


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

    def pre_trade_adjust(
        self,
        *,
        ts_ms,
        side,
        intended_qty,
        price,
        position_qty,
        total_notional=None,
        equity=None,
    ):
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

    def attach_to(self, sim):
        sim.risk = self


@pytest.fixture
def base_sim():
    sim = ExecutionSimulator()
    sim.set_symbol("BTCUSDT")
    sim.set_quantizer(DummyQuantizer())
    DummyFees().attach_to(sim)
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
    assert not rep.trades
    assert rep.status == "PENDING_NEXT_BAR"

    rep_next = sim.run_step(
        ts=3_600_000,
        ref_price=105.0,
        bid=104.0,
        ask=106.0,
        vol_factor=1.0,
        liquidity=1.0,
        actions=[],
    )
    assert len(rep_next.trades) == 1
    trade = rep_next.trades[0]
    assert trade.ts == 3_600_000
    assert trade.price == pytest.approx(105.0)
    assert rep_next.fee_total == pytest.approx(trade.price * trade.qty * 0.001)
    assert rep_next.position_qty == pytest.approx(trade.qty)


def test_execution_simulator_close_lag_applied():
    sim = ExecutionSimulator(filters_path=None, close_lag_ms=2000)
    sim.set_symbol("BTCUSDT")
    sim.set_quantizer(DummyQuantizer())
    DummyFees().attach_to(sim)
    sim.risk = DummyRisk()
    sim.set_ref_price(100.0)

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)
    report = sim.run_step(
        ts=1_000_000,
        ref_price=100.0,
        bid=99.5,
        ask=100.5,
        liquidity=1.0,
        actions=[(ActionType.MARKET, proto)],
    )

    assert not report.trades
    assert report.status == "WAITING_CLOSE_LAG"
    assert report.reason == {
        "code": "WAITING_CLOSE_LAG",
        "lag_ms": 2000,
        "ready_ts": 1_002_000,
    }

    release = sim.run_step(
        ts=1_002_000,
        ref_price=100.0,
        bid=99.5,
        ask=100.5,
        liquidity=0.0,
        actions=[],
    )

    assert release.trades
    assert all(t.ts == 1_002_000 for t in release.trades)


def test_run_step_reports_new_order_position_alignment(base_sim):
    sim = base_sim
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)
    report = sim.run_step(
        ts=1_000,
        ref_price=100.0,
        bid=99.5,
        ask=100.5,
        liquidity=1.0,
        actions=[(ActionType.MARKET, proto)],
    )

    assert len(report.new_order_ids) == len(report.new_order_pos)
    assert report.new_order_pos == [0] * len(report.new_order_ids)


def test_run_step_market_orders_do_not_emit_new_order_ids(base_sim):
    sim = base_sim
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)
    report = sim.run_step(
        ts=1_000,
        ref_price=100.0,
        bid=99.5,
        ask=100.5,
        liquidity=1.0,
        actions=[(ActionType.MARKET, proto)],
    )

    assert report.new_order_ids == []
    assert report.new_order_pos == []


def test_limit_order_reports_new_order_position_alignment(base_sim):
    sim = base_sim
    proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=1.0, abs_price=99.0)
    oid = sim.submit(proto)
    report = sim.pop_ready(ref_price=100.0)

    assert report.new_order_ids == [oid]
    assert report.new_order_pos == [0]


def test_mkt_open_next_h1_profile_close_lag(base_sim):
    sim = base_sim
    sim.close_lag_ms = 1500
    sim._timing_close_lag_ms = 1500
    sim.set_execution_profile("MKT_OPEN_NEXT_H1")
    sim.set_next_open_price(105.0)

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)

    first = sim.run_step(
        ts=1_800_000,
        ref_price=100.0,
        bid=99.0,
        ask=101.0,
        vol_factor=1.0,
        liquidity=1.0,
        actions=[(ActionType.MARKET, proto)],
    )

    assert first.status == "WAITING_CLOSE_LAG"
    assert not first.trades

    pending = sim.run_step(
        ts=1_801_500,
        ref_price=100.0,
        bid=99.0,
        ask=101.0,
        vol_factor=1.0,
        liquidity=0.0,
        actions=[],
    )

    assert pending.status == "PENDING_NEXT_BAR"
    assert not pending.trades

    wait_fill = sim.run_step(
        ts=3_600_000,
        ref_price=105.0,
        bid=104.0,
        ask=106.0,
        vol_factor=1.0,
        liquidity=1.0,
        actions=[],
    )

    assert wait_fill.status == "FILLED_NEXT_BAR"
    assert len(wait_fill.trades) == 1
    assert wait_fill.trades[0].ts >= 1_800_000 + sim.close_lag_ms


def test_vwap_current_h1_profile(base_sim):
    sim = base_sim
    sim.set_execution_profile("VWAP_CURRENT_H1")
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=2.0)
    rep = sim.run_step(
        ts=1_800_000,
        ref_price=200.0,
        bid=199.0,
        ask=201.0,
        vol_factor=1.0,
        liquidity=2.0,
        bar_timeframe_ms=3_600_000,
        actions=[(ActionType.MARKET, proto)],
    )
    assert len(rep.trades) == VWAPExecutor().fallback_parts
    trade_ts = [t.ts for t in rep.trades]
    assert trade_ts == sorted(trade_ts)
    assert trade_ts[0] == 1_800_000
    assert trade_ts[-1] == 3_600_000
    qty_total = sum(t.qty for t in rep.trades)
    assert rep.position_qty == pytest.approx(qty_total)
    assert rep.fee_total == pytest.approx(sum(t.price * t.qty for t in rep.trades) * 0.001)
def test_limit_mid_bps_params_build(base_sim):
    sim = base_sim
    params = {"limit_offset_bps": 50, "ttl_steps": 7, "tif": "IOC"}
    sim.set_ref_price(100.0)
    sim.set_execution_profile("LIMIT_MID_BPS", params)
    sim.latency = None
    sim.latency_steps = 0

    from decimal import Decimal
    from core_models import Order, Side, OrderType
    import impl_sim_executor as impl_mod
    from core_config import ExecutionParams, ExecutionProfile

    class _AP:
        def __init__(self, action_type, volume_frac, price_offset_ticks=0, ttl_steps=0, abs_price=None, tif="GTC", client_tag=None):
            self.action_type = action_type
            self.volume_frac = volume_frac
            self.price_offset_ticks = price_offset_ticks
            self.ttl_steps = ttl_steps
            self.abs_price = abs_price
            self.tif = tif
            self.client_tag = client_tag

    impl_mod.ActionProto = _AP
    SimExecutor = impl_mod.SimExecutor

    executor = SimExecutor(
        sim,
        symbol="BTCUSDT",
        quantizer=DummyQuantizer(),
        risk=DummyRisk(),
        fees=DummyFees(),
    )
    executor._exec_profile = ExecutionProfile.LIMIT_MID_BPS
    executor._exec_params = ExecutionParams(**params)

    order = Order(
        ts=0,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1"),
    )

    atype, proto = executor._order_to_action(order)
    assert atype == ActionType.LIMIT
    assert proto.abs_price == pytest.approx(100.0 * (1 - 0.005))
    assert proto.tif == "IOC"
    assert proto.ttl_steps == 7

    sim.set_market_snapshot(bid=98.0, ask=99.0, liquidity=1.0)


def test_sim_executor_close_lag_from_run_config():
    sim = ExecutionSimulator(filters_path=None)
    sim.set_ref_price(100.0)
    sim.latency = None
    sim.latency_steps = 0

    from types import SimpleNamespace
    run_config = SimpleNamespace(
        run_id="close-lag-test",
        timing=SimpleNamespace(close_lag_ms=1500, enforce_closed_bars=True),
    )

    import impl_sim_executor as impl_mod
    executor = impl_mod.SimExecutor(
        sim,
        symbol="BTCUSDT",
        quantizer=DummyQuantizer(),
        risk=DummyRisk(),
        fees=DummyFees(),
        run_config=run_config,
    )
    sim.latency = None
    sim.latency_steps = 0

    from decimal import Decimal
    from core_models import Order, Side, OrderType

    order = Order(
        ts=1_000_000,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1"),
    )

    report = executor.execute(order)
    assert report.exec_status == "NEW"
    follow = sim.run_step(
        ts=order.ts + run_config.timing.close_lag_ms,
        ref_price=100.0,
        bid=None,
        ask=None,
        vol_factor=None,
        liquidity=None,
        actions=[],
    )
    follow_reports = sim_report_dict_to_core_exec_reports(
        follow.to_dict(),
        symbol="BTCUSDT",
        run_id=executor._run_id,
        client_order_id=str(getattr(order, "client_order_id", "") or ""),
    )
    assert follow_reports
    assert follow_reports[0].ts == order.ts + run_config.timing.close_lag_ms
    assert follow_reports[0].quantity == Decimal("1")


def test_executor_switching():
    sim = ExecutionSimulator(execution_profile="VWAP_CURRENT_H1")
    assert isinstance(sim._executor, VWAPExecutor)
    sim.set_execution_profile("MKT_OPEN_NEXT_H1")
    assert isinstance(sim._executor, MarketOpenH1Executor)
