import importlib.util
import json
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pytest


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

spec_exec = importlib.util.spec_from_file_location(
    "execution_sim", BASE_DIR / "execution_sim.py"
)
exec_mod = importlib.util.module_from_spec(spec_exec)
sys.modules["execution_sim"] = exec_mod
spec_exec.loader.exec_module(exec_mod)

ActionProto = exec_mod.ActionProto
ActionType = exec_mod.ActionType
ExecutionSimulator = exec_mod.ExecutionSimulator


@dataclass
class _CompatReport:
    trades: list = field(default_factory=list)
    cancelled_ids: list = field(default_factory=list)
    cancelled_reasons: dict = field(default_factory=dict)
    new_order_ids: list = field(default_factory=list)
    fee_total: float = 0.0
    new_order_pos: list = field(default_factory=list)
    funding_cashflow: float = 0.0
    funding_events: list = field(default_factory=list)
    position_qty: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    equity: float = 0.0
    mark_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    mtm_price: float = 0.0
    risk_events: list = field(default_factory=list)
    risk_paused_until_ms: int = 0
    spread_bps: Optional[float] = None
    vol_factor: Optional[float] = None
    liquidity: Optional[float] = None
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_timeout_ratio: float = 0.0
    execution_profile: str = ""
    market_regime: Any | None = None
    vol_raw: Optional[Dict[str, float]] = None
    status: str = ""
    reason: Optional[Dict[str, Any]] = None


exec_mod.SimStepReport = _CompatReport  # type: ignore[attr-defined]
exec_mod.ExecReport = _CompatReport  # type: ignore[attr-defined]


class _StubRisk:
    def __init__(self) -> None:
        self.paused_until_ms = 0

    def pre_trade_adjust(
        self,
        *,
        ts_ms: int,
        side: str,
        intended_qty: float,
        price: float,
        position_qty: float,
        total_notional=None,
    ) -> float:
        return float(intended_qty)

    def pop_events(self):
        return []

    def can_send_order(self, ts_ms: int) -> bool:
        return True

    def on_new_order(self, ts_ms: int) -> None:
        pass

    def on_mark(self, ts_ms: int, equity: float) -> None:
        pass

    def _emit(self, ts_ms: int, code: str, message: str, **data) -> None:
        pass


def _write_bar_capacity(tmp_path: pathlib.Path, symbol: str, *, per_bar: float, timeframe_ms: int) -> pathlib.Path:
    bars_per_day = 86_400_000.0 / float(timeframe_ms if timeframe_ms > 0 else 1)
    daily_base = float(per_bar) * bars_per_day
    payload = {str(symbol).upper(): daily_base}
    adv_path = tmp_path / "adv_capacity.json"
    adv_path.write_text(json.dumps(payload))
    return adv_path


def _make_sim_with_capacity(tmp_path: pathlib.Path, per_bar: float) -> ExecutionSimulator:
    sim = ExecutionSimulator(symbol="BTCUSDT", filters_path=None)
    timeframe_ms = int(sim.step_ms if sim.step_ms > 0 else 1)
    adv_path = _write_bar_capacity(tmp_path, sim.symbol, per_bar=per_bar, timeframe_ms=timeframe_ms)
    sim.set_bar_capacity_base_config(
        enabled=True,
        timeframe_ms=timeframe_ms,
        adv_base_path=str(adv_path),
    )
    sim.risk = _StubRisk()
    return sim


def test_pop_ready_clamps_to_remaining_bar_capacity(tmp_path: pathlib.Path) -> None:
    sim = _make_sim_with_capacity(tmp_path, per_bar=2.0)
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=5.0)
    sim.submit(proto, now_ts=123_000)

    report = sim.pop_ready(now_ts=123_000, ref_price=100.0)

    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.qty == pytest.approx(2.0)
    assert trade.status == "FILLED"
    assert trade.used_base_before == pytest.approx(0.0)
    assert trade.used_base_after == pytest.approx(2.0)
    assert report.cancelled_ids == []
    assert report.cancelled_reasons == {}
    assert sim._used_base_in_bar[sim.symbol] == pytest.approx(2.0)


def test_run_step_emits_cancel_when_capacity_exhausted(tmp_path: pathlib.Path) -> None:
    sim = _make_sim_with_capacity(tmp_path, per_bar=2.0)
    first = ActionProto(action_type=ActionType.MARKET, volume_frac=2.0)
    second = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)

    report = sim.run_step(
        ts=456_000,
        ref_price=100.0,
        actions=[
            (ActionType.MARKET, first),
            (ActionType.MARKET, second),
        ],
    )

    assert [t.client_order_id for t in report.trades] == [1, 2]
    assert report.trades[0].qty == pytest.approx(2.0)
    assert report.trades[0].status == "FILLED"
    assert report.trades[0].used_base_before == pytest.approx(0.0)
    assert report.trades[0].used_base_after == pytest.approx(2.0)

    cancel_trade = report.trades[1]
    assert cancel_trade.qty == pytest.approx(0.0)
    assert cancel_trade.status == "CANCELED"
    assert cancel_trade.used_base_before == pytest.approx(2.0)
    assert cancel_trade.used_base_after == pytest.approx(2.0)
    assert report.cancelled_ids == [cancel_trade.client_order_id]
    assert report.cancelled_reasons[cancel_trade.client_order_id] == "BAR_CAPACITY_BASE"
    assert sim._used_base_in_bar[sim.symbol] == pytest.approx(2.0)


def test_limit_taker_execution_clamped_by_bar_capacity(
    tmp_path: pathlib.Path,
) -> None:
    sim = _make_sim_with_capacity(tmp_path, per_bar=2.0)
    sim._last_bid = 99.0
    sim._last_ask = 100.0
    sim._last_liquidity = 10.0
    proto = ActionProto(
        action_type=ActionType.LIMIT,
        volume_frac=3.0,
        abs_price=101.0,
    )

    cid = sim.submit(proto, now_ts=789_000)

    report = sim.pop_ready(now_ts=789_000, ref_price=100.0)

    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.qty == pytest.approx(2.0)
    assert trade.used_base_before == pytest.approx(0.0)
    assert trade.used_base_after == pytest.approx(2.0)
    assert trade.cap_base_per_bar == pytest.approx(2.0)
    assert trade.fill_ratio == pytest.approx(2.0 / 3.0)
    assert trade.capacity_reason == "BAR_CAPACITY_BASE"
    assert trade.exec_status == "PARTIAL"
    assert sim._used_base_in_bar[sim.symbol] == pytest.approx(2.0)
    assert report.new_order_ids == [cid]
    assert report.cancelled_ids == []
