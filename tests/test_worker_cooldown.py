import logging
from types import SimpleNamespace

import pytest

from core_config import ResolvedTurnoverCaps, ResolvedTurnoverLimit
from service_signal_runner import _Worker


def _make_bar_worker() -> _Worker:
    worker = _Worker(
        fp=SimpleNamespace(),
        policy=SimpleNamespace(),
        logger=logging.getLogger("test_cooldown"),
        executor=SimpleNamespace(),
        guards=None,
        enforce_closed_bars=True,
        throttle_cfg=None,
        execution_mode="bar",
        cooldown_settings={
            "cooldown_bars": 2,
            "turnover_cap": 0.4,
            "cooldown_small_delta": 0.05,
        },
    )
    emissions: list[tuple[str, int]] = []

    def _emit(order: object, symbol: str, bar_ts: int) -> bool:
        emissions.append((symbol, bar_ts))
        return True

    worker._emit = _emit  # type: ignore[method-assign]
    worker._last_bar_ts["BTCUSDT"] = 1
    return worker, emissions


def _make_order(symbol: str, target: float) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, meta={"payload": {"target_weight": target}})


def test_small_deltas_suppressed_during_cooldown() -> None:
    worker, emissions = _make_bar_worker()
    symbol = "BTCUSDT"

    big_order = _make_order(symbol, 0.5)
    result = worker.publish_decision(big_order, symbol, 1)
    assert result.action == "pass"
    assert worker._weights[symbol] == pytest.approx(0.5)
    assert worker._symbol_cooldowns[symbol] == 2

    worker._update_queue_metrics()
    status = worker._last_status_payload
    assert status["symbol_cooldowns"]["remaining"][symbol] == 2

    small_order = _make_order(symbol, 0.52)
    res = worker.publish_decision(small_order, symbol, 2)
    assert res.action == "drop"
    assert worker._weights[symbol] == pytest.approx(0.5)
    worker._advance_symbol_cooldown(symbol, 2, new_bar=True)
    worker._last_bar_ts[symbol] = 2

    smaller_order = _make_order(symbol, 0.51)
    res = worker.publish_decision(smaller_order, symbol, 3)
    assert res.action == "drop"
    worker._advance_symbol_cooldown(symbol, 3, new_bar=True)
    worker._last_bar_ts[symbol] = 3
    assert symbol not in worker._symbol_cooldowns

    worker._update_queue_metrics()
    post_status = worker._last_status_payload
    assert post_status.get("symbol_cooldowns", {}).get("count", 0) == 0

    final_order = _make_order(symbol, 0.52)
    res = worker.publish_decision(final_order, symbol, 4)
    assert res.action == "pass"
    assert worker._weights[symbol] == pytest.approx(0.52)

    # Only the initial large trade and the final adjustment should emit
    assert emissions == [(symbol, 1), (symbol, 4)]


def test_daily_turnover_limits_infer_missing_turnover() -> None:
    symbol = "BTCUSDT"
    caps = ResolvedTurnoverCaps(
        per_symbol=ResolvedTurnoverLimit(daily_usd=1_000.0),
        portfolio=ResolvedTurnoverLimit(daily_usd=1_000.0),
    )
    executor = SimpleNamespace(turnover_caps=SimpleNamespace(resolve=lambda: caps))
    worker = _Worker(
        fp=SimpleNamespace(),
        policy=SimpleNamespace(),
        logger=logging.getLogger("test_turnover_fallback"),
        executor=executor,
        guards=None,
        enforce_closed_bars=True,
        throttle_cfg=None,
        execution_mode="bar",
        portfolio_equity=1_000_000.0,
    )

    order = SimpleNamespace(symbol=symbol, meta={"payload": {"target_weight": 0.5}})

    adjusted = worker._apply_daily_turnover_limits([order], symbol, ts_ms=1)
    assert adjusted == [order]

    payload = worker._extract_signal_payload(order)
    assert payload["target_weight"] == pytest.approx(0.001)

    turnover_info = order.meta["daily_turnover"]
    assert turnover_info["clamped"] is True
    assert turnover_info["requested_usd"] == pytest.approx(500_000.0)
    assert turnover_info["executed_usd"] == pytest.approx(1_000.0)
    assert turnover_info["headroom_before_usd"] == pytest.approx(1_000.0)

    decision_meta = order.meta["decision"]
    assert decision_meta["turnover_usd"] == pytest.approx(1_000.0)
    assert decision_meta["daily_turnover_clamped"] is True


def test_order_equity_override_used_for_weight_and_turnover() -> None:
    symbol = "BTCUSDT"
    caps = ResolvedTurnoverCaps(
        per_symbol=ResolvedTurnoverLimit(daily_usd=60.0),
        portfolio=ResolvedTurnoverLimit(daily_usd=60.0),
    )
    executor = SimpleNamespace(turnover_caps=SimpleNamespace(resolve=lambda: caps))
    worker = _Worker(
        fp=SimpleNamespace(),
        policy=SimpleNamespace(),
        logger=logging.getLogger("test_turnover_equity_override"),
        executor=executor,
        guards=None,
        enforce_closed_bars=True,
        throttle_cfg=None,
        execution_mode="bar",
        portfolio_equity=1_000_000.0,
    )
    worker._last_prices[symbol] = 1.0

    first_order = SimpleNamespace(
        symbol=symbol,
        meta={"payload": {"target_weight": 0.5, "equity_usd": 1_000.0}},
    )
    worker._commit_exposure(first_order)

    assert worker._positions[symbol] == pytest.approx(500.0)
    assert worker._symbol_equity.get(symbol) == pytest.approx(1_000.0)

    second_order = SimpleNamespace(
        symbol=symbol,
        meta={"payload": {"target_weight": 0.6, "equity_usd": 1_000.0}},
    )
    adjusted = worker._apply_daily_turnover_limits([second_order], symbol, ts_ms=1)
    assert adjusted == [second_order]

    payload = worker._extract_signal_payload(second_order)
    assert payload["target_weight"] == pytest.approx(0.56)

    turnover_info = second_order.meta["daily_turnover"]
    assert turnover_info["requested_usd"] == pytest.approx(100.0)
    assert turnover_info["executed_usd"] == pytest.approx(60.0)

    decision_meta = second_order.meta["decision"]
    assert decision_meta["turnover_usd"] == pytest.approx(60.0)
    assert decision_meta.get("daily_turnover_clamped", False) is True

