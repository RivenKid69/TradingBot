from dataclasses import dataclass

from risk_guard import PortfolioLimitConfig, PortfolioLimitGuard


@dataclass
class SimpleOrder:
    symbol: str
    side: str
    quantity: float
    reduce_only: bool = False
    meta: dict | None = None


def _make_guard(config: PortfolioLimitConfig, positions: dict[str, float], prices: dict[str, float]) -> PortfolioLimitGuard:
    return PortfolioLimitGuard(
        config=config,
        get_positions=lambda: positions,
        get_price=lambda sym: prices.get(sym.upper(), prices.get(sym, None)),
        get_total_notional=None,
        get_equity=lambda: 1_000.0,
    )


def test_portfolio_guard_buffer_blocks_order():
    positions = {"BTC": 1.0}
    prices = {"BTC": 100.0}
    cfg = PortfolioLimitConfig(max_total_notional=150.0, exposure_buffer_frac=0.2)
    guard = _make_guard(cfg, positions, prices)
    order = SimpleOrder(symbol="BTC", side="BUY", quantity=0.5)

    approved, reason = guard.apply(ts_ms=0, symbol="BTC", decisions=[order])

    assert approved == []
    assert reason == "RISK_PORTFOLIO_LIMIT"


def test_portfolio_guard_without_buffer_allows_order():
    positions = {"BTC": 1.0}
    prices = {"BTC": 100.0}
    cfg = PortfolioLimitConfig(max_total_notional=150.0, exposure_buffer_frac=0.0)
    guard = _make_guard(cfg, positions, prices)
    order = SimpleOrder(symbol="BTC", side="BUY", quantity=0.5)

    approved, reason = guard.apply(ts_ms=0, symbol="BTC", decisions=[order])

    assert approved == [order]
    assert reason is None
