import math
from decimal import Decimal

from transformers import FeatureSpec

from core_models import Bar
from feature_pipe import FeaturePipe


def _make_bar(ts: int, close: str) -> Bar:
    price = Decimal(close)
    return Bar(
        ts=ts,
        symbol="BTCUSDT",
        open=price,
        high=price,
        low=price,
        close=price,
        volume_base=Decimal("1"),
        is_final=True,
    )


def test_feature_pipe_tracks_returns_sigma_and_warmup():
    pipe = FeaturePipe(FeatureSpec(lookbacks_prices=[1]), sigma_window=3, min_sigma_periods=2)
    bars = [
        _make_bar(0, "100"),
        _make_bar(60_000, "101"),
        _make_bar(120_000, "102"),
    ]
    for bar in bars:
        pipe.update(bar)

    snapshot = pipe.get_market_metrics("BTCUSDT")
    assert snapshot is not None
    assert snapshot.bar_count == 3
    assert snapshot.window_ready is True

    expected_returns = [0.01, 102.0 / 101.0 - 1.0]
    expected_mean = sum(expected_returns) / len(expected_returns)
    expected_var = sum((r - expected_mean) ** 2 for r in expected_returns) / (len(expected_returns) - 1)
    expected_sigma = math.sqrt(expected_var)

    assert snapshot.ret_last is not None
    assert math.isclose(snapshot.ret_last, expected_returns[-1], rel_tol=1e-12)
    assert snapshot.sigma is not None
    assert math.isclose(snapshot.sigma, expected_sigma, rel_tol=1e-12)


def test_feature_pipe_records_spread_with_ttl_expiry():
    pipe = FeaturePipe(FeatureSpec(lookbacks_prices=[1]), sigma_window=2, spread_ttl_ms=500)
    first = _make_bar(0, "100")
    pipe.update(first)
    pipe.record_spread("BTCUSDT", bid=100.0, ask=100.1, ts_ms=0)

    snap_initial = pipe.get_market_metrics("BTCUSDT")
    assert snap_initial is not None and snap_initial.spread_bps is not None
    assert snap_initial.spread_bps > 0

    second = _make_bar(2_000, "100")
    pipe.update(second)
    snap_after = pipe.get_market_metrics("BTCUSDT")
    assert snap_after is not None
    assert snap_after.last_bar_ts == 2_000
    assert snap_after.spread_bps is None
