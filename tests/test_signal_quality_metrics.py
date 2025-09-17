from __future__ import annotations

from decimal import Decimal
import os
import sys

sys.path.append(os.getcwd())

import pytest

from feature_pipe import FeaturePipe, SignalQualityMetrics
from transformers import FeatureSpec
from core_models import Bar


def _make_bar(
    ts: int,
    price: str,
    *,
    symbol: str = "BTCUSDT",
    volume_quote: str | None = None,
) -> Bar:
    px = Decimal(price)
    vq = Decimal(volume_quote) if volume_quote is not None else None
    return Bar(
        ts=ts,
        symbol=symbol,
        open=px,
        high=px,
        low=px,
        close=px,
        volume_quote=vq,
    )


def test_signal_quality_metrics_handles_invalid_close() -> None:
    metrics = SignalQualityMetrics(sigma_window=2, vol_median_window=2)

    metrics.update("BTCUSDT", _make_bar(1, "100", volume_quote="10"))
    second = metrics.update("BTCUSDT", _make_bar(2, "101", volume_quote="11"))
    assert second.current_sigma is not None

    third = metrics.update("BTCUSDT", _make_bar(3, "102", volume_quote="12"))
    assert third.window_ready is True

    snapshot_before = metrics.latest["BTCUSDT"]

    invalid = metrics.update("BTCUSDT", _make_bar(4, "NaN", volume_quote="13"))
    assert invalid == snapshot_before
    assert metrics.latest["BTCUSDT"] == snapshot_before


def test_feature_pipe_skips_invalid_bar_but_preserves_metrics() -> None:
    metrics = SignalQualityMetrics(sigma_window=2, vol_median_window=2)
    pipe = FeaturePipe(FeatureSpec(lookbacks_prices=[2]), metrics=metrics)

    valid_bar = _make_bar(1, "100", volume_quote="10")
    feats = pipe.update(valid_bar)
    assert feats["ref_price"] == pytest.approx(100.0)
    snapshot_before = pipe.signal_quality["BTCUSDT"]

    skipped = pipe.update(_make_bar(2, "NaN", volume_quote="11"))
    assert skipped == {}
    assert pipe.signal_quality["BTCUSDT"] == snapshot_before


def test_signal_quality_reset_symbol_clears_state() -> None:
    metrics = SignalQualityMetrics(sigma_window=2, vol_median_window=2)

    metrics.update("BTCUSDT", _make_bar(1, "100", volume_quote="10"))
    metrics.update("BTCUSDT", _make_bar(2, "101", volume_quote="11"))
    assert "BTCUSDT" in metrics.latest

    metrics.reset_symbol("BTCUSDT")
    assert "BTCUSDT" not in metrics.latest


def test_feature_pipe_can_skip_metric_update() -> None:
    metrics = SignalQualityMetrics(sigma_window=2, vol_median_window=2)
    pipe = FeaturePipe(FeatureSpec(lookbacks_prices=[2]), metrics=metrics)

    bar1 = _make_bar(1, "100", volume_quote="10")
    pipe.update(bar1)
    snapshot = pipe.signal_quality["BTCUSDT"]

    bar2 = _make_bar(2, "110", volume_quote="11")
    pipe.update(bar2, skip_metrics=True)

    assert pipe.signal_quality["BTCUSDT"] == snapshot
