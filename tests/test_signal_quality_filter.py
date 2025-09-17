import os
import sys
from decimal import Decimal
import types

sys.path.append(os.getcwd())

from core_models import Bar
from feature_pipe import SignalQualityMetrics
from service_signal_runner import SignalQualityConfig  # type: ignore
from service_signal_runner import _Worker  # type: ignore


class DummyFeaturePipe:
    def __init__(self, metrics: SignalQualityMetrics | None = None) -> None:
        self.metrics = metrics
        self.signal_quality: dict[str, object] = {}

    def warmup(self) -> None:
        return None

    def update(self, bar: Bar, *, skip_metrics: bool = False) -> dict[str, float]:
        if self.metrics is not None and not skip_metrics:
            snapshot = self.metrics.update(bar.symbol, bar)
            self.signal_quality[bar.symbol] = snapshot
            self.signal_quality[bar.symbol.upper()] = snapshot
        return {"ref_price": float(bar.close)}


class DummyPolicy:
    def __init__(self) -> None:
        self.call_count = 0

    def decide(self, feats, ctx):
        self.call_count += 1
        return []


class DummyLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, tuple, dict]] = []

    def info(self, msg, *args, **kwargs):
        self.messages.append((msg, args, kwargs))

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def _make_bar(ts: int, close: float, volume: float) -> Bar:
    price = Decimal(str(close))
    vol = Decimal(str(volume))
    return Bar(
        ts=ts,
        symbol="BTCUSDT",
        open=price,
        high=price,
        low=price,
        close=price,
        volume_quote=vol,
        is_final=True,
    )


def _make_worker(cfg: SignalQualityConfig, metrics: SignalQualityMetrics | None = None):
    fp = DummyFeaturePipe(metrics)
    policy = DummyPolicy()
    logger = DummyLogger()
    executor = types.SimpleNamespace(submit=lambda order: None)
    worker = _Worker(
        fp,
        policy,
        logger,
        executor,
        enforce_closed_bars=False,
        signal_quality_cfg=cfg,
    )
    return worker, fp, policy, logger


def test_signal_quality_filter_blocks_low_volume_before_policy() -> None:
    metrics = SignalQualityMetrics(sigma_window=2, vol_median_window=2)
    cfg = SignalQualityConfig(
        enabled=True,
        sigma_window=2,
        sigma_threshold=10.0,
        vol_median_window=2,
        vol_floor_frac=0.5,
        log_reason="QUALITY_BLOCK",
    )
    worker, _fp, policy, logger = _make_worker(cfg, metrics)

    bars = [
        _make_bar(1, 100.0, 100.0),
        _make_bar(2, 101.0, 110.0),
        _make_bar(3, 102.0, 120.0),
        _make_bar(4, 103.0, 1.0),
    ]

    for bar in bars[:-1]:
        worker.process(bar)

    assert policy.call_count == 1

    worker.process(bars[-1])

    assert policy.call_count == 1
    assert logger.messages
    msg, args, _ = logger.messages[-1]
    assert args == ()
    assert "DROP" in msg
    assert f"reason={cfg.log_reason}" in msg
    assert "detail=VOLUME_FLOOR" in msg
    assert "bar_close_at=" in msg


def test_signal_quality_filter_blocks_high_sigma() -> None:
    metrics = SignalQualityMetrics(sigma_window=2, vol_median_window=2)
    cfg = SignalQualityConfig(
        enabled=True,
        sigma_window=2,
        sigma_threshold=0.0001,
        vol_median_window=2,
        vol_floor_frac=0.0,
        log_reason="QUALITY_SIGMA",
    )
    worker, _fp, policy, logger = _make_worker(cfg, metrics)

    bars = [
        _make_bar(1, 100.0, 100.0),
        _make_bar(2, 200.0, 100.0),
        _make_bar(3, 10.0, 100.0),
    ]

    for bar in bars:
        worker.process(bar)

    assert policy.call_count == 0
    assert logger.messages
    msg, args, _ = logger.messages[-1]
    assert args == ()
    assert f"reason={cfg.log_reason}" in msg
    assert "detail=SIGMA_THRESHOLD" in msg


def test_signal_quality_filter_disabled_keeps_policy_path() -> None:
    metrics = SignalQualityMetrics(sigma_window=2, vol_median_window=2)
    cfg = SignalQualityConfig(
        enabled=False,
        sigma_window=2,
        sigma_threshold=0.0,
        vol_median_window=2,
        vol_floor_frac=1.0,
        log_reason="IGNORED",
    )
    worker, _fp, policy, logger = _make_worker(cfg, metrics)

    worker.process(_make_bar(1, 100.0, 100.0))

    assert policy.call_count == 1
    assert logger.messages == []


def test_signal_quality_filter_skips_logging_when_disabled_flag() -> None:
    metrics = SignalQualityMetrics(sigma_window=2, vol_median_window=2)
    cfg = SignalQualityConfig(
        enabled=True,
        sigma_window=2,
        sigma_threshold=10.0,
        vol_median_window=2,
        vol_floor_frac=0.5,
        log_reason=False,
    )
    worker, _fp, policy, logger = _make_worker(cfg, metrics)

    bars = [
        _make_bar(1, 100.0, 100.0),
        _make_bar(2, 101.0, 110.0),
        _make_bar(3, 102.0, 120.0),
        _make_bar(4, 103.0, 1.0),
    ]

    for bar in bars[:-1]:
        worker.process(bar)

    assert policy.call_count == 1

    worker.process(bars[-1])

    assert policy.call_count == 1
    assert logger.messages == []
