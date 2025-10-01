import threading
from types import SimpleNamespace
from typing import Optional

import pytest

import impl_binance_public


@pytest.fixture
def source(monkeypatch):
    monkeypatch.setattr(impl_binance_public, "websockets", SimpleNamespace())
    return impl_binance_public.BinancePublicBarSource(timeframe="1m")


def test_stream_bars_validations_and_streams(source):
    stream = source.stream_bars(["btcusdt"], interval_ms=1_000)
    with pytest.raises(ValueError, match="Timeframe mismatch"):
        next(stream)

    stream_empty = source.stream_bars([], interval_ms=source._interval_ms)
    with pytest.raises(ValueError, match="No symbols provided"):
        next(stream_empty)

    source._symbols = [s.lower() for s in ("BTCUSDT", "ETHUSDT")]
    assert source._streams() == ["btcusdt@kline_1m", "ethusdt@kline_1m"]


def test_stream_ticks_returns_empty_iterator(source):
    assert list(source.stream_ticks(["btcusdt"])) == []


def test_close_joins_background_thread(source):
    joined = threading.Event()

    class DummyThread:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout: Optional[float] = None) -> None:
            joined.set()

    dummy_thread = DummyThread()
    source._thr = dummy_thread

    source.close()

    assert source._stop.is_set()
    assert joined.is_set()
