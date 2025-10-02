# -*- coding: utf-8 -*-
"""
impl_binance_public.py
Источник рыночных данных Binance Public WS. Источник выдаёт объекты `Bar` через `stream_bars`.
Работает синхронно через внутренний поток, который обслуживает asyncio + websockets.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Iterator, List, Optional, Sequence
import json
import queue
import threading
import time
import asyncio
from datetime import datetime, timezone
import logging

try:
    import websockets  # type: ignore
except Exception:
    websockets = None  # type: ignore

from core_models import Bar, Tick
from core_contracts import MarketDataSource
from config import DataDegradationConfig


_BINANCE_WS = "wss://stream.binance.com:9443/stream"


logger = logging.getLogger(__name__)


@dataclass
class BinanceWSConfig:
    reconnect_backoff_s: float = 1.0
    reconnect_backoff_max_s: float = 30.0
    ping_interval_s: float = 10.0
    vendor: str = "binance"
    data_degradation: DataDegradationConfig | None = None


class BinancePublicBarSource(MarketDataSource):
    """Синхронный источник баров Binance через публичный WebSocket."""

    def __init__(self, timeframe: str, cfg: Optional[BinanceWSConfig] = None) -> None:
        ensure_timeframe(timeframe)
        if websockets is None:
            raise RuntimeError("Module 'websockets' is required for BinancePublicBarSource")
        self._tf = binance_tf(timeframe)
        self._interval_ms = timeframe_to_ms(timeframe)
        self._cfg = cfg or BinanceWSConfig()
        self._symbols: List[str] = []

        self._q: "queue.Queue[Bar]" = queue.Queue(maxsize=10000)
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._last_open_ts: Dict[str, int] = {}

    def stream_bars(self, symbols: Sequence[str], interval_ms: int) -> Iterator[Bar]:
        if interval_ms != self._interval_ms:
            raise ValueError(
                f"Timeframe mismatch. Source={self._interval_ms}ms, requested={interval_ms}ms"
            )
        if not symbols:
            raise ValueError("No symbols provided")

        self._symbols = [s.lower() for s in symbols]

        self._stop.clear()
        self._thr = threading.Thread(target=self._run_loop, name="binance-ws", daemon=True)
        self._thr.start()

        try:
            while not self._stop.is_set():
                try:
                    bar = self._q.get(timeout=0.5)
                    yield bar
                except queue.Empty:
                    continue
        finally:
            self.close()

    def stream_ticks(self, symbols: Sequence[str]) -> Iterator[Tick]:
        return iter([])

    def close(self) -> None:
        self._stop.set()
        if self._thr and self._thr.is_alive():
            self._thr.join(timeout=5.0)

    # ----- internal -----

    def _streams(self) -> List[str]:
        return [f"{s}@kline_{self._tf}" for s in self._symbols]

    async def _client(self) -> None:
        backoff = self._cfg.reconnect_backoff_s
        while not self._stop.is_set():
            url = f"{_BINANCE_WS}?streams={'/'.join(self._streams())}"
            try:
                async with websockets.connect(url, ping_interval=self._cfg.ping_interval_s) as ws:  # type: ignore
                    backoff = self._cfg.reconnect_backoff_s
                    while not self._stop.is_set():
                        msg = await asyncio.wait_for(ws.recv(), timeout=self._cfg.ping_interval_s * 2)  # type: ignore
                        self._handle_message(msg)
            except Exception:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, self._cfg.reconnect_backoff_max_s)

    def _handle_message(self, raw: str) -> None:
        try:
            d = json.loads(raw)
            payload = d.get("data") or d
            if "k" not in payload:
                return
            k = payload["k"]
            volume_quote = None
            raw_volume_quote = k.get("q")
            if raw_volume_quote not in (None, ""):
                try:
                    volume_quote = Decimal(raw_volume_quote)
                except (ArithmeticError, ValueError, TypeError):
                    volume_quote = None

            open_ts = int(k["t"])
            close_ts = int(k.get("T", k["t"]))

            bar = Bar(
                ts=close_ts,
                symbol=str(k["s"]).upper(),
                open=Decimal(k["o"]),
                high=Decimal(k["h"]),
                low=Decimal(k["l"]),
                close=Decimal(k["c"]),
                volume_base=Decimal(k["v"]),
                volume_quote=volume_quote,
                trades=int(k.get("n", 0)),
                is_final=bool(k.get("x", False)),
            )
            bar_open_ms = open_ts
            prev_open = self._last_open_ts.get(bar.symbol)
            gap_ms = None
            duplicate_ts = False
            if prev_open is not None:
                delta = bar_open_ms - prev_open
                if delta <= 0:
                    duplicate_ts = True
                elif self._interval_ms > 0 and delta > self._interval_ms:
                    gap_ms = delta
            if prev_open is None or bar_open_ms >= prev_open:
                self._last_open_ts[bar.symbol] = bar_open_ms
            if gap_ms is not None:
                try:
                    log_payload = {
                        "symbol": bar.symbol,
                        "previous_open_ms": prev_open,
                        "previous_open_at": _format_utc(prev_open),
                        "current_open_ms": bar_open_ms,
                        "current_open_at": _format_utc(bar_open_ms),
                        "gap_ms": gap_ms,
                        "interval_ms": self._interval_ms,
                    }
                    logger.warning("PUBLIC_WS_BAR_GAP %s", log_payload)
                except Exception:
                    pass
            if duplicate_ts and prev_open is not None:
                try:
                    log_payload = {
                        "symbol": bar.symbol,
                        "previous_open_ms": prev_open,
                        "previous_open_at": _format_utc(prev_open),
                        "current_open_ms": bar_open_ms,
                        "current_open_at": _format_utc(bar_open_ms),
                    }
                    logger.info("PUBLIC_WS_BAR_DUPLICATE %s", log_payload)
                except Exception:
                    pass
            try:
                self._q.put_nowait(bar)
            except queue.Full:
                _ = self._q.get_nowait()
                self._q.put_nowait(bar)
        except Exception:
            pass

    def _run_loop(self) -> None:
        try:
            asyncio.run(self._client())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._client())
            loop.close()


# ----- utilities -----

_VALID_TF = {
    "1s",
    "5s",
    "10s",
    "15s",
    "30s",
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
}


def ensure_timeframe(tf: str) -> str:
    tf = str(tf).lower()
    if tf not in _VALID_TF:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return tf


def timeframe_to_ms(tf: str) -> int:
    tf = ensure_timeframe(tf)
    mult = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}
    return int(tf[:-1]) * mult[tf[-1]]


def binance_tf(tf: str) -> str:
    tf = ensure_timeframe(tf)
    return tf.lower()


def _format_utc(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return (
        datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
