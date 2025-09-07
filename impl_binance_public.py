# -*- coding: utf-8 -*-
"""
impl_binance_public.py
Источник рыночных данных Binance Public WS. Выдаёт Bar через интерфейс MarketDataSource.
Работает синхронно через внутренний поток, который обслуживает asyncio + websockets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence
import json
import queue
import threading
import time
import asyncio

try:
    import websockets  # type: ignore
except Exception as e:
    websockets = None  # type: ignore

from core_contracts import MarketDataSource
from core_models import Bar, to_decimal
from market_utils import binance_tf, ensure_timeframe, timeframe_to_ms


_BINANCE_WS = "wss://stream.binance.com:9443/stream"


@dataclass
class BinanceWSConfig:
    reconnect_backoff_s: float = 1.0
    reconnect_backoff_max_s: float = 30.0
    ping_interval_s: float = 10.0
    vendor: str = "binance"


class BinancePublicBarSource(MarketDataSource):
    """Synchronous Binance WebSocket bar source implementing MarketDataSource."""

    def __init__(self, timeframe: str, cfg: Optional[BinanceWSConfig] = None) -> None:
        ensure_timeframe(timeframe)
        if websockets is None:
            raise RuntimeError("Module 'websockets' is required for BinancePublicBarSource")
        self._tf = binance_tf(timeframe)
        self._cfg = cfg or BinanceWSConfig()
        self._symbols: List[str] = []

        self._q: "queue.Queue[Bar]" = queue.Queue(maxsize=10000)
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    def stream_bars(self, symbols: Sequence[str], interval_ms: int) -> Iterator[Bar]:
        if interval_ms != timeframe_to_ms(self._tf):
            raise ValueError(f"Timeframe mismatch. Source={self._tf}, requested_ms={interval_ms}")
        self._symbols = [s.lower() for s in symbols]  # binance expects lower-case

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
            self._stop.set()
            if self._thr and self._thr.is_alive():
                self._thr.join(timeout=5.0)

    # ----- internal -----

    def _streams(self) -> List[str]:
        # пример: "btcusdt@kline_1m"
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
            payload = d.get("data") or d  # single or multiplexed
            if "k" not in payload:
                return
            k = payload["k"]
            bar = Bar(
                ts=int(k["t"]),
                symbol=str(k["s"]).upper(),
                open=to_decimal(k["o"]),
                high=to_decimal(k["h"]),
                low=to_decimal(k["l"]),
                close=to_decimal(k["c"]),
                volume_base=to_decimal(k["v"]),
                trades=int(k.get("n", 0)),
                is_final=bool(k.get("x", False)),
            )
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
            # если уже есть цикл, создаём новый
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._client())
            loop.close()
