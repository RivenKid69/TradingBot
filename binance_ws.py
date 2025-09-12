# realtime/binance_ws.py
from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Awaitable, Callable, List

from decimal import Decimal

import websockets

from core_models import Bar
from config import DataDegradationConfig


logger = logging.getLogger(__name__)


class BinanceWS:
    """
    Лёгкий клиент для публичных kline-стримов Binance (без ключей).
    Поддерживает авто-reconnect с backoff и пользовательский callback на закрытые свечи.
    """

    def __init__(
        self,
        *,
        symbols: List[str],
        interval: str = "1m",
        on_bar: Callable[[Bar], Awaitable[None]],
        base_url: str = "wss://stream.binance.com:9443",
        reconnect_initial_delay_s: float = 1.0,
        reconnect_max_delay_s: float = 60.0,
        heartbeat_interval_s: float = 15.0,
        data_degradation: DataDegradationConfig | None = None,
        stale_prob: float | None = None,
        drop_prob: float | None = None,
        dropout_prob: float | None = None,
        max_delay_ms: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        self.interval = str(interval)
        self.on_bar = on_bar
        self.base_url = base_url.rstrip("/")
        self.reconnect_initial_delay_s = float(reconnect_initial_delay_s)
        self.reconnect_max_delay_s = float(reconnect_max_delay_s)
        self.heartbeat_interval_s = float(heartbeat_interval_s)

        if data_degradation is None:
            data_degradation = DataDegradationConfig.default()
        if stale_prob is not None:
            data_degradation.stale_prob = stale_prob
        if drop_prob is not None:
            data_degradation.drop_prob = drop_prob
        if dropout_prob is not None:
            data_degradation.dropout_prob = dropout_prob
        if max_delay_ms is not None:
            data_degradation.max_delay_ms = max_delay_ms
        if seed is not None:
            data_degradation.seed = seed

        self.data_degradation = data_degradation
        self._rng = random.Random(self.data_degradation.seed)
        self._dd_total = 0
        self._dd_drop = 0
        self._dd_stale = 0
        self._dd_delay = 0

        if not self.symbols:
            raise ValueError("Не задан список symbols для подписки")

        streams = "/".join(f"{s.lower()}@kline_{self.interval}" for s in self.symbols)
        self.ws_url = f"{self.base_url}/stream?streams={streams}"
        self._stop = False

    async def _heartbeat(self, ws: websockets.WebSocketClientProtocol) -> None:
        while not self._stop:
            try:
                pong_waiter = await ws.ping()
                await asyncio.wait_for(pong_waiter, timeout=self.heartbeat_interval_s)
            except Exception:
                return
            await asyncio.sleep(self.heartbeat_interval_s)

    async def run_forever(self) -> None:
        delay = self.reconnect_initial_delay_s
        prev_bar: Bar | None = None
        try:
            while not self._stop:
                try:
                    async with websockets.connect(self.ws_url, ping_interval=None, close_timeout=5) as ws:
                        hb_task = asyncio.create_task(self._heartbeat(ws))
                        delay = self.reconnect_initial_delay_s  # сбросим backoff при успешном коннекте
                        async for msg in ws:
                            try:
                                payload = json.loads(msg)
                            except Exception:
                                continue
                            data = payload.get("data")
                            if not isinstance(data, dict):
                                continue
                            k = data.get("k")
                            if not isinstance(k, dict):
                                continue
                            if bool(k.get("x", False)):
                                try:
                                    bar = Bar(
                                        ts=int(k.get("t", 0)),
                                        symbol=str(k.get("s", "")).upper(),
                                        open=Decimal(k.get("o", 0.0)),
                                        high=Decimal(k.get("h", 0.0)),
                                        low=Decimal(k.get("l", 0.0)),
                                        close=Decimal(k.get("c", 0.0)),
                                        volume_base=Decimal(k.get("v", 0.0)),
                                        trades=int(k.get("n", 0)),
                                        is_final=True,
                                    )
                                except Exception:
                                    continue

                                self._dd_total += 1
                                if self._rng.random() < self.data_degradation.drop_prob:
                                    self._dd_drop += 1
                                    continue

                                if prev_bar is not None and self._rng.random() < self.data_degradation.stale_prob:
                                    self._dd_stale += 1
                                    if self._rng.random() < self.data_degradation.dropout_prob:
                                        delay_ms = self._rng.randint(0, self.data_degradation.max_delay_ms)
                                        if delay_ms > 0:
                                            self._dd_delay += 1
                                            await asyncio.sleep(delay_ms / 1000.0)
                                    await self.on_bar(prev_bar)
                                    continue

                                if self._rng.random() < self.data_degradation.dropout_prob:
                                    delay_ms = self._rng.randint(0, self.data_degradation.max_delay_ms)
                                    if delay_ms > 0:
                                        self._dd_delay += 1
                                        await asyncio.sleep(delay_ms / 1000.0)

                                prev_bar = bar
                                await self.on_bar(bar)
                        hb_task.cancel()
                except asyncio.CancelledError:
                    return
                except Exception:
                    await asyncio.sleep(delay)
                    delay = min(self.reconnect_max_delay_s, delay * 2.0)
        finally:
            if self._dd_total:
                logger.info(
                    "BinanceWS degradation: drop=%0.2f%% (%d/%d), stale=%0.2f%% (%d/%d), delay=%0.2f%% (%d/%d)",
                    self._dd_drop / self._dd_total * 100.0,
                    self._dd_drop,
                    self._dd_total,
                    self._dd_stale / self._dd_total * 100.0,
                    self._dd_stale,
                    self._dd_total,
                    self._dd_delay / self._dd_total * 100.0,
                    self._dd_delay,
                    self._dd_total,
                )

    def stop(self) -> None:
        self._stop = True
