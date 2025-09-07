# realtime/binance_ws.py
from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable, List

from decimal import Decimal

import websockets

from core_models import Bar


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
    ) -> None:
        self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        self.interval = str(interval)
        self.on_bar = on_bar
        self.base_url = base_url.rstrip("/")
        self.reconnect_initial_delay_s = float(reconnect_initial_delay_s)
        self.reconnect_max_delay_s = float(reconnect_max_delay_s)
        self.heartbeat_interval_s = float(heartbeat_interval_s)

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
                        # kline закрыта? поле 'x' == True
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
                            await self.on_bar(bar)
                    hb_task.cancel()
            except asyncio.CancelledError:
                return
            except Exception:
                await asyncio.sleep(delay)
                delay = min(self.reconnect_max_delay_s, delay * 2.0)

    def stop(self) -> None:
        self._stop = True
