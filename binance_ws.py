# realtime/binance_ws.py
from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import List
from clock import now_ms

from decimal import Decimal

import websockets

from core_models import Bar
from core_events import EventType, MarketEvent
from services.event_bus import EventBus
from config import DataDegradationConfig
from utils import SignalRateLimiter
from services import monitoring
import ws_dedup_state as signal_bus


logger = logging.getLogger(__name__)


class BinanceWS:
    """
    Лёгкий клиент для публичных kline-стримов Binance (без ключей).
    Поддерживает авто-reconnect с backoff и публикацию событий в EventBus.
    """

    def __init__(
        self,
        *,
        symbols: List[str],
        interval: str = "1m",
        bus: EventBus,
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
        rate_limit: float | None = None,
        backoff_base: float = 2.0,
        max_backoff: float = 60.0,
        ws_dedup_enabled: bool = False,
        ws_dedup_log_skips: bool = False,
        ws_dedup_persist_path: str | Path | None = None,
    ) -> None:
        self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        self.interval = str(interval)
        self._bus = bus
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

        self._rate_limiter = (
            SignalRateLimiter(rate_limit, backoff_base, max_backoff)
            if rate_limit and rate_limit > 0
            else None
        )
        self._rl_total = 0
        self._rl_delayed = 0
        self._rl_dropped = 0

        self._ws_dedup_enabled = ws_dedup_enabled
        self._ws_dedup_log_skips = ws_dedup_log_skips
        self._ws_dedup_persist_path = (
            Path(ws_dedup_persist_path)
            if ws_dedup_persist_path is not None
            else signal_bus.PERSIST_PATH
        )
        if self._ws_dedup_enabled:
            try:
                signal_bus.init(
                    enabled=True, persist_path=self._ws_dedup_persist_path
                )
                logger.info("WS_DEDUP_INIT size=%d", len(signal_bus.STATE))
            except Exception:
                pass

        if not self.symbols:
            raise ValueError("Не задан список symbols для подписки")

        streams = "/".join(f"{s.lower()}@kline_{self.interval}" for s in self.symbols)
        self.ws_url = f"{self.base_url}/stream?streams={streams}"
        self._stop = False

    async def _check_rate_limit(self) -> bool:
        if self._rate_limiter is None:
            self._rl_total += 1
            return True
        self._rl_total += 1
        allowed, status = self._rate_limiter.can_send(now_ms() / 1000.0)
        if allowed:
            return True
        if status == "rejected":
            self._rl_delayed += 1
            wait = max(self._rate_limiter._cooldown_until - now_ms() / 1000.0, 0.0)
            if wait > 0:
                await asyncio.sleep(wait)
            allowed, _ = self._rate_limiter.can_send(now_ms() / 1000.0)
            return allowed
        self._rl_dropped += 1
        return False

    async def _heartbeat(self, ws: websockets.WebSocketClientProtocol) -> None:
        while not self._stop:
            try:
                pong_waiter = await ws.ping()
                await asyncio.wait_for(pong_waiter, timeout=self.heartbeat_interval_s)
            except Exception:
                return
            await asyncio.sleep(self.heartbeat_interval_s)

    async def _emit(self, bar: Bar, close_ms: int) -> None:
        """Send bar event to the bus and persist dedup state."""
        feed_lag_ms = max(0, now_ms() - close_ms)
        event = MarketEvent(
            etype=EventType.MARKET_DATA_BAR,
            ts=now_ms(),
            bar=bar,
            meta={"feed_lag_ms": feed_lag_ms},
        )
        accepted = await self._bus.put(event)
        if not accepted:
            try:
                logger.info(
                    "BACKPRESSURE_DROP %s",
                    {"symbol": bar.symbol, "depth": self._bus.depth},
                )
            except Exception:
                pass
            try:
                monitoring.ws_backpressure_drop_count.labels(bar.symbol).inc()
            except Exception:
                pass
        try:
            signal_bus.update(bar.symbol, close_ms, auto_flush=False)
        except Exception:
            pass

    async def run_forever(self) -> None:
        delay = self.reconnect_initial_delay_s
        prev_bar: Bar | None = None
        prev_close_ms: int | None = None
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
                                        is_final=bool(k.get("x", False)),
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
                                    await self._emit(prev_bar, prev_close_ms or 0)
                                    continue

                                if self._rng.random() < self.data_degradation.dropout_prob:
                                    delay_ms = self._rng.randint(0, self.data_degradation.max_delay_ms)
                                    if delay_ms > 0:
                                        self._dd_delay += 1
                                        await asyncio.sleep(delay_ms / 1000.0)
                                bar_close_ms = int(k.get("T", 0))
                                if self._ws_dedup_enabled:
                                    try:
                                        if signal_bus.should_skip(bar.symbol, bar_close_ms):
                                            if self._ws_dedup_log_skips:
                                                try:
                                                    logger.info("WS_DUP_BAR_SKIP")
                                                except Exception:
                                                    pass
                                            try:
                                                monitoring.ws_dup_skipped_count.labels(bar.symbol).inc()
                                            except Exception:
                                                pass
                                            continue
                                    except Exception:
                                        pass

                                prev_bar = bar
                                prev_close_ms = bar_close_ms
                                if await self._check_rate_limit():
                                    await self._emit(bar, bar_close_ms)
                        hb_task.cancel()
                except asyncio.CancelledError:
                    return
                except Exception:
                    await asyncio.sleep(delay)
                    delay = min(self.reconnect_max_delay_s, delay * 2.0)
        finally:
            if self._ws_dedup_enabled:
                try:
                    signal_bus.shutdown()
                except Exception:
                    pass
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

            if self._rl_total:
                logger.info(
                    "BinanceWS rate limiting: delayed=%0.2f%% (%d/%d), dropped=%0.2f%% (%d/%d)",
                    self._rl_delayed / self._rl_total * 100.0,
                    self._rl_delayed,
                    self._rl_total,
                    self._rl_dropped / self._rl_total * 100.0,
                    self._rl_dropped,
                    self._rl_total,
                )

    def stop(self) -> None:
        self._stop = True
