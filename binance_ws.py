# realtime/binance_ws.py
from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from clock import now_ms

from decimal import Decimal

import websockets

from core_models import Bar
from core_events import EventType, MarketEvent
from services.event_bus import EventBus
from config import DataDegradationConfig
from utils import SignalRateLimiter
from services import monitoring, ops_kill_switch
from services.retry import retry_async
from core_config import RetryConfig
import ws_dedup_state as signal_bus
from services.monitoring import MonitoringAggregator


logger = logging.getLogger(__name__)


def _interval_to_ms(iv: str) -> int:
    iv = str(iv).strip().lower()
    mult = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}
    if iv[-1] not in mult:
        raise ValueError(f"Unsupported interval: {iv}")
    return int(iv[:-1]) * mult[iv[-1]]


def _format_utc(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return (
        datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


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
        retry_cfg: RetryConfig | None = None,
        ws_dedup_enabled: bool = False,
        ws_dedup_log_skips: bool = False,
        ws_dedup_persist_path: str | Path | None = None,
        monitoring_agg: MonitoringAggregator | None = None,
    ) -> None:
        self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        self.interval = str(interval)
        self._bus = bus
        self.base_url = base_url.rstrip("/")
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

        self._retry_cfg = retry_cfg or RetryConfig()

        self._monitoring = monitoring_agg
        self.consecutive_ws_failures = 0

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

        self._streams = [f"{s.lower()}@kline_{self.interval}" for s in self.symbols]
        self.ws_url = f"{self.base_url}/ws"

        self._ws: websockets.WebSocketClientProtocol | None = None
        self._stop = False
        self._last_close_ms = 0
        self._interval_ms = _interval_to_ms(self.interval)
        self._last_open_ts: Dict[str, int] = {}

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
        try:
            monitoring.report_feed_lag(bar.symbol, feed_lag_ms)
        except Exception:
            pass
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
                monitoring.report_ws_failure(bar.symbol)
            except Exception:
                pass
        try:
            signal_bus.update(bar.symbol, close_ms, auto_flush=False)
        except Exception:
            pass

    async def _connect_once(self) -> websockets.WebSocketClientProtocol:
        ws = await websockets.connect(self.ws_url, ping_interval=None, close_timeout=5)
        await ws.send(
            json.dumps(
                {
                    "method": "SUBSCRIBE",
                    "params": self._streams,
                    "id": 1,
                }
            )
        )
        self._dd_total = self._dd_drop = self._dd_stale = self._dd_delay = 0
        self._rl_total = self._rl_delayed = self._rl_dropped = 0
        if self._monitoring is not None:
            self._monitoring.record_ws("reconnect")
        self.consecutive_ws_failures = 0
        return ws

    async def run_forever(self) -> None:
        prev_bar: Bar | None = None
        prev_close_ms: int | None = None
        had_ws_error = False
        self._last_close_ms = now_ms()

        async def _stale_monitor() -> None:
            while not self._stop and not ops_kill_switch.tripped():
                await asyncio.sleep(self._interval_ms / 1000.0)
                last = self._last_close_ms
                if last and now_ms() - last > self._interval_ms * 2:
                    try:
                        ops_kill_switch.record_stale()
                    except Exception:
                        pass

        stale_task = asyncio.create_task(_stale_monitor())
        connect = retry_async(self._retry_cfg, lambda e: "ws")(self._connect_once)
        try:
            while not self._stop and not ops_kill_switch.tripped():
                try:
                    ws = await connect()
                    self._ws = ws
                    if had_ws_error:
                        try:
                            ops_kill_switch.manual_reset()
                        except Exception:
                            pass
                        had_ws_error = False
                    hb_task = asyncio.create_task(self._heartbeat(ws))
                    try:
                        async for msg in ws:
                            if self._stop or ops_kill_switch.tripped():
                                break
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
                                bar_open_ms = int(bar.ts)
                                prev_open = self._last_open_ts.get(bar.symbol)
                                gap_ms = None
                                duplicate_ts = False
                                if prev_open is not None:
                                    delta = bar_open_ms - prev_open
                                    if delta <= 0:
                                        duplicate_ts = True
                                    elif (
                                        self._interval_ms > 0
                                        and delta > self._interval_ms
                                    ):
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
                                        logger.warning("WS_BAR_GAP %s", log_payload)
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
                                        logger.info("WS_BAR_DUPLICATE %s", log_payload)
                                    except Exception:
                                        pass
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
                                                monitoring.report_ws_failure(bar.symbol)
                                            except Exception:
                                                pass
                                            continue
                                    except Exception:
                                        pass

                                prev_bar = bar
                                prev_close_ms = bar_close_ms
                                self._last_close_ms = bar_close_ms
                                if await self._check_rate_limit():
                                    await self._emit(bar, bar_close_ms)
                    finally:
                        hb_task.cancel()
                        try:
                            await hb_task
                        except Exception:
                            pass
                        try:
                            await ws.close()
                        except Exception:
                            pass
                        self._ws = None
                except asyncio.CancelledError:
                    return
                except Exception:
                    try:
                        ops_kill_switch.record_error("ws")
                    except Exception:
                        pass
                    had_ws_error = True
                    self.consecutive_ws_failures += 1
                    if self._monitoring is not None:
                        self._monitoring.record_ws("failure")
                    continue
        finally:
            self._ws = None
            stale_task.cancel()
            try:
                await stale_task
            except Exception:
                pass
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

    async def stop(self) -> None:
        self._stop = True
        ws = self._ws
        if ws is not None:
            try:
                await ws.send(
                    json.dumps(
                        {
                            "method": "UNSUBSCRIBE",
                            "params": self._streams,
                            "id": 1,
                        }
                    )
                )
            except Exception:
                pass
            try:
                await ws.close()
            except Exception:
                pass
            self._ws = None
