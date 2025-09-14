# -*- coding: utf-8 -*-
"""
services/service_signal_runner.py
Онлайн-оркестратор: MarketDataSource -> FeaturePipe -> Strategy -> RiskGuards -> TradeExecutor.
Не содержит бизнес-логики, только склейка компонентов.

Пример использования через конфиг:

```python
from core_config import CommonRunConfig
from service_signal_runner import from_config

cfg = CommonRunConfig(...)
for report in from_config(cfg):
    print(report)
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Iterator, Protocol, Callable
import os
import logging
import threading
from pathlib import Path
from collections import deque, defaultdict
import time

import yaml
import clock
from services import monitoring
from services.monitoring import skipped_incomplete_bars
from pipeline import check_ttl
from services.utils_app import append_row_csv
from services.signal_bus import log_drop

from sandbox.sim_adapter import SimAdapter  # исп. как TradeExecutor-подобный мост
from core_models import Bar
from core_contracts import FeaturePipe, SignalPolicy, PolicyCtx
from services.utils_config import snapshot_config  # снапшот конфига (Фаза 3)  # noqa: F401
from core_config import CommonRunConfig, ClockSyncConfig, ThrottleConfig
from utils import TokenBucket
import di_registry
import ws_dedup_state as signal_bus


class RiskGuards(Protocol):
    def apply(self, ts_ms: int, symbol: str, decisions: Sequence[Any]) -> Sequence[Any]: ...


@dataclass
class SignalRunnerConfig:
    snapshot_config_path: Optional[str] = None
    artifacts_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    run_id: Optional[str] = None


# обратная совместимость
RunnerConfig = SignalRunnerConfig


class _Worker:
    def __init__(
        self,
        fp: FeaturePipe,
        policy: SignalPolicy,
        logger: logging.Logger,
        executor: Any,
        guards: Optional[RiskGuards] = None,
        safe_mode_fn: Callable[[], bool] | None = None,
        *,
        enforce_closed_bars: bool,
        ws_dedup_enabled: bool = False,
        ws_dedup_log_skips: bool = False,
        ws_dedup_timeframe_ms: int = 0,
        throttle_cfg: ThrottleConfig | None = None,
    ) -> None:
        self._fp = fp
        self._policy = policy
        self._logger = logger
        self._executor = executor
        self._guards = guards
        self._safe_mode_fn = safe_mode_fn or (lambda: False)
        self._enforce_closed_bars = enforce_closed_bars
        self._ws_dedup_enabled = ws_dedup_enabled
        self._ws_dedup_log_skips = ws_dedup_log_skips
        self._ws_dedup_timeframe_ms = ws_dedup_timeframe_ms
        self._throttle_cfg = throttle_cfg
        self._global_bucket = None
        self._symbol_bucket_factory = None
        self._symbol_buckets = None
        self._queue = None
        if throttle_cfg is not None and throttle_cfg.enabled:
            self._global_bucket = TokenBucket(
                rps=throttle_cfg.global_.rps, burst=throttle_cfg.global_.burst
            )
            self._symbol_bucket_factory = lambda: TokenBucket(
                rps=throttle_cfg.symbol.rps, burst=throttle_cfg.symbol.burst
            )
            self._symbol_buckets = defaultdict(self._symbol_bucket_factory)
            self._queue = deque(maxlen=throttle_cfg.queue.max_items)

    def _acquire_tokens(self, symbol: str) -> tuple[bool, str | None]:
        if self._global_bucket is None:
            return True, None
        now = time.monotonic()
        if not self._global_bucket.consume(now=now):
            return False, "THROTTLED_GLOBAL"
        sb = self._symbol_buckets[symbol]
        if not sb.consume(now=now):
            self._global_bucket.tokens = min(
                self._global_bucket.tokens + 1.0, self._global_bucket.burst
            )
            return False, "THROTTLED_SYMBOL"
        return True, None

    def _refund_tokens(self, symbol: str) -> None:
        if self._global_bucket is not None:
            self._global_bucket.tokens = min(
                self._global_bucket.tokens + 1.0, self._global_bucket.burst
            )
        if self._symbol_buckets is not None:
            sb = self._symbol_buckets[symbol]
            sb.tokens = min(sb.tokens + 1.0, sb.burst)

    def _emit(self, o: Any, symbol: str, bar_close_ms: int) -> bool:
        created_ts = int(getattr(o, "created_ts_ms", 0) or 0)
        now_ms = clock.now_ms()
        ok, expires_at_ms, _ = check_ttl(
            bar_close_ms=created_ts,
            now_ms=now_ms,
            timeframe_ms=self._ws_dedup_timeframe_ms,
        )
        if not ok:
            try:
                self._logger.info(
                    "TTL_EXPIRED_PUBLISH %s",
                    {
                        "symbol": symbol,
                        "created_ts_ms": created_ts,
                        "now_ms": now_ms,
                        "expires_at_ms": expires_at_ms,
                    },
                )
            except Exception:
                pass
            try:
                monitoring.signal_absolute_count.labels(symbol).inc()
            except Exception:
                pass
            return False
        try:
            age_ms = now_ms - created_ts
            monitoring.age_at_publish_ms.labels(symbol).observe(age_ms)
            monitoring.signal_published_count.labels(symbol).inc()
        except Exception:
            pass

        score = float(getattr(o, "score", 0) or 0)
        fh = str(getattr(o, "features_hash", "") or "")
        side = getattr(o, "side", "")
        side = side.value if hasattr(side, "value") else str(side)
        side = str(side).upper()
        out_csv = getattr(signal_bus, "OUT_CSV", None)
        if out_csv:
            vol = getattr(o, "volume_frac", None)
            if vol is None:
                vol = getattr(o, "quantity", 0)
            try:
                vol_val = float(vol)
            except Exception:
                vol_val = 0.0
            header = [
                "ts_ms",
                "symbol",
                "side",
                "volume_frac",
                "score",
                "features_hash",
            ]
            row = [bar_close_ms, symbol, side, vol_val, score, fh]
            try:
                append_row_csv(out_csv, header, row)
            except Exception:
                pass
        try:
            self._logger.info("order %s", o)
        except Exception:
            pass
        if getattr(signal_bus, "ENABLED", False):
            try:
                signal_bus.publish_signal(
                    ts_ms=bar_close_ms,
                    symbol=symbol,
                    side=side,
                    score=score,
                    features_hash=fh,
                    bar_close_ms=bar_close_ms,
                )
            except Exception:
                pass
        submit = getattr(self._executor, "submit", None)
        if callable(submit):
            try:
                submit(o)
            except Exception:
                pass
        return True

    def _drain_queue(self) -> list[Any]:
        emitted: list[Any] = []
        if self._queue is None:
            return emitted
        now = time.monotonic()
        while self._queue:
            exp, symbol, bar_close_ms, order = self._queue[0]
            if exp <= now:
                self._queue.popleft()
                try:
                    log_drop(symbol, bar_close_ms, order, "QUEUE_EXPIRED")
                except Exception:
                    pass
                continue
            ok, _ = self._acquire_tokens(symbol)
            if not ok:
                break
            self._queue.popleft()
            if not self._emit(order, symbol, bar_close_ms):
                self._refund_tokens(symbol)
            else:
                emitted.append(order)
        return emitted

    def process(self, bar: Bar):
        if self._safe_mode_fn():
            return []

        emitted: list[Any] = []
        if self._queue is not None:
            emitted.extend(self._drain_queue())

        close_ms: int | None = None
        if self._ws_dedup_enabled:
            close_ms = int(bar.ts) + self._ws_dedup_timeframe_ms
            if signal_bus.should_skip(bar.symbol, close_ms):
                if self._ws_dedup_log_skips:
                    try:
                        self._logger.info("SKIP_DUPLICATE_BAR")
                    except Exception:
                        pass
                try:
                    monitoring.ws_dup_skipped_count.labels(bar.symbol).inc()
                except Exception:
                    pass
                try:
                    monitoring.queue_len.set(len(self._queue) if self._queue else 0)
                except Exception:
                    pass
                return emitted
        if self._enforce_closed_bars and not bar.is_final:
            try:
                self._logger.info("SKIP_INCOMPLETE_BAR")
            except Exception:
                pass
            try:
                skipped_incomplete_bars.labels(bar.symbol).inc()
            except Exception:
                pass
            try:
                monitoring.queue_len.set(len(self._queue) if self._queue else 0)
            except Exception:
                pass
            return emitted

        feats = self._fp.update(bar)
        ctx = PolicyCtx(ts=int(bar.ts), symbol=bar.symbol)
        orders = list(self._policy.decide({**feats}, ctx) or [])
        if self._guards:
            orders = list(self._guards.apply(int(bar.ts), bar.symbol, orders) or [])

        created_ts_ms = clock.now_ms()
        checked_orders = []
        for o in orders:
            ok, expires_at_ms, _ = check_ttl(
                bar_close_ms=int(bar.ts),
                now_ms=created_ts_ms,
                timeframe_ms=self._ws_dedup_timeframe_ms,
            )
            if not ok:
                try:
                    self._logger.info(
                        "TTL_EXPIRED_BOUNDARY %s",
                        {
                            "symbol": bar.symbol,
                            "bar_close_ms": int(bar.ts),
                            "now_ms": created_ts_ms,
                            "expires_at_ms": expires_at_ms,
                        },
                    )
                except Exception:
                    pass
                try:
                    monitoring.ttl_expired_boundary_count.labels(bar.symbol).inc()
                    monitoring.signal_boundary_count.labels(bar.symbol).inc()
                except Exception:
                    pass
                continue

            setattr(o, "created_ts_ms", created_ts_ms)
            checked_orders.append(o)

        for o in checked_orders:
            if self._throttle_cfg and self._throttle_cfg.enabled:
                ok, reason = self._acquire_tokens(bar.symbol)
                if not ok:
                    if self._throttle_cfg.mode == "drop":
                        try:
                            log_drop(bar.symbol, int(bar.ts), o, reason or "")
                        except Exception:
                            pass
                    elif self._throttle_cfg.mode == "queue" and self._queue is not None:
                        exp = time.monotonic() + self._throttle_cfg.queue.ttl_ms / 1000.0
                        self._queue.append((exp, bar.symbol, int(bar.ts), o))
                    continue
            if not self._emit(o, bar.symbol, int(bar.ts)):
                self._refund_tokens(bar.symbol)
            else:
                emitted.append(o)

        try:
            monitoring.queue_len.set(len(self._queue) if self._queue else 0)
        except Exception:
            pass

        if self._ws_dedup_enabled and close_ms is not None:
            try:
                signal_bus.update(bar.symbol, close_ms, auto_flush=False)
            except Exception:
                pass
        return emitted

    # Backwards compatibility: legacy callers may still use ``on_bar``.
    on_bar = process


async def worker_loop(bus: "EventBus", worker: _Worker) -> None:
    """Consume bars from ``bus`` and pass them to ``worker``.

    The function exits when :meth:`EventBus.close` is called and all
    remaining events are processed.  On cancellation, it drains any queued
    events before re-raising :class:`asyncio.CancelledError`.
    """
    import asyncio

    try:
        while True:
            event = await bus.get()
            if event is None:
                break
            bar = getattr(event, "bar", None)
            if bar is None:
                continue
            worker.process(bar)
    except asyncio.CancelledError:
        # Drain remaining events best-effort before shutting down
        try:
            while True:
                ev = bus._queue.get_nowait()  # type: ignore[attr-defined]
                if ev is None:
                    break
                bar = getattr(ev, "bar", None)
                if bar is not None:
                    worker.process(bar)
        except Exception:
            pass
        raise


class ServiceSignalRunner:
    def __init__(
        self,
        adapter: SimAdapter,
        feature_pipe: FeaturePipe,
        policy: SignalPolicy,
        risk_guards: Optional[RiskGuards] = None,
        cfg: Optional[SignalRunnerConfig] = None,
        clock_sync_cfg: ClockSyncConfig | None = None,
        throttle_cfg: ThrottleConfig | None = None,
        *,
        enforce_closed_bars: bool = True,
        ws_dedup_enabled: bool = False,
        ws_dedup_log_skips: bool = False,
        ws_dedup_timeframe_ms: int = 0,
    ) -> None:
        self.adapter = adapter
        self.feature_pipe = feature_pipe
        self.policy = policy
        self.risk_guards = risk_guards
        self.cfg = cfg or SignalRunnerConfig()
        self.logger = logging.getLogger(__name__)
        self.clock_sync_cfg = clock_sync_cfg
        self.throttle_cfg = throttle_cfg
        self._clock_safe_mode = False
        self._clock_stop = threading.Event()
        self._clock_thread: Optional[threading.Thread] = None
        self.enforce_closed_bars = enforce_closed_bars
        self.ws_dedup_enabled = ws_dedup_enabled
        self.ws_dedup_log_skips = ws_dedup_log_skips
        self.ws_dedup_timeframe_ms = ws_dedup_timeframe_ms

        if self.throttle_cfg is not None:
            self.logger.info(
                "throttle limits: enabled=%s global_rps=%s global_burst=%s symbol_rps=%s symbol_burst=%s mode=%s",
                self.throttle_cfg.enabled,
                self.throttle_cfg.global_.rps,
                self.throttle_cfg.global_.burst,
                self.throttle_cfg.symbol.rps,
                self.throttle_cfg.symbol.burst,
                self.throttle_cfg.mode,
            )

        run_id = self.cfg.run_id or "sim"
        logs_dir = self.cfg.logs_dir or "logs"
        sim = getattr(self.adapter, "sim", None) or getattr(self.adapter, "_sim", None)
        if sim is not None:
            logging_config = {
                "trades_path": os.path.join(logs_dir, f"log_trades_{run_id}.csv"),
                "reports_path": os.path.join(logs_dir, f"report_equity_{run_id}.csv"),
            }
            try:
                from logging import LogWriter, LogConfig  # type: ignore

                sim._logger = LogWriter(LogConfig.from_dict(logging_config), run_id=run_id)
            except Exception:
                pass

    def run(self) -> Iterator[Dict[str, Any]]:
        # снапшот конфига, если задан
        if self.cfg.snapshot_config_path and self.cfg.artifacts_dir:
            snapshot_config(self.cfg.snapshot_config_path, self.cfg.artifacts_dir)

        self.feature_pipe.warmup()
        worker = _Worker(
            self.feature_pipe,
            self.policy,
            self.logger,
            self.adapter,
            self.risk_guards,
            lambda: self._clock_safe_mode,
            enforce_closed_bars=self.enforce_closed_bars,
            ws_dedup_enabled=self.ws_dedup_enabled,
            ws_dedup_log_skips=self.ws_dedup_log_skips,
            ws_dedup_timeframe_ms=self.ws_dedup_timeframe_ms,
            throttle_cfg=self.throttle_cfg,
        )

        # Optional asynchronous event bus processing
        bus = getattr(self.adapter, "bus", None)
        loop_thread: threading.Thread | None = None
        if bus is not None:
            import asyncio

            n_workers = max(1, int(getattr(self.adapter, "n_workers", 1)))
            loop = asyncio.new_event_loop()

            async def _run_workers() -> None:
                tasks = [asyncio.create_task(worker_loop(bus, worker)) for _ in range(n_workers)]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

            def _loop_runner() -> None:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_run_workers())

            loop_thread = threading.Thread(target=_loop_runner, daemon=True)
            loop_thread.start()

        client = getattr(self.adapter, "client", None)
        if client is not None and self.clock_sync_cfg is not None:
            drift = clock.sync_clock(client, self.clock_sync_cfg, monitoring)
            try:
                monitoring.report_clock_sync(drift, 0.0, True, clock.system_utc_ms())
            except Exception:
                pass
            self.logger.info(
                "clock drift %.2fms (warn=%dms kill=%dms)",
                drift,
                int(self.clock_sync_cfg.warn_threshold_ms),
                int(self.clock_sync_cfg.kill_threshold_ms),
            )

            def _sync_loop() -> None:
                backoff = self.clock_sync_cfg.refresh_sec
                while not self._clock_stop.wait(backoff):
                    before = clock.last_sync_at
                    drift_local = clock.sync_clock(client, self.clock_sync_cfg, monitoring)
                    success = clock.last_sync_at != before
                    try:
                        monitoring.report_clock_sync(
                            drift_local, 0.0, success, clock.system_utc_ms()
                        )
                    except Exception:
                        pass
                    if not success:
                        try:
                            monitoring.clock_sync_fail.inc()
                        except Exception:
                            pass
                        backoff = min(backoff * 2.0, self.clock_sync_cfg.refresh_sec * 10.0)
                        continue
                    backoff = self.clock_sync_cfg.refresh_sec
                    if drift_local > self.clock_sync_cfg.kill_threshold_ms:
                        self._clock_safe_mode = True
                        try:
                            monitoring.clock_sync_fail.inc()
                        except Exception:
                            pass
                        self.logger.error(
                            "clock drift %.2fms exceeds kill threshold %.2fms; entering safe mode",
                            drift_local,
                            self.clock_sync_cfg.kill_threshold_ms,
                        )
                    else:
                        self._clock_safe_mode = False
                        if drift_local > self.clock_sync_cfg.warn_threshold_ms:
                            self.logger.warning(
                                "clock drift %.2fms exceeds warn threshold %.2fms",
                                drift_local,
                                self.clock_sync_cfg.warn_threshold_ms,
                            )
                            try:
                                monitoring.clock_sync_fail.inc()
                            except Exception:
                                pass

            self._clock_thread = threading.Thread(target=_sync_loop, daemon=True)
            self._clock_thread.start()

        try:
            for rep in self.adapter.run_events(worker):
                yield rep
        finally:
            if bus is not None:
                try:
                    bus.close()
                finally:
                    if loop_thread is not None:
                        loop_thread.join()
            self._clock_stop.set()
            if self._clock_thread is not None:
                try:
                    self._clock_thread.join(timeout=self.clock_sync_cfg.refresh_sec if self.clock_sync_cfg else 1.0)
                except Exception:
                    pass
            sim = getattr(self.adapter, "sim", None) or getattr(self.adapter, "_sim", None)
            logger = getattr(sim, "_logger", None) if sim is not None else None
            try:
                if logger:
                    logger.flush()
            except Exception:
                pass
            if signal_bus.ENABLED:
                try:
                    signal_bus.shutdown()
                except Exception:
                    pass


def from_config(
    cfg: CommonRunConfig,
    *,
    snapshot_config_path: str | None = None,
) -> Iterator[Dict[str, Any]]:
    """Build dependencies from ``cfg`` and run :class:`ServiceSignalRunner`."""

    logging.getLogger(__name__).info("timing settings: %s", cfg.timing.dict())

    # Load signal bus configuration
    signals_cfg: Dict[str, Any] = {}
    sig_cfg_path = Path("configs/signals.yaml")
    if sig_cfg_path.exists():
        try:
            with sig_cfg_path.open("r", encoding="utf-8") as f:
                signals_cfg = yaml.safe_load(f) or {}
        except Exception:
            signals_cfg = {}

    bus_enabled = bool(signals_cfg.get("enabled", False))
    ttl = int(signals_cfg.get("ttl_seconds", 0))
    out_csv = signals_cfg.get("out_csv")
    dedup_persist = signals_cfg.get("dedup_persist") or cfg.ws_dedup.persist_path

    signal_bus.init(
        enabled=bus_enabled,
        ttl_seconds=ttl,
        persist_path=dedup_persist,
        out_csv=out_csv,
    )

    cfg.ws_dedup.enabled = bus_enabled
    cfg.ws_dedup.persist_path = str(dedup_persist)

    svc_cfg = SignalRunnerConfig(
        snapshot_config_path=snapshot_config_path,
        artifacts_dir=cfg.artifacts_dir,
    )
    if svc_cfg.logs_dir is None:
        svc_cfg.logs_dir = cfg.logs_dir
    if svc_cfg.run_id is None:
        svc_cfg.run_id = cfg.run_id

    container = di_registry.build_graph(cfg.components, cfg)
    adapter: SimAdapter = container["executor"]
    fp: FeaturePipe = container["feature_pipe"]
    policy: SignalPolicy = container["policy"]
    guards: RiskGuards | None = container.get("risk_guards")
    service = ServiceSignalRunner(
        adapter,
        fp,
        policy,
        guards,
        svc_cfg,
        cfg.clock_sync,
        cfg.throttle,
        enforce_closed_bars=cfg.timing.enforce_closed_bars,
        ws_dedup_enabled=cfg.ws_dedup.enabled,
        ws_dedup_log_skips=cfg.ws_dedup.log_skips,
        ws_dedup_timeframe_ms=cfg.timing.timeframe_ms,
    )
    return service.run()


__all__ = ["SignalRunnerConfig", "RunnerConfig", "ServiceSignalRunner", "from_config"]
