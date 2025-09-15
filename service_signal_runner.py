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
import signal
from pathlib import Path
from collections import deque, defaultdict
import time
import shlex

import json
import yaml
import clock
from services import monitoring, ops_kill_switch
from services.monitoring import skipped_incomplete_bars, pipeline_stage_drop_count
from pipeline import (
    check_ttl,
    closed_bar_guard,
    apply_no_trade_windows,
    policy_decide,
    apply_risk,
    Stage,
    Reason,
    PipelineResult,
    PipelineConfig,
    PipelineStageConfig,
)
from services.signal_bus import log_drop
from services.event_bus import EventBus
from services.shutdown import ShutdownManager
from services.signal_csv_writer import SignalCSVWriter

from sandbox.sim_adapter import SimAdapter  # исп. как TradeExecutor-подобный мост
from core_models import Bar
from core_contracts import FeaturePipe, SignalPolicy
from services.utils_config import snapshot_config  # снапшот конфига (Фаза 3)  # noqa: F401
from core_config import CommonRunConfig, ClockSyncConfig, ThrottleConfig
from no_trade_config import NoTradeConfig
from utils import TokenBucket
import di_registry
import ws_dedup_state as signal_bus
import state_store


class RiskGuards(Protocol):
    def apply(
        self, ts_ms: int, symbol: str, decisions: Sequence[Any]
    ) -> tuple[Sequence[Any], str | None]: ...


@dataclass
class SignalRunnerConfig:
    snapshot_config_path: Optional[str] = None
    artifacts_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    marker_path: Optional[str] = None
    run_id: Optional[str] = None
    snapshot_metrics_json: Optional[str] = None
    snapshot_metrics_csv: Optional[str] = None
    snapshot_metrics_sec: int = 60


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
        no_trade_cfg: NoTradeConfig | None = None,
        pipeline_cfg: PipelineConfig | None = None,
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
        self._no_trade_cfg = no_trade_cfg
        self._pipeline_cfg = pipeline_cfg
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
        if self._guards is not None:
            checked, reason = self._guards.apply(bar_close_ms, symbol, [o])
            if reason or not checked:
                try:
                    self._logger.info(
                        "DROP %s",
                        {"stage": Stage.PUBLISH.name, "reason": reason or ""},
                    )
                except Exception:
                    pass
                try:
                    pipeline_stage_drop_count.labels(
                        symbol, Stage.PUBLISH.name, Reason.RISK_POSITION.name
                    ).inc()
                except Exception:
                    pass
                try:
                    log_drop(symbol, bar_close_ms, o, reason or "RISK")
                except Exception:
                    pass
                return False
            o = list(checked)[0]

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
        writer = getattr(signal_bus, "OUT_WRITER", None)
        if writer is not None:
            vol = getattr(o, "volume_frac", None)
            if vol is None:
                vol = getattr(o, "quantity", 0)
            try:
                vol_val = float(vol)
            except Exception:
                vol_val = 0.0
            row = {
                "ts_ms": bar_close_ms,
                "symbol": symbol,
                "side": side,
                "volume_frac": vol_val,
                "score": score,
                "features_hash": fh,
            }
            try:
                writer.write(row)
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

    def publish_decision(
        self,
        o: Any,
        symbol: str,
        bar_close_ms: int,
        *,
        stage_cfg: PipelineStageConfig | None = None,
    ) -> PipelineResult:
        """Final pipeline stage: throttle and emit order."""
        if stage_cfg is not None and not stage_cfg.enabled:
            return PipelineResult(action="pass", stage=Stage.PUBLISH, decision=o)
        if self._throttle_cfg and self._throttle_cfg.enabled:
            ok, reason = self._acquire_tokens(symbol)
            if not ok:
                if self._throttle_cfg.mode == "queue" and self._queue is not None:
                    exp = time.monotonic() + self._throttle_cfg.queue.ttl_ms / 1000.0
                    self._queue.append((exp, symbol, bar_close_ms, o))
                    try:
                        monitoring.throttle_enqueued_count.labels(symbol, reason or "").inc()
                    except Exception:
                        pass
                    return PipelineResult(action="queue", stage=Stage.PUBLISH)
                try:
                    log_drop(symbol, bar_close_ms, o, reason or "")
                    monitoring.throttle_dropped_count.labels(symbol, reason or "").inc()
                except Exception:
                    pass
                return PipelineResult(action="drop", stage=Stage.PUBLISH, reason=Reason.OTHER)
        if not self._emit(o, symbol, bar_close_ms):
            self._refund_tokens(symbol)
            return PipelineResult(action="drop", stage=Stage.PUBLISH, reason=Reason.OTHER)
        return PipelineResult(action="pass", stage=Stage.PUBLISH, decision=o)

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
                    monitoring.throttle_queue_expired_count.labels(symbol).inc()
                    monitoring.throttle_dropped_count.labels(symbol, "QUEUE_EXPIRED").inc()
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
        if self._pipeline_cfg is not None and not self._pipeline_cfg.enabled:
            return []
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
        guard_res = closed_bar_guard(
            bar=bar,
            now_ms=clock.now_ms(),
            enforce=self._enforce_closed_bars,
            lag_ms=0,
            stage_cfg=self._pipeline_cfg.get("closed_bar") if self._pipeline_cfg else None,
        )
        if guard_res.action == "drop":
            try:
                self._logger.info(
                    "DROP %s",
                    {
                        "stage": guard_res.stage.name,
                        "reason": getattr(guard_res.reason, "name", ""),
                    },
                )
            except Exception:
                pass
            try:
                skipped_incomplete_bars.labels(bar.symbol).inc()
            except Exception:
                pass
            try:
                pipeline_stage_drop_count.labels(
                    bar.symbol,
                    guard_res.stage.name,
                    guard_res.reason.name if guard_res.reason else "",
                ).inc()
            except Exception:
                pass
            try:
                monitoring.queue_len.set(len(self._queue) if self._queue else 0)
            except Exception:
                pass
            return emitted

        if self._no_trade_cfg is not None:
            win_res = apply_no_trade_windows(
                int(bar.ts),
                bar.symbol,
                self._no_trade_cfg,
                stage_cfg=self._pipeline_cfg.get("windows") if self._pipeline_cfg else None,
            )
            if win_res.action == "drop":
                try:
                    self._logger.info(
                        "DROP %s",
                        {
                            "stage": win_res.stage.name,
                            "reason": getattr(win_res.reason, "name", ""),
                        },
                    )
                except Exception:
                    pass
                try:
                    pipeline_stage_drop_count.labels(
                        bar.symbol,
                        win_res.stage.name,
                        win_res.reason.name if win_res.reason else "",
                    ).inc()
                except Exception:
                    pass
                return emitted

        pol_res = policy_decide(
            self._fp,
            self._policy,
            bar,
            stage_cfg=self._pipeline_cfg.get("policy") if self._pipeline_cfg else None,
        )
        if pol_res.action == "drop":
            try:
                self._logger.info(
                    "DROP %s",
                    {
                        "stage": pol_res.stage.name,
                        "reason": getattr(pol_res.reason, "name", ""),
                    },
                )
            except Exception:
                pass
            try:
                pipeline_stage_drop_count.labels(
                    bar.symbol,
                    pol_res.stage.name,
                    pol_res.reason.name if pol_res.reason else "",
                ).inc()
            except Exception:
                pass
            return emitted
        orders = list(pol_res.decision or [])

        risk_res = apply_risk(
            int(bar.ts),
            bar.symbol,
            self._guards,
            orders,
            stage_cfg=self._pipeline_cfg.get("risk") if self._pipeline_cfg else None,
        )
        if risk_res.action == "drop":
            try:
                self._logger.info(
                    "DROP %s",
                    {
                        "stage": risk_res.stage.name,
                        "reason": getattr(risk_res.reason, "name", ""),
                    },
                )
            except Exception:
                pass
            try:
                pipeline_stage_drop_count.labels(
                    bar.symbol,
                    risk_res.stage.name,
                    risk_res.reason.name if risk_res.reason else "",
                ).inc()
            except Exception:
                pass
            return emitted
        orders = list(risk_res.decision or [])

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
            res = self.publish_decision(
                o,
                bar.symbol,
                int(bar.ts),
                stage_cfg=self._pipeline_cfg.get("publish") if self._pipeline_cfg else None,
            )
            if res.action == "pass":
                emitted.append(o)
            elif res.action == "drop":
                try:
                    self._logger.info(
                        "DROP %s",
                        {
                            "stage": res.stage.name,
                            "reason": getattr(res.reason, "name", ""),
                        },
                    )
                except Exception:
                    pass
                try:
                    pipeline_stage_drop_count.labels(
                        bar.symbol,
                        res.stage.name,
                        res.reason.name if res.reason else "",
                    ).inc()
                except Exception:
                    pass

        try:
            monitoring.queue_len.set(len(self._queue) if self._queue else 0)
        except Exception:
            pass

        if self._ws_dedup_enabled and close_ms is not None:
            try:
                signal_bus.update(bar.symbol, close_ms)
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
        no_trade_cfg: NoTradeConfig | None = None,
        pipeline_cfg: PipelineConfig | None = None,
        shutdown_cfg: Dict[str, Any] | None = None,
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
        self.no_trade_cfg = no_trade_cfg
        self.pipeline_cfg = pipeline_cfg
        self.shutdown_cfg = shutdown_cfg or {}
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

        logs_dir = self.cfg.logs_dir or "logs"
        marker_path = Path(self.cfg.marker_path or os.path.join(logs_dir, "shutdown.marker"))
        dirty_restart = True
        try:
            if marker_path.exists():
                dirty_restart = False
            marker_path.unlink()
        except Exception:
            dirty_restart = True
        if dirty_restart:
            try:
                state_store.load()
            except Exception:
                pass
            try:
                monitoring.reset_kill_switch_counters()
            except Exception:
                pass

        shutdown = ShutdownManager(self.shutdown_cfg)
        stop_event = threading.Event()
        shutdown.on_stop(stop_event.set)

        self.feature_pipe.warmup()
        monitoring.configure_kill_switch(self.cfg.kill_switch)

        ops_cfg = {
            "rest_limit": self.cfg.kill_switch_ops.error_limit,
            "ws_limit": self.cfg.kill_switch_ops.error_limit,
            "duplicate_limit": self.cfg.kill_switch_ops.duplicate_limit,
            "stale_limit": self.cfg.kill_switch_ops.stale_intervals_limit,
            "reset_cooldown_sec": self.cfg.kill_switch_ops.reset_cooldown_sec,
        }
        if self.cfg.kill_switch_ops.flag_path:
            ops_cfg["flag_path"] = self.cfg.kill_switch_ops.flag_path
        if self.cfg.kill_switch_ops.alert_command:
            ops_cfg["alert_command"] = shlex.split(
                self.cfg.kill_switch_ops.alert_command
            )
        ops_kill_switch.init(ops_cfg)

        ops_flush_stop = threading.Event()
        rest_failures = 0

        def _ops_flush_loop() -> None:
            nonlocal rest_failures
            limit = int(self.cfg.kill_switch_ops.error_limit)
            while not ops_flush_stop.wait(1.0):
                try:
                    ops_kill_switch.tick()
                    if rest_failures:
                        try:
                            ops_kill_switch.manual_reset()
                        except Exception:
                            pass
                        rest_failures = 0
                except Exception:
                    rest_failures += 1
                    if limit > 0 and rest_failures >= limit:
                        try:
                            ops_kill_switch.record_error("rest")
                        except Exception:
                            pass
                        rest_failures = 0

        ops_flush_thread = threading.Thread(target=_ops_flush_loop, daemon=True)
        ops_flush_thread.start()
        shutdown.on_flush(ops_kill_switch.tick)
        shutdown.on_finalize(ops_flush_stop.set)
        shutdown.on_finalize(lambda: ops_flush_thread.join(timeout=1.0))

        worker = _Worker(
            self.feature_pipe,
            self.policy,
            self.logger,
            self.adapter,
            self.risk_guards,
            lambda: self._clock_safe_mode
            or monitoring.kill_switch_triggered()
            or ops_kill_switch.tripped(),
            enforce_closed_bars=self.enforce_closed_bars,
            ws_dedup_enabled=self.ws_dedup_enabled,
            ws_dedup_log_skips=self.ws_dedup_log_skips,
            ws_dedup_timeframe_ms=self.ws_dedup_timeframe_ms,
            throttle_cfg=self.throttle_cfg,
            no_trade_cfg=self.no_trade_cfg,
            pipeline_cfg=self.pipeline_cfg,
        )

        out_csv = getattr(signal_bus, "OUT_CSV", None)
        if out_csv:
            try:
                signal_bus.OUT_WRITER = SignalCSVWriter(out_csv)
                shutdown.on_flush(signal_bus.OUT_WRITER.flush_fsync)
                shutdown.on_finalize(signal_bus.OUT_WRITER.close)
            except Exception:
                signal_bus.OUT_WRITER = None

        json_path = self.cfg.snapshot_metrics_json or os.path.join(logs_dir, "snapshot_metrics.json")
        csv_path = self.cfg.snapshot_metrics_csv or os.path.join(logs_dir, "snapshot_metrics.csv")
        snapshot_stop = threading.Event()
        snapshot_thread: threading.Thread | None = None
        if self.cfg.snapshot_metrics_sec > 0:
            def _snapshot_loop() -> None:
                while not snapshot_stop.wait(self.cfg.snapshot_metrics_sec):
                    try:
                        monitoring.snapshot_metrics(json_path, csv_path)
                    except Exception:
                        pass

            try:
                monitoring.snapshot_metrics(json_path, csv_path)
            except Exception:
                pass
            snapshot_thread = threading.Thread(target=_snapshot_loop, daemon=True)
            snapshot_thread.start()
            shutdown.on_stop(snapshot_stop.set)

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
            shutdown.on_stop(bus.close)
            shutdown.on_finalize(lambda: loop_thread.join(timeout=1.0) if loop_thread else None)

        ws_client = getattr(self.adapter, "ws", None) or getattr(self.adapter, "source", None)
        if ws_client is not None and hasattr(ws_client, "stop"):
            async def _stop_ws_client() -> None:
                await ws_client.stop()

            shutdown.on_stop(_stop_ws_client)

        shutdown.on_stop(worker._drain_queue)

        sim = getattr(self.adapter, "sim", None) or getattr(self.adapter, "_sim", None)
        signal_log = getattr(sim, "_logger", None) if sim is not None else None
        if signal_log is not None:
            if hasattr(signal_log, "flush_fsync"):
                shutdown.on_flush(signal_log.flush_fsync)
            elif hasattr(signal_log, "flush"):
                shutdown.on_flush(signal_log.flush)
        if hasattr(signal_bus, "flush"):
            shutdown.on_flush(signal_bus.flush)

        def _flush_snapshot() -> None:
            try:
                summary, json_str, _ = monitoring.snapshot_metrics(json_path, csv_path)
                self.logger.info("SNAPSHOT %s", json_str)
            except Exception:
                pass

        shutdown.on_flush(_flush_snapshot)
        shutdown.on_flush(state_store.save)

        def _write_marker() -> None:
            try:
                marker_path.touch()
            except Exception:
                pass

        def _final_summary() -> None:
            try:
                summary, json_str, _ = monitoring.snapshot_metrics(json_path, csv_path)
                self.logger.info("SUMMARY %s", json_str)
            except Exception:
                pass

        shutdown.on_finalize(_write_marker)
        if snapshot_thread is not None:
            shutdown.on_finalize(lambda: snapshot_thread.join(timeout=1.0))
        shutdown.on_finalize(lambda: self._clock_stop.set())
        shutdown.on_finalize(lambda: self._clock_thread.join(timeout=self.clock_sync_cfg.refresh_sec if self.clock_sync_cfg else 1.0) if self._clock_thread is not None else None)
        shutdown.on_finalize(lambda: signal_bus.shutdown() if signal_bus.ENABLED else None)
        shutdown.on_finalize(_final_summary)

        shutdown.register(signal.SIGINT, signal.SIGTERM)

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

        ws_failures = 0
        limit = int(self.cfg.kill_switch_ops.error_limit)
        while not stop_event.is_set():
            try:
                for rep in self.adapter.run_events(worker):
                    if stop_event.is_set():
                        break
                    if ws_failures:
                        try:
                            ops_kill_switch.manual_reset()
                        except Exception:
                            pass
                        ws_failures = 0
                    yield rep
                break
            except Exception:
                ws_failures += 1
                if limit > 0 and ws_failures >= limit:
                    try:
                        ops_kill_switch.record_error("ws")
                    except Exception:
                        pass
                    ws_failures = 0
                time.sleep(1.0)

        shutdown.unregister(signal.SIGINT, signal.SIGTERM)
        shutdown.request_shutdown()


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

    # Load runtime parameters
    rt_cfg: Dict[str, Any] = {}
    rt_cfg_path = Path("configs/runtime.yaml")
    if rt_cfg_path.exists():
        try:
            with rt_cfg_path.open("r", encoding="utf-8") as f:
                rt_cfg = yaml.safe_load(f) or {}
        except Exception:
            rt_cfg = {}

    # Queue configuration for the asynchronous event bus
    queue_cfg = rt_cfg.get("queue", {})
    queue_capacity = int(queue_cfg.get("capacity", 0))
    drop_policy = str(queue_cfg.get("drop_policy", "newest"))
    bus = EventBus(queue_size=queue_capacity, drop_policy=drop_policy)

    # WS deduplication overrides
    ws_cfg = rt_cfg.get("ws", {})
    cfg.ws_dedup.enabled = bool(ws_cfg.get("enabled", cfg.ws_dedup.enabled))
    cfg.ws_dedup.persist_path = str(ws_cfg.get("persist_path", cfg.ws_dedup.persist_path))
    cfg.ws_dedup.log_skips = bool(ws_cfg.get("log_skips", cfg.ws_dedup.log_skips))

    # Throttle configuration overrides
    throttle_cfg = rt_cfg.get("throttle", {})
    if throttle_cfg:
        cfg.throttle.enabled = bool(throttle_cfg.get("enabled", cfg.throttle.enabled))
        global_cfg = throttle_cfg.get("global", {})
        cfg.throttle.global_.rps = float(global_cfg.get("rps", cfg.throttle.global_.rps))
        cfg.throttle.global_.burst = int(global_cfg.get("burst", cfg.throttle.global_.burst))
        sym_cfg = throttle_cfg.get("symbol", {})
        cfg.throttle.symbol.rps = float(sym_cfg.get("rps", cfg.throttle.symbol.rps))
        cfg.throttle.symbol.burst = int(sym_cfg.get("burst", cfg.throttle.symbol.burst))
        cfg.throttle.mode = str(throttle_cfg.get("mode", cfg.throttle.mode))
        q_cfg = throttle_cfg.get("queue", {})
        cfg.throttle.queue.max_items = int(q_cfg.get("max_items", cfg.throttle.queue.max_items))
        cfg.throttle.queue.ttl_ms = int(q_cfg.get("ttl_ms", cfg.throttle.queue.ttl_ms))
        cfg.throttle.time_source = str(throttle_cfg.get("time_source", cfg.throttle.time_source))

    # Kill switch overrides
    kill_cfg = rt_cfg.get("ops", {}).get("kill_switch", {})
    if kill_cfg:
        cfg.kill_switch.feed_lag_ms = float(
            kill_cfg.get("feed_lag_ms", cfg.kill_switch.feed_lag_ms)
        )
        cfg.kill_switch.ws_failures = float(
            kill_cfg.get("ws_failures", cfg.kill_switch.ws_failures)
        )
        cfg.kill_switch.error_rate = float(
            kill_cfg.get("error_rate", cfg.kill_switch.error_rate)
        )
        cfg.kill_switch_ops.enabled = bool(
            kill_cfg.get("enabled", cfg.kill_switch_ops.enabled)
        )
        cfg.kill_switch_ops.error_limit = int(
            kill_cfg.get("error_limit", cfg.kill_switch_ops.error_limit)
        )
        cfg.kill_switch_ops.duplicate_limit = int(
            kill_cfg.get("duplicate_limit", cfg.kill_switch_ops.duplicate_limit)
        )
        cfg.kill_switch_ops.stale_intervals_limit = int(
            kill_cfg.get(
                "stale_intervals_limit", cfg.kill_switch_ops.stale_intervals_limit
            )
        )
        cfg.kill_switch_ops.reset_cooldown_sec = int(
            kill_cfg.get(
                "reset_cooldown_sec", cfg.kill_switch_ops.reset_cooldown_sec
            )
        )
        cfg.kill_switch_ops.flag_path = kill_cfg.get(
            "flag_path", cfg.kill_switch_ops.flag_path
        )
        cfg.kill_switch_ops.alert_command = kill_cfg.get(
            "alert_command", cfg.kill_switch_ops.alert_command
        )

    # Pipeline configuration
    def _parse_pipeline(data: Dict[str, Any]) -> PipelineConfig:
        enabled = bool(data.get("enabled", True))
        stages_cfg = data.get("stages", {}) or {}
        stages: Dict[str, PipelineStageConfig] = {}
        for name, st in stages_cfg.items():
            if isinstance(st, dict):
                st_enabled = bool(st.get("enabled", True))
                params = {k: v for k, v in st.items() if k != "enabled"}
            else:
                st_enabled = bool(st)
                params = {}
            stages[name] = PipelineStageConfig(enabled=st_enabled, params=params)
        return PipelineConfig(enabled=enabled, stages=stages)

    base_pipeline = PipelineConfig()
    base_shutdown: Dict[str, Any] = {}
    if snapshot_config_path:
        try:
            with open(snapshot_config_path, "r", encoding="utf-8") as f:
                base_data = yaml.safe_load(f) or {}
            base_pipeline = _parse_pipeline(base_data.get("pipeline", {}))
            base_shutdown = base_data.get("shutdown", {}) or {}
        except Exception:
            base_pipeline = PipelineConfig()
            base_shutdown = {}

    ops_pipeline = PipelineConfig()
    ops_shutdown: Dict[str, Any] = {}
    ops_retry: Dict[str, Any] = {}
    ops_data: Dict[str, Any] = {}
    for name, loader in (
        ("configs/ops.yaml", yaml.safe_load),
        ("configs/ops.json", json.load),
    ):
        ops_path = Path(name)
        if ops_path.exists():
            try:
                with ops_path.open("r", encoding="utf-8") as f:
                    ops_data = loader(f) or {}
            except Exception:
                ops_data = {}
            break
    if ops_data:
        ops_pipeline = _parse_pipeline(ops_data.get("pipeline", {}))
        ops_shutdown = ops_data.get("shutdown", {}) or {}
        ops_retry = ops_data.get("retry", {}) or {}
    elif rt_cfg.get("pipeline"):
        ops_pipeline = _parse_pipeline(rt_cfg.get("pipeline", {}))

    pipeline_cfg = base_pipeline.merge(ops_pipeline)

    shutdown_cfg: Dict[str, Any] = {}
    shutdown_cfg.update(base_shutdown)
    shutdown_cfg.update(ops_shutdown)
    shutdown_cfg.update(rt_cfg.get("shutdown", {}))

    if ops_retry:
        cfg.retry.max_attempts = int(
            ops_retry.get("max_attempts", cfg.retry.max_attempts)
        )
        cfg.retry.backoff_base_s = float(
            ops_retry.get("backoff_base_s", cfg.retry.backoff_base_s)
        )
        cfg.retry.max_backoff_s = float(
            ops_retry.get("max_backoff_s", cfg.retry.max_backoff_s)
        )

    # Ensure components receive the bus if they accept it
    try:
        cfg.components.market_data.params.setdefault("bus", bus)
    except Exception:
        pass
    try:
        cfg.components.executor.params.setdefault("bus", bus)
    except Exception:
        pass

    svc_cfg = SignalRunnerConfig(
        snapshot_config_path=snapshot_config_path,
        artifacts_dir=cfg.artifacts_dir,
        logs_dir=cfg.logs_dir,
        marker_path=os.path.join(cfg.logs_dir, "shutdown.marker"),
        run_id=cfg.run_id,
        snapshot_metrics_json=os.path.join(cfg.logs_dir, "snapshot_metrics.json"),
        snapshot_metrics_csv=os.path.join(cfg.logs_dir, "snapshot_metrics.csv"),
    )
    sec = rt_cfg.get("ops", {}).get("snapshot_metrics_sec")
    if isinstance(sec, (int, float)) and sec > 0:
        svc_cfg.snapshot_metrics_sec = int(sec)

    container = di_registry.build_graph(cfg.components, cfg)
    adapter: SimAdapter = container["executor"]
    if not hasattr(adapter, "bus"):
        try:
            adapter.bus = bus
        except Exception:
            pass
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
        NoTradeConfig(**(cfg.no_trade or {})),
        pipeline_cfg=pipeline_cfg,
        shutdown_cfg=shutdown_cfg,
        enforce_closed_bars=cfg.timing.enforce_closed_bars,
        ws_dedup_enabled=cfg.ws_dedup.enabled,
        ws_dedup_log_skips=cfg.ws_dedup.log_skips,
        ws_dedup_timeframe_ms=cfg.timing.timeframe_ms,
    )
    return service.run()


__all__ = ["SignalRunnerConfig", "RunnerConfig", "ServiceSignalRunner", "from_config"]
