"""Clock synchronization utilities."""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, List, Mapping, Tuple

from binance_public import BinancePublicClient
from core_config import ClockSyncConfig
from services.rest_budget import RestBudgetSession

# Global clock skew in milliseconds (server - local) and last sync timestamp
clock_skew_ms: float = 0.0
last_sync_at: float = 0.0


def system_utc_ms() -> int:
    """Return current system time in milliseconds since the epoch."""
    return int(time.time() * 1000.0)


def now_ms() -> int:
    """Return current corrected time accounting for clock skew."""
    return int(system_utc_ms() + clock_skew_ms)


def clock_skew() -> float:
    """Return current clock skew in milliseconds."""
    return float(clock_skew_ms)


def last_sync_age_sec() -> float:
    """Return age of last successful sync in seconds."""
    if last_sync_at <= 0:
        return float("inf")
    return (system_utc_ms() - last_sync_at) / 1000.0


def _collect_sync_samples(
    client: BinancePublicClient, attempts: int
) -> Tuple[List[float], List[float]]:
    offsets: List[float] = []
    rtts: List[float] = []
    for _ in range(max(1, attempts)):
        server_ms, rtt_ms = client.get_server_time()
        local_ms = system_utc_ms()
        offsets.append(
            float(server_ms) + float(rtt_ms) / 2.0 - float(local_ms)
        )
        rtts.append(float(rtt_ms))
    return offsets, rtts


def _apply_sync_samples(
    offsets: List[float], rtts: List[float], cfg: ClockSyncConfig
) -> Tuple[float, float]:
    global clock_skew_ms, last_sync_at

    if not offsets:
        return float(clock_skew_ms), 0.0

    filtered_offsets = list(offsets)
    filtered_rtts = list(rtts)
    if len(filtered_rtts) > 1:
        sorted_rtts = sorted(filtered_rtts)
        idx = max(0, int(math.ceil(len(sorted_rtts) * 0.9)) - 1)
        p90 = sorted_rtts[idx]
        paired = [
            (off, rtt)
            for off, rtt in zip(filtered_offsets, filtered_rtts)
            if rtt <= p90
        ]
        if paired:
            filtered_offsets = [off for off, _ in paired]
            filtered_rtts = [rtt for _, rtt in paired]

    median_offset = statistics.median(filtered_offsets)
    alpha = float(getattr(cfg, "ema_alpha", 1.0))
    alpha = min(max(alpha, 0.0), 1.0)
    new_skew = (1.0 - alpha) * float(clock_skew_ms) + alpha * float(median_offset)
    step = new_skew - float(clock_skew_ms)
    max_step = float(getattr(cfg, "max_step_ms", 0.0))
    if max_step > 0 and abs(step) > max_step:
        new_skew = float(clock_skew_ms) + math.copysign(max_step, step)

    clock_skew_ms = float(new_skew)
    last_sync_at = float(system_utc_ms())
    median_rtt = statistics.median(filtered_rtts) if filtered_rtts else 0.0
    return float(clock_skew_ms), float(median_rtt)


def sync_clock(client: BinancePublicClient, cfg: ClockSyncConfig, monitor) -> float:
    """Synchronize local clock with exchange server time.

    Parameters
    ----------
    client : BinancePublicClient
        Client used to fetch server time.
    cfg : ClockSyncConfig
        Configuration parameters controlling sync behaviour.
    monitor : object
        Monitoring object with ``clock_sync_fail`` counter (optional).

    Returns
    -------
    float
        Updated clock skew in milliseconds.
    """
    global clock_skew_ms, last_sync_at

    try:
        attempts = max(1, int(getattr(cfg, "attempts", 1)))
    except Exception:
        attempts = 1
    try:
        offsets, rtts = _collect_sync_samples(client, attempts)
    except Exception:
        if monitor is not None and hasattr(monitor, "clock_sync_fail"):
            try:
                monitor.clock_sync_fail.inc()
            except Exception:
                pass
        return float(clock_skew_ms)
    skew, _ = _apply_sync_samples(offsets, rtts, cfg)
    return skew


def manual_sync(
    clock_cfg: ClockSyncConfig | Mapping[str, Any],
    session: RestBudgetSession | None = None,
) -> Tuple[float, float]:
    """Perform a blocking clock sync returning the drift and median RTT."""

    if isinstance(clock_cfg, ClockSyncConfig):
        cfg = clock_cfg
    else:
        cfg = ClockSyncConfig.parse_obj(clock_cfg)

    rest_cfg: Mapping[str, Any] | None = None
    if isinstance(clock_cfg, Mapping):
        rest_cfg = clock_cfg.get("rest_session") or clock_cfg.get("rest_budget")
    else:
        rest_cfg = getattr(clock_cfg, "rest_session", None)
    if isinstance(rest_cfg, Mapping):
        session_cfg = dict(rest_cfg)
    else:
        session_cfg = {}

    created_session = False
    sess = session
    if sess is None:
        sess = RestBudgetSession(session_cfg)
        created_session = True
    try:
        client = BinancePublicClient(session=sess)
        try:
            attempts = max(1, int(getattr(cfg, "attempts", 1)))
        except Exception:
            attempts = 1
        offsets, rtts = _collect_sync_samples(client, attempts)
        skew, median_rtt = _apply_sync_samples(offsets, rtts, cfg)
        return skew, median_rtt
    finally:
        if created_session:
            try:
                sess.close()
            except Exception:
                pass


__all__ = [
    "system_utc_ms",
    "now_ms",
    "clock_skew",
    "clock_skew_ms",
    "last_sync_at",
    "last_sync_age_sec",
    "sync_clock",
    "manual_sync",
]
