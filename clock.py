"""Clock synchronization utilities."""

from __future__ import annotations

import math
import statistics
import time
from typing import List

from binance_public import BinancePublicClient
from core_config import ClockSyncConfig

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

    offsets: List[float] = []
    rtts: List[float] = []
    try:
        attempts = max(1, int(getattr(cfg, "attempts", 1)))
        for _ in range(attempts):
            server_ms, rtt_ms = client.get_server_time()
            local_ms = system_utc_ms()
            offsets.append(float(server_ms) + float(rtt_ms) / 2.0 - float(local_ms))
            rtts.append(float(rtt_ms))
    except Exception:
        if monitor is not None and hasattr(monitor, "clock_sync_fail"):
            try:
                monitor.clock_sync_fail.inc()
            except Exception:
                pass
        return float(clock_skew_ms)

    if not offsets:
        return float(clock_skew_ms)

    # Filter out samples with RTT above the 90th percentile
    if len(rtts) > 1:
        sorted_rtts = sorted(rtts)
        idx = max(0, int(math.ceil(len(sorted_rtts) * 0.9)) - 1)
        p90 = sorted_rtts[idx]
        filtered = [off for off, rtt in zip(offsets, rtts) if rtt <= p90]
        if filtered:
            offsets = filtered
    median_offset = statistics.median(offsets)

    alpha = float(getattr(cfg, "ema_alpha", 1.0))
    alpha = min(max(alpha, 0.0), 1.0)
    new_skew = (1.0 - alpha) * float(clock_skew_ms) + alpha * float(median_offset)
    step = new_skew - float(clock_skew_ms)
    max_step = float(getattr(cfg, "max_step_ms", 0.0))
    if max_step > 0 and abs(step) > max_step:
        new_skew = float(clock_skew_ms) + math.copysign(max_step, step)

    clock_skew_ms = float(new_skew)
    last_sync_at = float(system_utc_ms())
    return float(clock_skew_ms)
