import types
from types import SimpleNamespace

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import clock
from core_config import ClockSyncConfig


class DummyCounter:
    def __init__(self):
        self.count = 0

    def inc(self):
        self.count += 1


class DummyMonitor:
    def __init__(self):
        self.clock_sync_fail = DummyCounter()


def test_sync_clock_updates_skew(monkeypatch):
    clock.clock_skew_ms = 0.0
    clock.last_sync_at = 0.0

    times = iter([1000, 2000, 3000, 4000])
    monkeypatch.setattr(clock, "system_utc_ms", lambda: next(times))

    class DummyClient:
        def __init__(self):
            self.calls = 0
            self.responses = [(1075, 50), (2050, 100), (3095, 10)]

        def get_server_time(self):
            resp = self.responses[self.calls]
            self.calls += 1
            return resp

    cfg = ClockSyncConfig(attempts=3, ema_alpha=1.0, max_step_ms=1000.0)
    monitor = DummyMonitor()
    client = DummyClient()

    drift = clock.sync_clock(client, cfg, monitor)
    assert drift == clock.clock_skew()
    assert clock.clock_skew() == 100.0
    assert clock.last_sync_at == 4000

    monkeypatch.setattr(clock, "system_utc_ms", lambda: 4500)
    assert clock.last_sync_age_sec() == 0.5


def test_sync_clock_failure(monkeypatch):
    clock.clock_skew_ms = 42.0
    clock.last_sync_at = 0.0

    class BadClient:
        def get_server_time(self):
            raise RuntimeError("boom")

    cfg = ClockSyncConfig(attempts=1)
    monitor = DummyMonitor()
    drift = clock.sync_clock(BadClient(), cfg, monitor)
    assert drift == 42.0
    assert clock.clock_skew() == 42.0
    assert clock.last_sync_at == 0.0
    assert monitor.clock_sync_fail.count == 1
