import sys
import importlib.util
import pathlib
import random
import asyncio
import json
from types import SimpleNamespace

# Ensure stdlib logging is used instead of local logging.py
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_orig_sys_path = list(sys.path)
sys.path = [p for p in sys.path if p not in ("", str(REPO_ROOT))]
import logging as std_logging  # type: ignore
sys.modules["logging"] = std_logging
sys.path = _orig_sys_path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import logging

import pandas as pd
import pytest

from config import DataDegradationConfig
import impl_offline_data
from impl_offline_data import OfflineCSVConfig, OfflineCSVBarSource

class _DummyWS:
    pass
sys.modules.setdefault("websockets", _DummyWS())
import binance_ws
from services.event_bus import EventBus


def _simulate_bar_stream(n: int, cfg: DataDegradationConfig) -> tuple[int, int, int]:
    rng = random.Random(cfg.seed)
    drop = stale = delay = 0
    prev = False
    for _ in range(n):
        if rng.random() < cfg.drop_prob:
            drop += 1
            continue
        if prev and rng.random() < cfg.stale_prob:
            stale += 1
            if rng.random() < cfg.dropout_prob and rng.randint(0, cfg.max_delay_ms) > 0:
                delay += 1
            continue
        if rng.random() < cfg.dropout_prob and rng.randint(0, cfg.max_delay_ms) > 0:
            delay += 1
        prev = True
    return drop, stale, delay


def _simulate_latency_queue(n: int, cfg: DataDegradationConfig, step_ms: int) -> tuple[int, int, int]:
    rng = random.Random(cfg.seed)
    max_delay_steps = cfg.max_delay_ms // step_ms
    class P:
        def __init__(self) -> None:
            self.remaining = 0
            self.delayed = False
    queue = [P() for _ in range(n)]
    total = drop = delay = 0
    while queue:
        rest = []
        for p in queue:
            total += 1
            if rng.random() < cfg.drop_prob:
                drop += 1
                continue
            if rng.random() < cfg.dropout_prob:
                steps = rng.randint(0, max_delay_steps)
                if steps > 0:
                    p.remaining += steps
                    if not p.delayed:
                        delay += 1
                        p.delayed = True
            if p.remaining <= 0:
                continue
            p.remaining -= 1
            rest.append(p)
        queue = rest
    return total, drop, delay


def test_offline_csv_degradation_logging(tmp_path, monkeypatch, caplog):
    cfg = DataDegradationConfig(stale_prob=0.2, drop_prob=0.1, dropout_prob=0.5, max_delay_ms=5, seed=1)
    rows = [
        {"ts": i * 60_000, "symbol": "BTC", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
        for i in range(20)
    ]
    path = tmp_path / "bars.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    monkeypatch.setattr(impl_offline_data.time, "sleep", lambda _: None)
    caplog.set_level(logging.INFO, logger=impl_offline_data.__name__)
    src = OfflineCSVBarSource(OfflineCSVConfig(paths=[str(path)], timeframe="1m"), data_degradation=cfg)
    list(src.stream_bars(["BTC"], 60_000))

    drop, stale, delay = _simulate_bar_stream(20, cfg)
    total = 20
    rec = next(r for r in caplog.records if "OfflineCSVBarSource degradation" in r.message)
    assert f"drop={drop/total*100:.2f}% ({drop}/{total})" in rec.message
    assert f"stale={stale/total*100:.2f}% ({stale}/{total})" in rec.message
    assert f"delay={delay/total*100:.2f}% ({delay}/{total})" in rec.message


def test_binance_ws_degradation_logging(monkeypatch, caplog):
    async def run() -> None:
        cfg = DataDegradationConfig(stale_prob=0.2, drop_prob=0.1, dropout_prob=0.5, max_delay_ms=5, seed=1)
        messages = [
            json.dumps({"data": {"k": {"x": True, "t": i, "s": "BTCUSDT", "o": "1", "h": "1", "l": "1", "c": "1", "v": "1", "n": 1}}})
            for i in range(20)
        ]
        bus = EventBus(queue_size=100, drop_policy="newest")
        client = binance_ws.BinanceWS(symbols=["BTCUSDT"], bus=bus, data_degradation=cfg)

        class MockWS:
            def __init__(self, msgs):
                self.msgs = list(msgs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                await client.stop()

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.msgs:
                    raise StopAsyncIteration
                return self.msgs.pop(0)

            async def ping(self):
                fut = asyncio.Future()
                fut.set_result(None)
                return fut

            async def send(self, _msg):
                return None

            async def close(self):
                return None

        async def dummy_sleep(_):
            pass

        monkeypatch.setattr(binance_ws, "websockets", SimpleNamespace(connect=lambda *a, **k: MockWS(messages)))
        monkeypatch.setattr(binance_ws.asyncio, "sleep", dummy_sleep)
        caplog.set_level(logging.INFO, logger=binance_ws.__name__)

        await client.run_forever()

        drop, stale, delay = _simulate_bar_stream(20, cfg)
        total = 20
        rec = next(r for r in caplog.records if "BinanceWS degradation" in r.message)
        assert f"drop={drop/total*100:.2f}% ({drop}/{total})" in rec.message
        assert f"stale={stale/total*100:.2f}% ({stale}/{total})" in rec.message
        assert f"delay={delay/total*100:.2f}% ({delay}/{total})" in rec.message

    asyncio.run(run())


def test_execution_simulator_degradation_logging(caplog):
    spec = importlib.util.spec_from_file_location("execution_sim", REPO_ROOT / "execution_sim.py")
    exec_mod = importlib.util.module_from_spec(spec)
    sys.modules["execution_sim"] = exec_mod
    spec.loader.exec_module(exec_mod)
    ExecutionSimulator = exec_mod.ExecutionSimulator
    ActionProto = exec_mod.ActionProto
    ActionType = exec_mod.ActionType
    Pending = exec_mod.Pending
    from types import SimpleNamespace

    cfg = DataDegradationConfig(drop_prob=0.2, dropout_prob=1.0, max_delay_ms=1, seed=1)
    run_cfg = SimpleNamespace(step_ms=1)
    sim = ExecutionSimulator(latency_steps=0, data_degradation=cfg, run_config=run_cfg)

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)
    for _ in range(5):
        sim._q.push(Pending(proto=proto, client_order_id=1, remaining_lat=0, timestamp=0))

    caplog.set_level(logging.INFO, logger=exec_mod.__name__)
    while sim._q._q:
        sim._q.pop_ready()
    sim.stop()

    total, drop, delay = _simulate_latency_queue(5, cfg, step_ms=1)
    rec = next(r for r in caplog.records if "LatencyQueue degradation" in r.message)
    assert f"drop={drop/total*100:.2f}% ({drop}/{total})" in rec.message
    assert f"delay={delay/total*100:.2f}% ({delay}/{total})" in rec.message
