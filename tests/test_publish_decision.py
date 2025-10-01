import sys
from pathlib import Path
from types import SimpleNamespace
import logging

sys.path.append(str(Path(__file__).resolve().parents[1]))

from service_signal_runner import _Worker


def _make_worker(mode: str) -> _Worker:
    throttle_cfg = SimpleNamespace(
        enabled=True,
        global_=SimpleNamespace(rps=0.0, burst=1.0),
        symbol=SimpleNamespace(rps=0.0, burst=1.0),
        mode=mode,
        queue=SimpleNamespace(ttl_ms=1000, max_items=10),
    )
    worker = _Worker(
        fp=SimpleNamespace(),
        policy=SimpleNamespace(),
        logger=logging.getLogger("test"),
        executor=SimpleNamespace(submit=lambda o: None),
        guards=None,
        enforce_closed_bars=False,
        throttle_cfg=throttle_cfg,
    )
    return worker


def test_publish_decision_queue_mode():
    worker = _make_worker("queue")
    called = []
    def _emit(order, symbol, bar_close_ms, *, bar_open_ms=None):
        called.append(order)
        return True

    worker._emit = _emit  # type: ignore[method-assign]
    order = SimpleNamespace()
    res = worker.publish_decision(order, "BTC", 1)
    assert res.action == "queue"
    assert len(worker._queue) == 1
    assert called == []


def test_publish_decision_drop_mode():
    worker = _make_worker("drop")
    called = []
    def _emit(order, symbol, bar_close_ms, *, bar_open_ms=None):
        called.append(order)
        return True

    worker._emit = _emit  # type: ignore[method-assign]
    order = SimpleNamespace()
    res = worker.publish_decision(order, "BTC", 1)
    assert res.action == "drop"
    assert len(worker._queue or []) == 0
    assert called == []
