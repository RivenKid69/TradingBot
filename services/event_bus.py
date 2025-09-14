"""Asynchronous in-memory event bus with backpressure handling.

The bus maintains a bounded queue of events.  When the queue is full new
events are dropped according to ``drop_policy``:

``"oldest"``  – remove the oldest queued event and enqueue the new one.

``"newest"``  – drop the incoming event.

Simple Prometheus metrics are emitted via :mod:`services.monitoring`:

``queue_depth`` -- current queue size

``events_in`` -- total number of events accepted into the queue

``dropped_bp`` -- events dropped because of backpressure
"""
from __future__ import annotations

import asyncio
from typing import Any

from . import monitoring


class EventBus:
    """Lightweight asynchronous event bus with drop policies."""

    def __init__(self, queue_size: int, drop_policy: str) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max(0, int(queue_size)))
        if drop_policy not in {"oldest", "newest", "drop_oldest", "drop_newest"}:
            raise ValueError("drop_policy must be 'oldest' or 'newest'")
        self._drop_oldest = drop_policy in {"oldest", "drop_oldest"}
        self._closed = False
        self._sentinel: object = object()

    # ------------------------------------------------------------------
    def _set_depth(self) -> None:
        try:
            monitoring.queue_depth.set(self._queue.qsize())
        except Exception:
            pass

    # ------------------------------------------------------------------
    async def put(self, event: Any) -> None:
        """Put ``event`` into the queue honoring the drop policy."""

        if self._closed:
            raise RuntimeError("EventBus is closed")
        try:
            self._queue.put_nowait(event)
            try:
                monitoring.events_in.inc()
            except Exception:
                pass
        except asyncio.QueueFull:
            try:
                monitoring.dropped_bp.inc()
            except Exception:
                pass
            if self._drop_oldest:
                try:
                    # Discard oldest event
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._queue.put_nowait(event)
                    try:
                        monitoring.events_in.inc()
                    except Exception:
                        pass
                except asyncio.QueueFull:
                    # Queue size zero, drop new event as well
                    pass
            else:
                # Drop newest – do nothing
                pass
        self._set_depth()

    # ------------------------------------------------------------------
    async def get(self) -> Any:
        """Return the next event or ``None`` after :meth:`close`."""

        item = await self._queue.get()
        self._set_depth()
        if item is self._sentinel:
            # keep sentinel for other consumers
            await self._queue.put(self._sentinel)
            return None
        return item

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Signal that no more events will be published."""

        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(self._sentinel)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(self._sentinel)
            except asyncio.QueueFull:
                pass
        self._set_depth()

    # ------------------------------------------------------------------
    async def __aenter__(self) -> "EventBus":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def __enter__(self) -> "EventBus":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["EventBus"]

