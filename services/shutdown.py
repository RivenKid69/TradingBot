"""Utilities for graceful application shutdown."""

from __future__ import annotations

import asyncio
import signal
from typing import Any, Callable, Iterable, Awaitable, Dict, List, Optional


Callback = Callable[[], Any | Awaitable[Any]]


class ShutdownManager:
    """Co-ordinate application shutdown across asynchronous stages.

    The manager executes registered callbacks in three sequential phases:

    ``stop`` -> ``flush`` -> ``finalize``.
    Each phase can have a timeout configured.  Callbacks may be regular
    functions or coroutines.  ``request_shutdown`` kicks off the procedure and
    is safe to call multiple times.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Create a new manager from ``config``.

        Parameters
        ----------
        config:
            Mapping containing ``grace_period`` (float seconds), ``drain_policy``
            (string) and optional ``timeouts`` mapping with ``stop``, ``flush``
            and ``finalize`` entries specifying phase timeouts in seconds.
        """

        self.grace_period = float(config.get("grace_period", 0.0))
        self.drain_policy = str(config.get("drain_policy", "graceful"))
        timeouts = config.get("timeouts", {})
        self.stop_timeout = timeouts.get("stop")
        self.flush_timeout = timeouts.get("flush")
        self.finalize_timeout = timeouts.get("finalize")

        self._shutdown_requested = False
        self._shutdown_task: asyncio.Task[None] | None = None

        self._on_stop: List[Callback] = []
        self._on_flush: List[Callback] = []
        self._on_finalize: List[Callback] = []

        self._orig_handlers: Dict[int, Any] = {}

    # ------------------------------------------------------------------
    def on_stop(self, cb: Callback) -> None:
        """Register ``cb`` for the stop phase."""

        self._on_stop.append(cb)

    # ------------------------------------------------------------------
    def on_flush(self, cb: Callback) -> None:
        """Register ``cb`` for the flush phase."""

        self._on_flush.append(cb)

    # ------------------------------------------------------------------
    def on_finalize(self, cb: Callback) -> None:
        """Register ``cb`` for the finalize phase."""

        self._on_finalize.append(cb)

    # ------------------------------------------------------------------
    def register(self, *signals_: int) -> None:
        """Attach :func:`request_shutdown` to ``signals_``."""

        for sig in signals_:
            if sig not in self._orig_handlers:
                self._orig_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)

    # ------------------------------------------------------------------
    def unregister(self, *signals_: int) -> None:
        """Restore original handlers for ``signals_``."""

        for sig in signals_:
            orig = self._orig_handlers.pop(sig, None)
            if orig is not None:
                signal.signal(sig, orig)

    # ------------------------------------------------------------------
    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.request_shutdown(reason=f"signal {signum}")

    # ------------------------------------------------------------------
    def request_shutdown(self, *, reason: str | None = None) -> None:
        """Trigger the shutdown sequence once."""

        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._run_sequence())
            self._shutdown_task = None
        else:
            self._shutdown_task = loop.create_task(self._run_sequence(), name="shutdown")

    # ------------------------------------------------------------------
    async def _run_callbacks(self, cbs: Iterable[Callback], timeout: float | None) -> None:
        for cb in cbs:
            try:
                if asyncio.iscoroutinefunction(cb):
                    coro = cb()
                else:
                    res = cb()
                    coro = res if asyncio.iscoroutine(res) else None
                if coro is not None:
                    if timeout is not None:
                        await asyncio.wait_for(coro, timeout)
                    else:
                        await coro
            except Exception:
                pass

    # ------------------------------------------------------------------
    async def _run_sequence(self) -> None:
        await self._run_callbacks(self._on_stop, self.stop_timeout)
        if self.grace_period > 0:
            try:
                await asyncio.sleep(self.grace_period)
            except asyncio.CancelledError:
                return
        await self._run_callbacks(self._on_flush, self.flush_timeout)
        await self._run_callbacks(self._on_finalize, self.finalize_timeout)


__all__ = ["ShutdownManager"]

