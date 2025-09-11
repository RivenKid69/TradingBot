"""Prometheus helper with graceful fallback.

Provides :class:`Counter` compatible with :mod:`prometheus_client` if
available.  When the dependency is missing, a no-op stub is used so that
metrics calls do not fail in environments without Prometheus support.
"""
from __future__ import annotations

try:  # pragma: no cover - simple import
    from prometheus_client import Counter  # type: ignore
except Exception:  # pragma: no cover - fallback for missing dependency
    class _DummyCounter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def labels(self, *args, **kwargs) -> "_DummyCounter":
            return self

        def inc(self, *args, **kwargs) -> None:
            pass

    Counter = _DummyCounter  # type: ignore
