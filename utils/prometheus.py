"""Prometheus helper with graceful fallback.

Provides :class:`Counter` compatible with :mod:`prometheus_client` if
available.  When the dependency is missing, a no-op stub is used so that
metrics calls do not fail in environments without Prometheus support.
"""
from __future__ import annotations

try:  # pragma: no cover - simple import
    from prometheus_client import Counter, Histogram, Summary  # type: ignore
except Exception:  # pragma: no cover - fallback for missing dependency
    class _DummyCounter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def labels(self, *args, **kwargs) -> "_DummyCounter":
            return self

        def inc(self, *args, **kwargs) -> None:
            pass

    class _DummyHistogram:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def labels(self, *args, **kwargs) -> "_DummyHistogram":
            return self

        def observe(self, *args, **kwargs) -> None:
            pass

    class _DummySummary(_DummyHistogram):
        """Fallback Summary with Histogram-compatible API."""

        pass

    Counter = _DummyCounter  # type: ignore
    Histogram = _DummyHistogram  # type: ignore
    Summary = _DummySummary  # type: ignore

__all__ = ["Counter", "Histogram", "Summary"]
