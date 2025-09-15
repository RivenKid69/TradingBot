"""Binance Spot Private API adapter stubs.

These functions will require an authenticated connection once private
mode is supported.
"""
from __future__ import annotations

from core_config import RetryConfig
from services.retry import retry_sync

# Default retry configuration for private requests
DEFAULT_RETRY_CFG = RetryConfig(max_attempts=5, backoff_base_s=0.5, max_backoff_s=60.0)


def _no_retry(_: Exception) -> str | None:
    """Placeholder classifier for retry logic."""
    return None


@retry_sync(DEFAULT_RETRY_CFG, _no_retry)
def place_order(*args, **kwargs):
    """Stub for placing an order on Binance Spot.

    TODO: Implement actual connection and request signing when private
    trading mode becomes available.
    """
    raise NotImplementedError("Binance spot private API is not yet connected")


@retry_sync(DEFAULT_RETRY_CFG, _no_retry)
def cancel_order(*args, **kwargs):
    """Stub for cancelling an order on Binance Spot.

    TODO: Implement actual connection and request signing when private
    trading mode becomes available.
    """
    raise NotImplementedError("Binance spot private API is not yet connected")
