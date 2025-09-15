"""Binance Spot Private API adapter stubs.

These functions will require an authenticated connection once private
mode is supported.
"""
from __future__ import annotations

from core_config import RetryConfig
from services.retry import retry_sync
from typing import Any, Dict

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


@retry_sync(DEFAULT_RETRY_CFG, _no_retry)
def reconcile_state(local_state, client) -> Dict[str, Any]:
    """Fetch remote state and compare with ``local_state``.

    This function queries Binance Spot private REST endpoints for
    account balances and open orders, then compares them with the
    provided ``local_state`` (typically loaded from persistence).

    Returns a summary dictionary describing discrepancies between
    local and remote data.
    """

    try:
        remote_orders = client.get_open_orders() or []
        account = client.get_account()
        balances = account.get("balances", [])
    except Exception as e:  # pragma: no cover - network/auth errors
        raise RuntimeError("failed to fetch remote state") from e

    remote_order_ids = {
        str(o.get("orderId") or o.get("clientOrderId")) for o in remote_orders
    }
    local_orders = {
        str(oid): data for oid, data in getattr(local_state, "open_orders", {}).items()
    }
    missing_open = sorted(remote_order_ids - local_orders.keys())
    extra_open = sorted(local_orders.keys() - remote_order_ids)

    remote_positions = {
        str(b.get("asset")):
        float(b.get("free", 0.0)) + float(b.get("locked", 0.0))
        for b in balances
    }
    local_positions = {
        str(sym): float(qty)
        for sym, qty in getattr(local_state, "positions", {}).items()
    }
    position_diffs: Dict[str, Dict[str, float]] = {}
    for asset in set(remote_positions) | set(local_positions):
        r = remote_positions.get(asset, 0.0)
        l = local_positions.get(asset, 0.0)
        if abs(r - l) > 1e-8:
            position_diffs[asset] = {"local": l, "remote": r}

    return {
        "missing_open_orders": missing_open,
        "extra_open_orders": extra_open,
        "position_diffs": position_diffs,
    }
