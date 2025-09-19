"""Binance Spot Private API helpers."""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping
from urllib import parse

try:  # pragma: no cover - optional dependency in tests
    import requests
except Exception:  # pragma: no cover - allow lazy import in functions
    requests = None  # type: ignore[assignment]

from core_config import RetryConfig
from services.retry import retry_sync

logger = logging.getLogger(__name__)

# Default retry configuration for private requests
DEFAULT_RETRY_CFG = RetryConfig(max_attempts=5, backoff_base_s=0.5, max_backoff_s=60.0)

_DEFAULT_SAPI_BASE = "https://api.binance.com"
_ACCOUNT_INFO_PATH = "/sapi/v1/account"


def _no_retry(_: Exception) -> str | None:
    """Placeholder classifier for retry logic used by stubbed operations."""

    return None


def _classify_private_error(exc: Exception) -> str | None:
    """Map exceptions raised by private REST calls to kill-switch buckets."""

    global requests  # reuse optional import if available
    try:  # pragma: no cover - requests may be unavailable in tests
        import requests as _requests  # type: ignore
        requests = _requests
    except Exception:  # pragma: no cover - fallback to stub
        _requests = None

    if _requests is None:
        return "binance-private-error"

    if isinstance(exc, _requests.exceptions.Timeout):
        return "binance-private-timeout"
    if isinstance(exc, _requests.exceptions.ConnectionError):
        return "binance-private-connection"
    if isinstance(exc, _requests.exceptions.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (418, 429) or (isinstance(status, int) and status >= 500):
            return "binance-private-http"
        return None
    return "binance-private-error"


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_positive_int(value: Any) -> int | None:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    if ivalue < 0:
        return None
    return ivalue


def _commission_to_bps(value: Any) -> float | None:
    rate = _safe_float(value)
    if rate is None:
        return None
    return rate * 10_000.0


@dataclass
class AccountFeeInfo:
    """Normalised fee information returned by :func:`fetch_account_fee_info`."""

    vip_tier: int | None = None
    maker_bps: float | None = None
    taker_bps: float | None = None
    maker_rate: float | None = None
    taker_rate: float | None = None
    update_time_ms: int | None = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_fee_overrides(self) -> Dict[str, Any]:
        """Return a compact dict with values consumable by :mod:`impl_fees`."""

        overrides: Dict[str, Any] = {}
        if self.maker_bps is not None:
            overrides["maker_bps"] = float(self.maker_bps)
        if self.taker_bps is not None:
            overrides["taker_bps"] = float(self.taker_bps)
        if self.vip_tier is not None:
            overrides["vip_tier"] = int(self.vip_tier)
        return overrides


def _parse_account_payload(payload: Mapping[str, Any]) -> AccountFeeInfo:
    vip_tier = _safe_positive_int(payload.get("vipLevel") or payload.get("vipTier"))

    maker_rate = None
    taker_rate = None
    maker_bps = None
    taker_bps = None

    rates_block = payload.get("commissionRates")
    if isinstance(rates_block, Mapping):
        maker_rate = _safe_float(rates_block.get("maker"))
        taker_rate = _safe_float(rates_block.get("taker"))
        if maker_rate is not None:
            maker_bps = _commission_to_bps(maker_rate)
        if taker_rate is not None:
            taker_bps = _commission_to_bps(taker_rate)

    maker_commission = _safe_float(payload.get("makerCommission"))
    taker_commission = _safe_float(payload.get("takerCommission"))
    if maker_bps is None and maker_commission is not None:
        maker_bps = maker_commission
        if maker_rate is None:
            maker_rate = maker_commission / 10_000.0
    if taker_bps is None and taker_commission is not None:
        taker_bps = taker_commission
        if taker_rate is None:
            taker_rate = taker_commission / 10_000.0

    update_time = _safe_positive_int(payload.get("updateTime"))

    raw: Dict[str, Any] = {
        "vipLevel": vip_tier,
        "makerCommission": maker_commission,
        "takerCommission": taker_commission,
        "commissionRates": {
            "maker": maker_rate,
            "taker": taker_rate,
        },
    }
    if update_time is not None:
        raw["updateTime"] = update_time

    return AccountFeeInfo(
        vip_tier=vip_tier,
        maker_bps=maker_bps,
        taker_bps=taker_bps,
        maker_rate=maker_rate,
        taker_rate=taker_rate,
        update_time_ms=update_time,
        raw=raw,
    )


@retry_sync(DEFAULT_RETRY_CFG, _classify_private_error)
def fetch_account_fee_info(
    *,
    api_key: str,
    api_secret: str,
    base_url: str | None = None,
    recv_window_ms: int | None = 5_000,
    timeout: float | None = 10.0,
    session: Any | None = None,
) -> AccountFeeInfo:
    """Fetch VIP tier and maker/taker rates from ``/sapi/v1/account``."""

    if not api_key or not api_secret:
        raise ValueError("api_key and api_secret are required for account info fetch")

    base = str(base_url or _DEFAULT_SAPI_BASE).rstrip("/")
    url = f"{base}{_ACCOUNT_INFO_PATH}"

    params: Dict[str, Any] = {"timestamp": int(time.time() * 1000)}
    if recv_window_ms:
        params["recvWindow"] = int(recv_window_ms)

    query = parse.urlencode(params, doseq=True)
    secret_bytes = api_secret.encode("utf-8")
    signature = hmac.new(secret_bytes, query.encode("utf-8"), hashlib.sha256).hexdigest()
    signed_params = dict(params)
    signed_params["signature"] = signature

    headers = {"X-MBX-APIKEY": api_key}
    timeout_s = None if timeout is None else float(timeout)

    global requests
    if session is None:
        if requests is None:  # pragma: no cover - resolved in classify helper
            try:
                import requests as _requests  # type: ignore
            except Exception as exc:  # pragma: no cover - propagate descriptive error
                raise RuntimeError("requests dependency is required for private API calls") from exc

            requests = _requests
        request_func = getattr(requests, "get", None)
    else:
        request_func = getattr(session, "get", None)

    if request_func is None:
        raise RuntimeError("HTTP client does not provide a 'get' method")

    logger.debug("Fetching Binance account info from %s", url)
    response = request_func(url, params=signed_params, headers=headers, timeout=timeout_s)
    try:
        response.raise_for_status()
    except AttributeError as exc:  # pragma: no cover - custom session interface
        raise RuntimeError("HTTP response object missing raise_for_status") from exc

    try:
        payload = response.json()
    except AttributeError as exc:  # pragma: no cover - non requests responses
        raise RuntimeError("HTTP response object missing json() method") from exc
    except ValueError as exc:
        raise RuntimeError("Failed to decode account info response") from exc

    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Unexpected account info payload: {type(payload).__name__}")

    logger.debug("Received Binance account info snapshot: keys=%s", list(payload.keys()))
    return _parse_account_payload(payload)


@retry_sync(DEFAULT_RETRY_CFG, _no_retry)
def place_order(*args: Any, **kwargs: Any):
    """Stub for placing an order on Binance Spot."""

    raise NotImplementedError("Binance spot private API is not yet connected")


@retry_sync(DEFAULT_RETRY_CFG, _no_retry)
def cancel_order(*args: Any, **kwargs: Any):
    """Stub for cancelling an order on Binance Spot."""

    raise NotImplementedError("Binance spot private API is not yet connected")


@retry_sync(DEFAULT_RETRY_CFG, _no_retry)
def reconcile_state(local_state, client) -> Dict[str, Any]:
    """Fetch remote state and compare with ``local_state``."""

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
