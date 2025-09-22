"""Helpers for building Binance spot fee snapshots from public data."""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import hmac
import json
import logging
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib import error as urlerror
from urllib import parse, request

logger = logging.getLogger(__name__)


EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
PUBLIC_FEE_URL = (
    "https://www.binance.com/bapi/asset/v1/public/asset-service/fee/get-product-fee-rate"
)
PRIVATE_TRADE_FEE_URL = "https://api.binance.com/sapi/v1/asset/tradeFee"

SCHEMA_VERSION = 1
DEFAULT_VIP_TIER_LABEL = "VIP 0"
DEFAULT_BNB_DISCOUNT_RATE = 0.25
DEFAULT_UPDATE_THRESHOLD_DAYS = 30
USER_AGENT = "TradingBot fee_refresh/1.0"


def ensure_aware(dt: _dt.datetime) -> _dt.datetime:
    """Return an aware datetime in UTC."""

    if dt.tzinfo is None:
        return dt.replace(tzinfo=_dt.timezone.utc)
    return dt.astimezone(_dt.timezone.utc)


def parse_timestamp(raw: Any) -> _dt.datetime | None:
    """Parse an ISO timestamp, returning ``None`` on failure."""

    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = _dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return ensure_aware(parsed)


def format_timestamp(ts: _dt.datetime) -> str:
    """Format an aware timestamp using Binance's canonical ``Z`` suffix."""

    ts = ensure_aware(ts).replace(microsecond=0)
    return ts.isoformat().replace("+00:00", "Z")


def _format_number(value: float | None) -> float | int | None:
    if value is None:
        return None
    rounded = round(float(value), 8)
    if abs(rounded - round(rounded)) < 1e-9:
        return int(round(rounded))
    return rounded


def _coerce_decimal(raw: Any) -> Decimal | None:
    if raw is None:
        return None
    if isinstance(raw, Decimal):
        return raw
    if isinstance(raw, (int, float)):
        return Decimal(str(raw))
    text = str(raw).strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _normalize_bps(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        upper = text.upper()
        if upper.endswith("BPS"):
            dec = _coerce_decimal(upper[:-3])
            return float(dec) if dec is not None else None
        if upper.endswith("%"):
            dec = _coerce_decimal(upper[:-1])
            if dec is None:
                return None
            return float(dec * 100)
    dec = _coerce_decimal(raw)
    if dec is None:
        return None
    if dec <= Decimal("0.5"):
        bps = float(dec * Decimal(10000))
        if bps > 1000:
            bps = float(dec * Decimal(100))
        return bps
    if dec <= Decimal("1000"):
        return float(dec)
    return None


@dataclass
class FeeRecord:
    symbol: str
    maker_bps: float | None = None
    taker_bps: float | None = None
    fee_rounding_step_bps: float | None = None
    bnb_discount_bps: float | None = None

    def merge(self, other: "FeeRecord") -> None:
        if other.maker_bps is not None:
            self.maker_bps = other.maker_bps
        if other.taker_bps is not None:
            self.taker_bps = other.taker_bps
        if other.fee_rounding_step_bps is not None:
            self.fee_rounding_step_bps = other.fee_rounding_step_bps
        if other.bnb_discount_bps is not None:
            self.bnb_discount_bps = other.bnb_discount_bps

    def to_payload(self) -> dict[str, float | int]:
        payload: dict[str, float | int] = {}
        maker = _format_number(self.maker_bps)
        if maker is not None:
            payload["maker_bps"] = maker
        taker = _format_number(self.taker_bps)
        if taker is not None:
            payload["taker_bps"] = taker
        rounding = _format_number(self.fee_rounding_step_bps)
        if rounding is not None:
            payload["fee_rounding_step_bps"] = rounding
        discount = _format_number(self.bnb_discount_bps)
        if discount is not None:
            payload["bnb_discount_bps"] = discount
        return payload


def _http_get_json(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: int = 30,
) -> Any:
    query = parse.urlencode(params or {}, doseq=True)
    full_url = url
    if query:
        delimiter = "&" if parse.urlparse(url).query else "?"
        full_url = f"{url}{delimiter}{query}"
    req = request.Request(full_url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            body = resp.read().decode(charset)
    except urlerror.HTTPError as exc:  # pragma: no cover - network failure
        detail = exc.read().decode("utf-8", "replace") if exc.fp else exc.reason
        raise RuntimeError(f"HTTP {exc.code} from {full_url}: {detail}") from exc
    except urlerror.URLError as exc:  # pragma: no cover - network failure
        raise RuntimeError(f"Failed to fetch {full_url}: {exc.reason}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {full_url}: {exc}") from exc


def _extract_fee_entries(payload: Any) -> Iterable[Mapping[str, Any]]:
    stack: list[Any] = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            symbol = item.get("symbol") or item.get("tradePair")
            maker_keys = {
                "maker_bps",
                "makerCommission",
                "makerRate",
                "makerFee",
                "makerFeeRate",
                "maker",
            }
            taker_keys = {
                "taker_bps",
                "takerCommission",
                "takerRate",
                "takerFee",
                "takerFeeRate",
                "taker",
            }
            if symbol and any(key in item for key in maker_keys | taker_keys):
                yield item
                continue
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _record_from_mapping(raw: Mapping[str, Any]) -> FeeRecord | None:
    symbol_raw = raw.get("symbol") or raw.get("tradePair")
    if not symbol_raw:
        return None
    symbol = str(symbol_raw).strip().upper()
    if not symbol:
        return None
    maker = None
    taker = None
    for key in ("maker_bps", "makerCommission", "makerRate", "makerFee", "makerFeeRate", "maker"):
        maker = _normalize_bps(raw.get(key))
        if maker is not None:
            break
    for key in ("taker_bps", "takerCommission", "takerRate", "takerFee", "takerFeeRate", "taker"):
        taker = _normalize_bps(raw.get(key))
        if taker is not None:
            break
    rounding = None
    for key in (
        "fee_rounding_step_bps",
        "feeRoundingStepBps",
        "feeRoundingStep",
        "roundingStep",
    ):
        rounding = _normalize_bps(raw.get(key))
        if rounding is not None:
            break
    discount = None
    for key in ("bnb_discount_bps", "discountBps", "bnbDiscountBps", "bnbDiscount"):
        discount = _normalize_bps(raw.get(key))
        if discount is not None:
            break
    return FeeRecord(symbol, maker, taker, rounding, discount)


def fetch_exchange_symbols(timeout: int = 30) -> set[str]:
    payload = _http_get_json(EXCHANGE_INFO_URL, timeout=timeout)
    symbols: set[str] = set()
    if not isinstance(payload, Mapping):
        raise RuntimeError("Unexpected exchangeInfo response structure")
    for item in payload.get("symbols", []):
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status", "")).upper()
        if status != "TRADING":
            continue
        if not (item.get("isSpotTradingAllowed") or "SPOT" in {str(p).upper() for p in item.get("permissions", [])}):
            continue
        symbol = str(item.get("symbol", "")).strip().upper()
        if symbol:
            symbols.add(symbol)
    return symbols


def _collect_from_public(url: str, timeout: int) -> tuple[dict[str, FeeRecord], str]:
    payload = _http_get_json(url, timeout=timeout)
    records: dict[str, FeeRecord] = {}
    for entry in _extract_fee_entries(payload):
        record = _record_from_mapping(entry)
        if record is None:
            continue
        existing = records.get(record.symbol)
        if existing:
            existing.merge(record)
        else:
            records[record.symbol] = record
    return records, f"Binance public fee endpoint {url}"


def _collect_from_private(
    *,
    api_key: str,
    api_secret: str,
    timeout: int,
) -> tuple[dict[str, FeeRecord], str]:
    params = {
        "timestamp": int(time.time() * 1000),
        "recvWindow": 5000,
    }
    query = parse.urlencode(params, doseq=True)
    signature = hmac.new(api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
    signed_url = f"{PRIVATE_TRADE_FEE_URL}?{query}&signature={signature}"
    headers = {"X-MBX-APIKEY": api_key}
    payload = _http_get_json(signed_url, headers=headers, timeout=timeout)
    records: dict[str, FeeRecord] = {}
    entries: Iterable[Mapping[str, Any]]
    if isinstance(payload, Mapping) and isinstance(payload.get("tradeFee"), list):
        entries = [entry for entry in payload.get("tradeFee", []) if isinstance(entry, Mapping)]
    elif isinstance(payload, list):
        entries = [entry for entry in payload if isinstance(entry, Mapping)]
    else:
        raise RuntimeError("Unexpected tradeFee response structure")
    for entry in entries:
        record = _record_from_mapping(entry)
        if record is None:
            continue
        existing = records.get(record.symbol)
        if existing:
            existing.merge(record)
        else:
            records[record.symbol] = record
    return records, "Binance private tradeFee endpoint"


def _collect_from_csv(path: Path) -> tuple[dict[str, FeeRecord], str]:
    records: dict[str, FeeRecord] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            record = _record_from_mapping(row)
            if record is None:
                continue
            existing = records.get(record.symbol)
            if existing:
                existing.merge(record)
            else:
                records[record.symbol] = record
    return records, f"CSV export {path}"


def _apply_discount(records: Mapping[str, FeeRecord], discount_rate: float) -> None:
    if discount_rate <= 0.0:
        return
    keep_fraction = 1.0 - discount_rate
    if keep_fraction <= 0.0:
        return
    for record in records.values():
        if record.bnb_discount_bps is None and record.taker_bps is not None:
            record.bnb_discount_bps = record.taker_bps * keep_fraction


def _build_payload(
    *,
    records: Mapping[str, FeeRecord],
    symbols: Iterable[str],
    vip_tier: str,
    source: str,
    metadata_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    now = ensure_aware(_dt.datetime.utcnow())
    metadata: dict[str, Any] = {
        "built_at": format_timestamp(now),
        "source": source,
        "vip_tier": vip_tier,
        "schema_version": SCHEMA_VERSION,
    }
    if metadata_overrides:
        for key, value in metadata_overrides.items():
            if value is None:
                continue
            metadata[key] = value
    fees: dict[str, dict[str, float | int]] = {}
    missing: list[str] = []
    for symbol in sorted({str(s).upper() for s in symbols}):
        record = records.get(symbol)
        if record is None:
            missing.append(symbol)
            continue
        payload = record.to_payload()
        if not payload:
            missing.append(symbol)
            continue
        fees[symbol] = payload
    if missing:
        logger.warning(
            "Missing fee information for %d symbols (e.g. %s)",
            len(missing),
            ", ".join(missing[:10]),
        )
    return {"metadata": metadata, "fees": fees}


def _most_common(values: Sequence[float]) -> float | None:
    if not values:
        return None
    counts: Counter[float] = Counter(round(v, 6) for v in values)
    key, _ = max(counts.items(), key=lambda kv: (kv[1], -abs(kv[0])))
    for value in values:
        if round(value, 6) == key:
            return value
    return values[0]


def _compute_discount_multiplier(records: Mapping[str, FeeRecord]) -> float | None:
    ratios: list[float] = []
    for record in records.values():
        if record.taker_bps and record.bnb_discount_bps:
            if record.taker_bps <= 0:
                continue
            ratios.append(record.bnb_discount_bps / record.taker_bps)
    if not ratios:
        return None
    try:
        return float(statistics.median(ratios))
    except statistics.StatisticsError:
        return float(ratios[0])


@dataclass
class PublicFeeSnapshot:
    payload: dict[str, Any]
    symbols: set[str]
    records: dict[str, FeeRecord]
    source: str
    vip_label: str
    vip_tier: int | None
    maker_bps_default: float | None
    taker_bps_default: float | None
    maker_discount_mult: float | None
    taker_discount_mult: float | None
    discount_rate: float | None
    use_bnb_discount: bool


def load_public_fee_snapshot(
    *,
    vip_tier: str = DEFAULT_VIP_TIER_LABEL,
    vip_tier_numeric: int | None = None,
    use_bnb_discount: bool | None = None,
    maker_discount_mult: float | None = None,
    taker_discount_mult: float | None = None,
    timeout: int = 30,
    public_url: str = PUBLIC_FEE_URL,
    csv_path: Path | str | None = None,
    api_key: str | None = None,
    api_secret: str | None = None,
    bnb_discount_rate: float = DEFAULT_BNB_DISCOUNT_RATE,
    symbols: Iterable[str] | None = None,
) -> PublicFeeSnapshot:
    if symbols is None:
        symbols = fetch_exchange_symbols(timeout=timeout)
    symbol_set = {str(s).upper() for s in symbols if str(s).strip()}

    records: dict[str, FeeRecord]
    source: str
    if csv_path:
        records, source = _collect_from_csv(Path(csv_path))
    elif api_key and api_secret:
        records, source = _collect_from_private(
            api_key=api_key,
            api_secret=api_secret,
            timeout=timeout,
        )
    else:
        records, source = _collect_from_public(public_url, timeout)

    discount_rate = float(bnb_discount_rate)
    _apply_discount(records, discount_rate)
    maker_default = _most_common([r.maker_bps for r in records.values() if r.maker_bps is not None])
    taker_default = _most_common([r.taker_bps for r in records.values() if r.taker_bps is not None])
    discount_mult_estimate = _compute_discount_multiplier(records)

    if maker_discount_mult is not None:
        try:
            maker_discount_mult = float(maker_discount_mult)
        except (TypeError, ValueError):
            maker_discount_mult = None
    if taker_discount_mult is not None:
        try:
            taker_discount_mult = float(taker_discount_mult)
        except (TypeError, ValueError):
            taker_discount_mult = None

    fallback_discount_mult = discount_mult_estimate
    if fallback_discount_mult is None:
        keep_fraction = 1.0 - discount_rate
        if keep_fraction > 0.0:
            fallback_discount_mult = keep_fraction

    maker_discount = maker_discount_mult
    if maker_discount is None and fallback_discount_mult is not None:
        maker_discount = float(fallback_discount_mult)
    taker_discount = taker_discount_mult
    if taker_discount is None and fallback_discount_mult is not None:
        taker_discount = float(fallback_discount_mult)

    use_discount_flag = use_bnb_discount
    if use_discount_flag is None:
        use_discount_flag = any(
            mult is not None and mult < 0.9999 for mult in (maker_discount, taker_discount)
        )

    if use_discount_flag is False:
        if maker_discount_mult is None:
            maker_discount = 1.0
        if taker_discount_mult is None:
            taker_discount = 1.0
    elif use_discount_flag is True:
        if maker_discount is None and fallback_discount_mult is not None:
            maker_discount = float(fallback_discount_mult)
        if taker_discount is None and fallback_discount_mult is not None:
            taker_discount = float(fallback_discount_mult)
        if maker_discount is None:
            maker_discount = 1.0 - discount_rate if discount_rate < 1.0 else 1.0
        if taker_discount is None:
            taker_discount = 1.0 - discount_rate if discount_rate < 1.0 else 1.0

    if maker_discount is not None:
        maker_discount = float(maker_discount)
    if taker_discount is not None:
        taker_discount = float(taker_discount)

    vip_numeric: int | None = None
    if vip_tier_numeric is not None:
        try:
            vip_numeric = int(vip_tier_numeric)
        except (TypeError, ValueError):
            vip_numeric = None
        else:
            if vip_numeric < 0:
                vip_numeric = None

    vip_text = vip_tier.strip()
    for token in vip_text.replace("-", " ").split():
        if token.isdigit():
            try:
                vip_numeric = int(token)
                if vip_numeric < 0:
                    vip_numeric = None
                    continue
                break
            except ValueError:
                continue

    metadata_overrides: dict[str, Any] = {
        "bnb_discount_rate": discount_rate,
    }
    if vip_numeric is not None:
        metadata_overrides["vip_tier_numeric"] = vip_numeric

    account_overrides: dict[str, Any] = {}
    if vip_numeric is not None:
        account_overrides["vip_tier"] = vip_numeric
    if use_discount_flag is not None:
        account_overrides["use_bnb_discount"] = bool(use_discount_flag)
    if maker_discount is not None:
        account_overrides["maker_discount_mult"] = float(maker_discount)
    if taker_discount is not None:
        account_overrides["taker_discount_mult"] = float(taker_discount)
    if account_overrides:
        metadata_overrides["account_overrides"] = account_overrides
        metadata_overrides.setdefault("use_bnb_discount", account_overrides.get("use_bnb_discount"))
        metadata_overrides.setdefault("maker_discount_mult", account_overrides.get("maker_discount_mult"))
        metadata_overrides.setdefault("taker_discount_mult", account_overrides.get("taker_discount_mult"))

    payload = _build_payload(
        records=records,
        symbols=symbol_set,
        vip_tier=vip_tier,
        source=source,
        metadata_overrides=metadata_overrides,
    )

    return PublicFeeSnapshot(
        payload=payload,
        symbols=symbol_set,
        records=records,
        source=source,
        vip_label=vip_text,
        vip_tier=vip_numeric,
        maker_bps_default=maker_default,
        taker_bps_default=taker_default,
        maker_discount_mult=maker_discount,
        taker_discount_mult=taker_discount,
        discount_rate=discount_rate,
        use_bnb_discount=bool(use_discount_flag),
    )


__all__ = [
    "DEFAULT_BNB_DISCOUNT_RATE",
    "DEFAULT_UPDATE_THRESHOLD_DAYS",
    "DEFAULT_VIP_TIER_LABEL",
    "ensure_aware",
    "PUBLIC_FEE_URL",
    "PublicFeeSnapshot",
    "fetch_exchange_symbols",
    "format_timestamp",
    "load_public_fee_snapshot",
    "parse_timestamp",
]
