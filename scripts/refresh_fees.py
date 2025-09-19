"""CLI helper for rebuilding the Binance spot fee table.

The utility pulls the latest maker/taker basis points per symbol either from
Binance public endpoints or from the authenticated ``tradeFee`` endpoint when
API keys are available.  The resulting structure mirrors
``data/fees/fees_by_symbol.json`` and includes a metadata block describing the
data source and VIP tier.

Example usage::

    python scripts/refresh_fees.py --dry-run
    python scripts/refresh_fees.py --vip-tier "VIP 0" \
        --out data/fees/fees_by_symbol.json

Use ``--csv`` when the official CSV export is available locally, or configure
``BINANCE_API_KEY``/``BINANCE_API_SECRET`` in the environment to authorise the
private ``tradeFee`` call.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import hmac
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib import error as urlerror
from urllib import parse, request


EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
PUBLIC_FEE_URL = (
    "https://www.binance.com/bapi/asset/v1/public/asset-service/fee/get-product-fee-rate"
)
PRIVATE_TRADE_FEE_URL = "https://api.binance.com/sapi/v1/asset/tradeFee"

DEFAULT_OUTPUT = Path("data/fees/fees_by_symbol.json")
DEFAULT_VIP_TIER = "VIP 0"
SCHEMA_VERSION = 1
UPDATE_THRESHOLD_DAYS = 30
USER_AGENT = "TradingBot refresh_fees/1.0"


logger = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Binance spot trading fees and rebuild fees_by_symbol.json",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help="Destination JSON file. Defaults to data/fees/fees_by_symbol.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and report changes without writing the output file.",
    )
    parser.add_argument(
        "--vip-tier",
        default=DEFAULT_VIP_TIER,
        help="Label stored in metadata.vip_tier (default: %(default)s)",
    )
    parser.add_argument(
        "--csv",
        help="Optional CSV export with fee information to use instead of HTTP",
    )
    parser.add_argument(
        "--bnb-discount-rate",
        type=float,
        default=0.25,
        help="Fractional taker discount when paying fees with BNB (default: 0.25)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds for Binance requests (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BINANCE_API_KEY"),
        help="Binance API key for the private tradeFee endpoint. Overrides env var.",
    )
    parser.add_argument(
        "--api-secret",
        default=os.environ.get("BINANCE_API_SECRET"),
        help="Binance API secret for the private tradeFee endpoint. Overrides env var.",
    )
    parser.add_argument(
        "--public-url",
        default=PUBLIC_FEE_URL,
        help="Override the default public fee endpoint.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: %(default)s)",
    )
    return parser.parse_args(argv)


def _ensure_aware(dt: _dt.datetime) -> _dt.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_dt.timezone.utc)
    return dt.astimezone(_dt.timezone.utc)


def _parse_timestamp(raw: Any) -> _dt.datetime | None:
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
    return _ensure_aware(parsed)


def _format_timestamp(ts: _dt.datetime) -> str:
    ts = _ensure_aware(ts).replace(microsecond=0)
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
        # fall back to decimal parsing below
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


def _collect_exchange_symbols(timeout: int) -> set[str]:
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


def _check_refresh_frequency(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        logger.info("No existing fee table at %s; a full refresh will be performed", path)
        return
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to inspect existing fee table %s: %s", path, exc)
        return

    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    if not isinstance(metadata, Mapping):
        logger.info("Existing file %s lacks metadata; proceeding with refresh", path)
        return
    built_at = _parse_timestamp(metadata.get("built_at"))
    if built_at is None:
        logger.info("Existing fee table %s missing built_at field", path)
        return
    now = _ensure_aware(_dt.datetime.utcnow())
    age_days = (now - built_at).total_seconds() / 86400.0
    if age_days < UPDATE_THRESHOLD_DAYS:
        logger.info(
            "Existing fee table was generated %.1f days ago (< %d days). Use --dry-run to compare before overwriting.",
            age_days,
            UPDATE_THRESHOLD_DAYS,
        )
    else:
        logger.warning(
            "Existing fee table is %.1f days old (>= %d days). Refresh recommended.",
            age_days,
            UPDATE_THRESHOLD_DAYS,
        )


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
) -> dict[str, Any]:
    now = _ensure_aware(_dt.datetime.utcnow())
    metadata = {
        "built_at": _format_timestamp(now),
        "source": source,
        "vip_tier": vip_tier,
        "schema_version": SCHEMA_VERSION,
    }
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    tmp_path.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    out_path = Path(args.out)
    _check_refresh_frequency(out_path)

    logger.info("Fetching active spot symbols from Binance")
    symbols = _collect_exchange_symbols(args.timeout)
    logger.info("Discovered %d spot symbols", len(symbols))

    records: dict[str, FeeRecord]
    source: str
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise SystemExit(f"CSV file not found: {csv_path}")
        records, source = _collect_from_csv(csv_path)
    elif args.api_key and args.api_secret:
        logger.info("Using private tradeFee endpoint")
        records, source = _collect_from_private(
            api_key=args.api_key,
            api_secret=args.api_secret,
            timeout=args.timeout,
        )
    else:
        logger.info("Fetching public fee snapshot")
        records, source = _collect_from_public(args.public_url, args.timeout)

    logger.info("Received fee data for %d symbols", len(records))

    _apply_discount(records, float(args.bnb_discount_rate))
    payload = _build_payload(records=records, symbols=symbols, vip_tier=args.vip_tier, source=source)

    if args.dry_run:
        logger.info(
            "Dry run: would write %d symbol entries to %s", len(payload.get("fees", {})), out_path
        )
        sample = list(sorted(payload.get("fees", {}).items()))[:5]
        if sample:
            logger.info("Sample entries: %s", sample)
        return 0

    _write_json(out_path, payload)
    logger.info("Wrote fee table with %d symbols to %s", len(payload.get("fees", {})), out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())

