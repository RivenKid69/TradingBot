"""CLI helper for rebuilding the Binance spot fee table."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from binance_fee_refresh import (
    DEFAULT_BNB_DISCOUNT_RATE,
    DEFAULT_UPDATE_THRESHOLD_DAYS,
    DEFAULT_VIP_TIER_LABEL,
    PUBLIC_FEE_URL,
    ensure_aware,
    fetch_exchange_symbols,
    load_public_fee_snapshot,
    parse_timestamp,
)


DEFAULT_OUTPUT = Path("data/fees/fees_by_symbol.json")


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
        default=DEFAULT_VIP_TIER_LABEL,
        help="Label stored in metadata.vip_tier (default: %(default)s)",
    )
    parser.add_argument(
        "--csv",
        help="Optional CSV export with fee information to use instead of HTTP",
    )
    parser.add_argument(
        "--bnb-discount-rate",
        type=float,
        default=DEFAULT_BNB_DISCOUNT_RATE,
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
    built_at = parse_timestamp(metadata.get("built_at"))
    if built_at is None:
        logger.info("Existing fee table %s missing built_at field", path)
        return
    now = ensure_aware(_dt.datetime.utcnow())
    age_days = (now - built_at).total_seconds() / 86400.0
    if age_days < DEFAULT_UPDATE_THRESHOLD_DAYS:
        logger.info(
            "Existing fee table was generated %.1f days ago (< %d days). Use --dry-run to compare before overwriting.",
            age_days,
            DEFAULT_UPDATE_THRESHOLD_DAYS,
        )
    else:
        logger.warning(
            "Existing fee table is %.1f days old (>= %d days). Refresh recommended.",
            age_days,
            DEFAULT_UPDATE_THRESHOLD_DAYS,
        )


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
    symbols = fetch_exchange_symbols(timeout=args.timeout)
    logger.info("Discovered %d spot symbols", len(symbols))

    csv_path = Path(args.csv) if args.csv else None
    if csv_path is not None and not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    api_key = args.api_key if args.api_key and args.api_secret else None
    api_secret = args.api_secret if args.api_key and args.api_secret else None
    if api_key and api_secret:
        logger.info("Using private tradeFee endpoint")
    else:
        logger.info("Fetching public fee snapshot")

    snapshot = load_public_fee_snapshot(
        vip_tier=args.vip_tier,
        timeout=args.timeout,
        public_url=args.public_url,
        csv_path=csv_path,
        api_key=api_key,
        api_secret=api_secret,
        bnb_discount_rate=float(args.bnb_discount_rate),
        symbols=symbols,
    )

    payload = snapshot.payload
    logger.info("Received fee data for %d symbols", len(snapshot.records))
    if snapshot.maker_bps_default is not None or snapshot.taker_bps_default is not None:
        logger.info(
            "Default maker/taker bps: %s / %s (discount multiplier %.4f)",
            snapshot.maker_bps_default,
            snapshot.taker_bps_default,
            snapshot.taker_discount_mult if snapshot.taker_discount_mult is not None else 1.0,
        )

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
