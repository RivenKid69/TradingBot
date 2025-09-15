#!/usr/bin/env python3
"""Fetch Binance exchange filters and store them as JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from typing import Iterable, List

from binance_public import BinancePublicClient


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Binance spot exchange filters and save them as JSON",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Destination JSON file path",
    )
    parser.add_argument(
        "--universe",
        action="store_true",
        help="Load symbols from services.universe.get_symbols()",
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Symbols to include (defaults to all when omitted)",
    )
    return parser.parse_args(argv)


def _normalize_symbols(raw: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for symbol in raw:
        if not symbol:
            continue
        sym = symbol.strip().upper()
        if sym:
            cleaned.append(sym)
    # Preserve order while removing duplicates
    return list(dict.fromkeys(cleaned))


def _load_symbols(args: argparse.Namespace) -> List[str]:
    symbols: List[str] = []
    if args.universe:
        from services.universe import get_symbols as get_universe_symbols

        symbols.extend(get_universe_symbols())
    if args.symbols:
        symbols.extend(args.symbols)
    return _normalize_symbols(symbols)


def _ensure_directory(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _write_json_atomic(path: str, payload: dict) -> None:
    _ensure_directory(path)
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".binance_filters_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(payload, tmp_file, ensure_ascii=False, indent=2, sort_keys=True)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _build_metadata(filters: dict) -> dict:
    return {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": "/api/v3/exchangeInfo",
        "symbols_count": len(filters),
    }


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        symbols = _load_symbols(args)
        client = BinancePublicClient()
        filters = client.get_exchange_filters(market="spot", symbols=symbols)
        metadata = _build_metadata(filters)
        payload = {"metadata": metadata, "filters": filters}
        _write_json_atomic(args.out, payload)
        print(
            f"Fetched {metadata['symbols_count']} symbol filters "
            f"from {metadata['source']} into {args.out}"
        )
        return 0
    except Exception as exc:  # pragma: no cover - CLI error handling
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
