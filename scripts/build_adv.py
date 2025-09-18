from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from build_adv_base import (
    INTERVAL_TO_MS,
    aggregate_daily_quote_volume,
    compute_adv_quote,
    fetch_klines_for_symbols,
)
from services.rest_budget import RestBudgetSession


def _normalize_symbols(items: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for item in items:
        text = str(item).strip().upper()
        if not text:
            continue
        if text not in result:
            result.append(text)
    return result


def _load_symbols_file(path: str | None) -> list[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        payload = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[WARN] failed to read symbols file {path}: {exc}", file=sys.stderr)
        return []
    text = payload.strip()
    if not text:
        return []
    if file_path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(data, Sequence):
            return _normalize_symbols(data)
        if isinstance(data, dict):
            return _normalize_symbols(data.keys())
        return []
    lines = [line.strip() for line in text.replace(",", "\n").splitlines()]
    return _normalize_symbols(line for line in lines if line)


def _default_symbols() -> list[str]:
    default_path = Path("data/universe/symbols.json")
    if not default_path.exists():
        return []
    try:
        data = json.loads(default_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, Sequence):
        return _normalize_symbols(data)
    if isinstance(data, dict):
        return _normalize_symbols(data.keys())
    return []


def _resolve_symbols(symbols_arg: str, symbols_file: str) -> list[str]:
    direct: list[str] = []
    if symbols_arg:
        parts: list[str] = []
        if os.path.exists(symbols_arg):
            parts = _load_symbols_file(symbols_arg)
        else:
            for chunk in symbols_arg.replace("\n", ",").split(","):
                chunk = chunk.strip()
                if chunk:
                    parts.append(chunk)
        direct = _normalize_symbols(parts)
    if direct:
        return direct
    file_symbols = _load_symbols_file(symbols_file)
    if file_symbols:
        return file_symbols
    return _default_symbols()


def _isoformat_ms(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    try:
        dt = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return dt.isoformat().replace("+00:00", "Z")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build average daily quote volume dataset from Binance klines.",
    )
    parser.add_argument("--market", choices=["spot", "futures"], default="futures")
    parser.add_argument("--interval", default="1d", help="Kline interval (e.g. 1h,4h,1d)")
    parser.add_argument("--window-days", type=int, default=30, help="Rolling window in days")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols or path to file")
    parser.add_argument(
        "--symbols-file",
        default="",
        help="Optional path to JSON/TXT with symbols (fallback to data/universe/symbols.json)",
    )
    parser.add_argument("--out", required=True, help="Destination JSON path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    interval = str(args.interval).lower()
    if interval not in INTERVAL_TO_MS:
        raise SystemExit(f"Unsupported interval: {args.interval!r}")

    window_days = max(1, int(args.window_days))
    symbols = _resolve_symbols(args.symbols, args.symbols_file)
    if not symbols:
        raise SystemExit("No symbols resolved; provide --symbols or --symbols-file")

    unique_symbols = _normalize_symbols(symbols)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    end_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=window_days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    with RestBudgetSession({}) as session:
        datasets = fetch_klines_for_symbols(
            session,
            unique_symbols,
            market=args.market,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    generated_at = datetime.now(timezone.utc)
    generated_ms = int(generated_at.timestamp() * 1000)
    results: dict[str, Any] = {}

    for symbol in unique_symbols:
        df = datasets.get(symbol, None)
        if df is None:
            results[symbol] = {
                "adv_quote": None,
                "days": 0,
                "days_total": 0,
                "last_day": None,
            }
            continue
        daily = aggregate_daily_quote_volume(df)
        adv_value, used_days, total_days = compute_adv_quote(
            daily,
            window_days=window_days,
            min_days=1,
        )
        last_day: str | None = None
        if not daily.empty:
            last_idx = daily.dropna().index.max()
            if last_idx is not None:
                try:
                    last_dt = last_idx.to_pydatetime()
                except AttributeError:
                    last_dt = None
                if last_dt is None and isinstance(last_idx, datetime):
                    last_dt = last_idx
                if last_dt is not None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                    last_day = last_dt.isoformat().replace("+00:00", "Z")
        results[symbol] = {
            "adv_quote": float(adv_value) if adv_value is not None else None,
            "days": int(used_days),
            "days_total": int(total_days),
            "last_day": last_day,
        }

    payload = {
        "meta": {
            "version": 1,
            "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
            "generated_at_ms": generated_ms,
            "window_days": window_days,
            "interval": interval,
            "market": args.market,
            "source": "binance",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_at": _isoformat_ms(start_ms),
            "end_at": _isoformat_ms(end_ms),
            "symbols": unique_symbols,
        },
        "data": results,
    }

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(out_path.parent), delete=False
        ) as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
            tmp_path = Path(fh.name)
    except Exception:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    else:
        if tmp_path is not None:
            os.replace(tmp_path, out_path)

    print(
        json.dumps(
            {
                "out": str(out_path),
                "symbols": unique_symbols,
                "window_days": window_days,
                "generated_at": payload["meta"]["generated_at"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
