"""CLI for building ADV OHLCV snapshots using Binance public data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from build_adv_base import BuildAdvConfig, build_adv
from services.rest_budget import RestBudgetSession
from utils_time import parse_time_to_ms


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
    try:
        payload = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
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
    try:
        with default_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []
    if isinstance(data, Sequence):
        return _normalize_symbols(data)
    if isinstance(data, dict):
        return _normalize_symbols(data.keys())
    return []


def _resolve_symbols(symbols_arg: str, symbols_file: str) -> list[str]:
    direct = _normalize_symbols(symbols_arg.split(",") if symbols_arg else [])
    if direct:
        return direct
    file_symbols = _load_symbols_file(symbols_file)
    if file_symbols:
        return file_symbols
    return _default_symbols()


def _load_rest_config(path: str) -> dict[str, Any]:
    path = path.strip()
    if not path:
        return {}
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        print(f"[WARN] rest budget config not found: {path}", file=sys.stderr)
        return {}
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] failed to load rest budget config {path}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        return {}
    return dict(data)


def _parse_time(value: str, name: str) -> int:
    try:
        return parse_time_to_ms(value)
    except Exception as exc:  # pragma: no cover - validation
        raise SystemExit(f"Invalid {name}: {value!r} ({exc})")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Binance OHLCV history and build ADV dataset parquet.",
    )
    parser.add_argument("--market", choices=["spot", "futures"], default="futures")
    parser.add_argument("--interval", default="1h", help="Kline interval (e.g. 1h,4h,1d)")
    parser.add_argument("--start", required=True, help="Start of history (ISO8601 or unix ms)")
    parser.add_argument("--end", required=True, help="End of history (ISO8601 or unix ms)")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol list")
    parser.add_argument(
        "--symbols-file",
        default="",
        help="Optional path to JSON/TXT with symbols; fallback to data/universe/symbols.json",
    )
    parser.add_argument("--out", default="data/adv/klines.parquet", help="Destination dataset path")
    parser.add_argument(
        "--cache-dir",
        default="data/adv/cache",
        help="Directory for per-symbol parquet cache",
    )
    parser.add_argument("--limit", type=int, default=1500, help="Maximum bars per request")
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=30,
        help="Chunk size in days for planning fetch windows",
    )
    parser.add_argument(
        "--rest-budget-config",
        default="configs/rest_budget.yaml",
        help="Path to RestBudgetSession YAML configuration",
    )
    parser.add_argument(
        "--cache-mode",
        default=None,
        help="RestBudgetSession cache mode override (off/read/read_write)",
    )
    parser.add_argument(
        "--cache-ttl",
        type=float,
        default=None,
        help="RestBudgetSession cache TTL in days",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Override checkpoint path (defaults to <cache-dir>/checkpoint.json)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume from checkpoint if present",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume_from_checkpoint",
        action="store_false",
        help="Ignore checkpoint even if present",
    )
    parser.set_defaults(resume_from_checkpoint=False)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    start_ms = _parse_time(args.start, "start")
    end_ms = _parse_time(args.end, "end")
    if end_ms <= start_ms:
        raise SystemExit("end must be greater than start")

    symbols = _resolve_symbols(args.symbols, args.symbols_file)
    if not symbols:
        raise SystemExit("No symbols resolved; provide --symbols or --symbols-file")

    out_path = Path(args.out)
    cache_dir = Path(args.cache_dir)

    limit = max(1, int(args.limit))
    chunk_days = max(1, int(args.chunk_days))

    config = BuildAdvConfig(
        market=args.market,
        interval=args.interval,
        start_ms=start_ms,
        end_ms=end_ms,
        out_path=out_path,
        cache_dir=cache_dir,
        limit=limit,
        chunk_days=chunk_days,
        resume_from_checkpoint=bool(args.resume_from_checkpoint),
    )

    rest_cfg = _load_rest_config(str(args.rest_budget_config))

    checkpoint_path = args.checkpoint_path.strip()
    if not checkpoint_path:
        checkpoint_path = str(cache_dir / "checkpoint.json")

    cache_override = str(cache_dir)
    cache_mode = args.cache_mode if args.cache_mode else None

    with RestBudgetSession(
        rest_cfg,
        cache_dir=cache_override,
        ttl_days=args.cache_ttl,
        mode=cache_mode,
        checkpoint_path=checkpoint_path,
        checkpoint_enabled=bool(checkpoint_path),
        resume_from_checkpoint=bool(args.resume_from_checkpoint),
    ) as session:
        result = build_adv(session, symbols, config)
        payload = {
            "result": result.to_dict(),
            "rest": session.stats(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
