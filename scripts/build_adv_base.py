from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from build_adv_base import (
    INTERVAL_TO_MS,
    aggregate_daily_base_volume,
    compute_adv_base,
    fetch_klines_for_symbols,
)
from build_adv import _normalize_rest_budget_config, _load_offline_config as _load_offline_defaults
from services.rest_budget import RestBudgetSession


STATS_PATH = Path("logs/offline/build_adv_base_stats.json")


def _merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {key: value for key, value in base.items()}
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


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
        if isinstance(data, Mapping):
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
    if isinstance(data, Mapping):
        return _normalize_symbols(data.keys())
    return []


def _resolve_symbols(
    symbols_arg: str,
    symbols_file: str,
    universe_symbols: Sequence[str],
) -> list[str]:
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
    if universe_symbols:
        return _normalize_symbols(universe_symbols)
    return _default_symbols()


def _load_universe(path: str | None) -> tuple[dict[str, Any], list[str]]:
    if not path:
        return {}, []
    file_path = Path(path)
    if not file_path.exists():
        logging.debug("universe file %s does not exist", path)
        return {}, []
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logging.warning("failed to parse universe file %s: %s", path, exc)
        return {}, []
    except OSError as exc:
        logging.warning("failed to read universe file %s: %s", path, exc)
        return {}, []

    if isinstance(payload, (list, tuple, set)):
        return {}, _normalize_symbols(payload)

    if not isinstance(payload, Mapping):
        return {}, []

    meta_raw = payload.get("meta")
    meta: dict[str, Any] = dict(meta_raw) if isinstance(meta_raw, Mapping) else {}
    for key in ("window_days", "refresh_days", "market"):
        if key in payload and key not in meta:
            meta[key] = payload[key]

    symbols: list[str] = []
    if "symbols" in payload:
        symbols = _normalize_symbols(payload.get("symbols", []))
    elif "symbols" in meta:
        symbols = _normalize_symbols(meta.get("symbols", []))
    else:
        data_section = payload.get("data")
        if isinstance(data_section, Mapping):
            symbols = _normalize_symbols(data_section.keys())

    return meta, symbols


def _pick_meta_number(meta: Mapping[str, Any], key: str) -> Any:
    if key in meta:
        return meta[key]
    nested = meta.get("adv")
    if isinstance(nested, Mapping) and key in nested:
        return nested[key]
    defaults = meta.get("defaults")
    if isinstance(defaults, Mapping) and key in defaults:
        return defaults[key]
    return None


def _load_rest_budget_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logging.warning("rest budget config not found: %s", path)
        return {}
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.warning("failed to load rest budget config %s: %s", path, exc)
        return {}
    if not isinstance(data, Mapping):
        logging.warning("rest budget config %s must be a mapping", path)
        return {}
    return dict(data)


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
        description="Build average daily base volume dataset from Binance klines.",
    )
    parser.add_argument("--market", choices=["spot", "futures"], default=None)
    parser.add_argument("--interval", default="1d", help="Kline interval (e.g. 1h,4h,1d)")
    parser.add_argument("--window-days", type=int, default=None, help="Rolling window in days")
    parser.add_argument("--refresh-days", type=int, default=None, help="Refresh cadence in days")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols or path to file")
    parser.add_argument(
        "--symbols-file",
        default="",
        help="Optional path to JSON/TXT with symbols (fallback to data/universe/symbols.json)",
    )
    parser.add_argument(
        "--universe",
        default="data/universe/symbols.json",
        help="Path to universe JSON providing default symbols and metadata",
    )
    parser.add_argument(
        "--rest-budget-config",
        default="configs/rest_budget.yaml",
        help="Path to RestBudgetSession YAML configuration",
    )
    parser.add_argument(
        "--config",
        default="configs/offline.yaml",
        help="Path to offline YAML config providing default rest budget settings",
    )
    parser.add_argument(
        "--clip-lower",
        type=float,
        default=5.0,
        help="Lower percentile for clipping daily volumes (set >= upper to disable)",
    )
    parser.add_argument(
        "--clip-upper",
        type=float,
        default=95.0,
        help="Upper percentile for clipping daily volumes (set <= lower to disable)",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=None,
        help="Minimum valid days required within the window",
    )
    parser.add_argument(
        "--min-total-days",
        type=int,
        default=None,
        help="Minimum total valid days before computing ADV",
    )
    parser.add_argument("--out", required=True, help="Destination JSON path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    interval = str(args.interval).lower()
    if interval not in INTERVAL_TO_MS:
        raise SystemExit(f"Unsupported interval: {args.interval!r}")

    universe_meta, universe_symbols = _load_universe(args.universe)

    raw_market = args.market or _pick_meta_number(universe_meta, "market") or "futures"
    market = str(raw_market).strip().lower()
    if market not in {"spot", "futures"}:
        logging.warning("unsupported market %s, falling back to futures", raw_market)
        market = "futures"

    window_default = _pick_meta_number(universe_meta, "window_days")
    refresh_default = _pick_meta_number(universe_meta, "refresh_days")

    window_days = int(args.window_days or window_default or 30)
    refresh_days: int | None = None
    if args.refresh_days is not None:
        refresh_days = max(0, int(args.refresh_days))
    elif refresh_default is not None:
        try:
            refresh_days = max(0, int(refresh_default))
        except (TypeError, ValueError):
            refresh_days = None

    symbols = _resolve_symbols(args.symbols, args.symbols_file, universe_symbols)
    if not symbols:
        raise SystemExit("No symbols resolved; provide --symbols or --symbols-file")

    unique_symbols = _normalize_symbols(symbols)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    end_dt = now
    start_dt = end_dt - timedelta(days=window_days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    offline_payload: Mapping[str, Any] | dict[str, Any] = {}
    if args.config:
        try:
            offline_payload = _load_offline_defaults(Path(args.config))
        except Exception:
            offline_payload = {}
    offline_rest_cfg = _normalize_rest_budget_config(
        offline_payload.get("rest_budget") if isinstance(offline_payload, Mapping) else None
    )
    rest_cfg_file = _load_rest_budget_config(args.rest_budget_config)
    rest_cfg_loaded = _normalize_rest_budget_config(rest_cfg_file)
    rest_cfg: dict[str, Any] = offline_rest_cfg
    if rest_cfg_loaded:
        rest_cfg = _merge_dicts(rest_cfg, rest_cfg_loaded) if rest_cfg else rest_cfg_loaded

    stats_path = STATS_PATH
    with RestBudgetSession(rest_cfg) as session:
        try:
            session.write_stats(stats_path)
        except Exception:
            logging.debug("Failed to write initial stats snapshot", exc_info=True)
        datasets = fetch_klines_for_symbols(
            session,
            unique_symbols,
            market=market,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            stats_path=stats_path,
        )

    generated_at = datetime.now(timezone.utc)
    generated_ms = int(generated_at.timestamp() * 1000)
    results: dict[str, Any] = {}

    clip_lower = args.clip_lower
    clip_upper = args.clip_upper
    clip_tuple: tuple[float, float] | None
    try:
        lower_val = float(clip_lower)
        upper_val = float(clip_upper)
    except (TypeError, ValueError):
        clip_tuple = None
    else:
        if upper_val <= lower_val:
            clip_tuple = None
        else:
            clip_tuple = (lower_val, upper_val)

    min_days = args.min_days
    if min_days is None:
        min_days = max(1, window_days // 2)
    else:
        min_days = max(1, int(min_days))

    min_total_days = args.min_total_days
    if min_total_days is None:
        min_total_days = max(min_days, window_days // 2)
    else:
        min_total_days = max(0, int(min_total_days))

    for symbol in unique_symbols:
        df = datasets.get(symbol, None)
        if df is None:
            results[symbol] = {
                "adv_base": None,
                "days": 0,
                "days_total": 0,
                "last_day": None,
            }
            logging.warning("no klines returned for %s", symbol)
            continue
        daily = aggregate_daily_base_volume(df)
        adv_value, used_days, total_days = compute_adv_base(
            daily,
            window_days=window_days,
            min_days=min_days,
            min_total_days=min_total_days,
            clip_percentiles=clip_tuple,
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
        if adv_value is None or used_days < min_days:
            logging.warning(
                "insufficient ADV data for %s (used=%d total=%d)",
                symbol,
                used_days,
                total_days,
            )

        results[symbol] = {
            "adv_base": float(adv_value) if adv_value is not None else None,
            "days": int(used_days),
            "days_total": int(total_days),
            "last_day": last_day,
        }

    payload = {
        "meta": {
            "built_at": generated_at.isoformat().replace("+00:00", "Z"),
            "built_at_ms": generated_ms,
            "window_days": int(window_days),
            "source": {
                "exchange": "binance",
                "market": market,
                "interval": interval,
                "refresh_days": refresh_days,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_at": _isoformat_ms(start_ms),
                "end_at": _isoformat_ms(end_ms),
                "symbols": unique_symbols,
                "clip_percentiles": clip_tuple,
                "min_days": min_days,
                "min_total_days": min_total_days,
            },
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
                "built_at": payload["meta"]["built_at"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

