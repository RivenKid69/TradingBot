"""Extract liquidity seasonality multipliers.

The hour-of-week index uses ``0 = Monday 00:00 UTC``.

The script supports two modes of operation:

1. Load an existing OHLCV snapshot from ``--data`` and compute multipliers.
2. Fetch missing Binance klines for the requested symbols/intervals, persist
   them under ``--cache-dir``, merge the results into ``--data`` and then run
   the multiplier calculation.

When fetching data the helper relies on :class:`RestBudgetSession` for request
budgeting, optional caching and checkpointing.  Checkpoints are updated after
each completed chunk so interrupted runs can resume with
``--resume-from-checkpoint``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from binance_public import BinancePublicClient
from services.rest_budget import RestBudgetSession
from utils.time import hour_of_week
from utils_time import parse_time_to_ms


INTERVAL_TO_MS: Mapping[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


KLINE_COLUMNS = [
    "ts_ms",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
]


@dataclass(frozen=True)
class FetchTask:
    """Single kline chunk request."""

    symbol: str
    interval: str
    start_ms: int
    bars: int

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "start_ms": int(self.start_ms),
            "bars": int(self.bars),
        }


def load_ohlcv(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def compute_multipliers(df: pd.DataFrame) -> np.ndarray:
    if "ts_ms" not in df.columns:
        raise ValueError("ts_ms column required")
    vol_col = next(
        (c for c in ["quote_asset_volume", "quote_volume", "volume"] if c in df.columns),
        None,
    )
    if vol_col is None:
        raise ValueError("volume column not found")
    # ``hour_of_week`` uses Monday 00:00 UTC as index 0
    ts_ms = df["ts_ms"].to_numpy(dtype=np.int64)
    df = df.assign(hour_of_week=hour_of_week(ts_ms))
    grouped = df.groupby("hour_of_week")[vol_col].mean()
    overall = df[vol_col].mean()
    mult = grouped / overall if overall else grouped * 0.0 + 1.0
    mult = mult.reindex(range(168), fill_value=1.0)
    return mult.to_numpy(dtype=float)


def write_checksum(path: Path) -> Path:
    """Compute sha256 checksum for *path* and write `<path>.sha256`."""

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    checksum_path.write_text(digest)
    return checksum_path


def _normalize_symbol_list(raw: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for sym in raw:
        if not sym:
            continue
        token = sym.strip().upper()
        if not token:
            continue
        if token not in cleaned:
            cleaned.append(token)
    return cleaned


def _load_default_symbols() -> List[str]:
    default_path = Path("data/universe/symbols.json")
    if not default_path.exists():
        return []
    try:
        with default_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return _normalize_symbol_list(str(x) for x in payload)
    except Exception:
        return []
    return []


def _parse_symbols_arg(value: str | None) -> List[str]:
    if not value:
        return []
    return _normalize_symbol_list(value.split(","))


def _resolve_symbols(arg_value: str | None) -> List[str]:
    explicit = _parse_symbols_arg(arg_value)
    if explicit:
        return explicit
    return _load_default_symbols()


def _resolve_intervals(value: str | Sequence[str] | None) -> List[str]:
    if value is None:
        return ["1h"]
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
    else:
        tokens = [str(token).strip() for token in value if str(token).strip()]
    cleaned: List[str] = []
    for token in tokens or ["1h"]:
        norm = token.lower()
        if norm not in INTERVAL_TO_MS:
            raise ValueError(f"Unsupported interval: {token}")
        if norm not in cleaned:
            cleaned.append(norm)
    return cleaned


def _load_rest_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[WARN] rest budget config not found: {path}")
        return {}
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] failed to load rest budget config {path}: {exc}")
        return {}
    if not isinstance(payload, MutableMapping):
        return {}
    return dict(payload)


def _cache_path(cache_dir: Path, symbol: str, interval: str) -> Path:
    safe_sym = symbol.upper()
    return cache_dir / f"{safe_sym}_{interval}.parquet"


def _load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=KLINE_COLUMNS)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _align_range(start_ms: int, end_ms: int, step_ms: int) -> Tuple[int, int]:
    aligned_start = (start_ms // step_ms) * step_ms
    aligned_end = ((end_ms + step_ms - 1) // step_ms) * step_ms
    return aligned_start, aligned_end


def _iter_missing_ranges(
    existing: pd.DataFrame,
    *,
    start_ms: int,
    end_ms: int,
    step_ms: int,
) -> Iterator[Tuple[int, int]]:
    if start_ms >= end_ms:
        return
    mask = (existing["ts_ms"].astype("int64") >= start_ms) & (
        existing["ts_ms"].astype("int64") < end_ms
    )
    present = set(existing.loc[mask, "ts_ms"].astype("int64"))
    expected = np.arange(start_ms, end_ms, step_ms, dtype=np.int64)
    if not len(expected):
        return
    current_start: int | None = None
    previous: int | None = None
    for ts in expected:
        if int(ts) in present:
            if current_start is not None and previous is not None:
                yield current_start, previous
                current_start = None
                previous = None
            continue
        if current_start is None:
            current_start = int(ts)
            previous = int(ts)
        else:
            assert previous is not None
            if int(ts) == previous + step_ms:
                previous = int(ts)
            else:
                yield current_start, previous
                current_start = int(ts)
                previous = int(ts)
    if current_start is not None and previous is not None:
        yield current_start, previous


def _split_ranges_to_tasks(
    ranges: Iterable[Tuple[int, int]],
    *,
    step_ms: int,
    limit: int,
    symbol: str,
    interval: str,
) -> List[FetchTask]:
    tasks: List[FetchTask] = []
    max_bars = max(1, int(limit))
    for start, end in ranges:
        bars_total = int((end - start) // step_ms) + 1
        remaining = bars_total
        cursor = start
        while remaining > 0:
            chunk = min(remaining, max_bars)
            tasks.append(FetchTask(symbol, interval, cursor, chunk))
            cursor += chunk * step_ms
            remaining -= chunk
    return tasks


def _raw_to_df(raw: Sequence[Sequence[Any]], symbol: str, interval: str) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    df = pd.DataFrame(
        raw,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    out = pd.DataFrame(
        {
            "ts_ms": df["open_time"].astype("int64"),
            "symbol": symbol.upper(),
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
            "quote_asset_volume": pd.to_numeric(
                df["quote_asset_volume"], errors="coerce"
            ),
            "number_of_trades": pd.to_numeric(
                df["number_of_trades"], errors="coerce"
            ).astype("Int64"),
            "taker_buy_base": pd.to_numeric(
                df["taker_buy_base"], errors="coerce"
            ),
            "taker_buy_quote": pd.to_numeric(
                df["taker_buy_quote"], errors="coerce"
            ),
        }
    )
    out["interval"] = interval
    return out[KLINE_COLUMNS + ["interval"]]


def _merge_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return incoming.copy()
    if incoming.empty:
        return existing
    merged = (
        pd.concat([existing, incoming], ignore_index=True)
        .drop_duplicates(subset=["ts_ms"], keep="last")
        .sort_values("ts_ms")
        .reset_index(drop=True)
    )
    return merged


def _prepare_tasks(
    *,
    symbols: Sequence[str],
    intervals: Sequence[str],
    start_ms: int,
    end_ms: int,
    limit: int,
    cache_dir: Path,
) -> Tuple[List[FetchTask], Dict[Tuple[str, str], pd.DataFrame], Dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks: List[FetchTask] = []
    datasets: Dict[Tuple[str, str], pd.DataFrame] = {}
    summary: Dict[str, Any] = {
        "total_missing": 0,
        "per_pair": {},
    }
    for symbol in symbols:
        for interval in intervals:
            step_ms = INTERVAL_TO_MS[interval]
            aligned_start, aligned_end = _align_range(start_ms, end_ms, step_ms)
            cache_path = _cache_path(cache_dir, symbol, interval)
            df = _load_cache(cache_path)
            datasets[(symbol, interval)] = df
            ranges = list(
                _iter_missing_ranges(
                    df, start_ms=aligned_start, end_ms=aligned_end, step_ms=step_ms
                )
            )
            pair_tasks = _split_ranges_to_tasks(
                ranges,
                step_ms=step_ms,
                limit=limit,
                symbol=symbol,
                interval=interval,
            )
            tasks.extend(pair_tasks)
            missing_bars = sum(task.bars for task in pair_tasks)
            existing_bars = int(
                ((df["ts_ms"].astype("int64") >= aligned_start)
                & (df["ts_ms"].astype("int64") < aligned_end)).sum()
            )
            summary["per_pair"][f"{symbol}_{interval}"] = {
                "missing_bars": missing_bars,
                "existing_bars": existing_bars,
            }
            summary["total_missing"] += missing_bars
    tasks.sort(key=lambda t: (t.symbol, t.interval, t.start_ms))
    summary["total_tasks"] = len(tasks)
    summary["symbols"] = list(symbols)
    summary["intervals"] = list(intervals)
    summary["start_ms"] = int(start_ms)
    summary["end_ms"] = int(end_ms)
    return tasks, datasets, summary


def _determine_start_index(
    checkpoint: Mapping[str, Any] | None,
    *,
    signature: Mapping[str, Any],
    total_tasks: int,
) -> int:
    if not checkpoint:
        return 0
    if checkpoint.get("completed"):
        return total_tasks
    saved_signature = checkpoint.get("signature")
    if saved_signature != signature:
        return 0
    try:
        index = int(checkpoint.get("task_index", 0))
    except (TypeError, ValueError):
        return 0
    if index < 0:
        return 0
    if index > total_tasks:
        return total_tasks
    return index


def _save_checkpoint(
    session: RestBudgetSession,
    *,
    position: int,
    total: int,
    signature: Mapping[str, Any],
    current: FetchTask | None = None,
    completed: bool = False,
) -> None:
    payload: dict[str, Any] = {
        "task_index": int(position),
        "tasks_total": int(total),
        "signature": dict(signature),
    }
    if current is not None:
        payload["current"] = current.to_checkpoint()
    if completed:
        payload["completed"] = True
    session.save_checkpoint(payload)


def _write_dataset(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _combine_datasets(
    datasets: Mapping[Tuple[str, str], pd.DataFrame],
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for (symbol, interval), df in datasets.items():
        if df.empty:
            continue
        subset = df.copy()
        if start_ms is not None:
            subset = subset[subset["ts_ms"].astype("int64") >= int(start_ms)]
        if end_ms is not None:
            subset = subset[subset["ts_ms"].astype("int64") < int(end_ms)]
        if subset.empty:
            continue
        subset = subset.assign(symbol=symbol.upper(), interval=interval)
        frames.append(subset)
    if not frames:
        return pd.DataFrame(columns=KLINE_COLUMNS + ["interval"])
    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.drop_duplicates(subset=["symbol", "interval", "ts_ms"], keep="last")
        .sort_values(["symbol", "interval", "ts_ms"])
        .reset_index(drop=True)
    )
    return combined


def _plan_description(summary: Mapping[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract liquidity seasonality multipliers")
    parser.add_argument(
        "--data",
        default="data/seasonality_source/latest.parquet",
        help="Path to OHLCV data (csv or parquet). Updated when fetching is enabled.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/seasonality_source/cache",
        help="Directory with per-symbol cached OHLCV parquet files",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols (defaults to data/universe/symbols.json)",
    )
    parser.add_argument(
        "--intervals",
        default="1h",
        help="Comma-separated kline intervals (e.g. 1h,4h)",
    )
    parser.add_argument(
        "--market",
        choices=["spot", "futures"],
        default="futures",
        help="Binance market to query",
    )
    parser.add_argument("--start", help="Start of requested history (ISO8601 or unix ms)")
    parser.add_argument("--end", help="End of requested history (ISO8601 or unix ms)")
    parser.add_argument(
        "--limit",
        type=int,
        default=1500,
        help="Maximum bars per request (Binance limit is 1500)",
    )
    parser.add_argument(
        "--rest-budget-config",
        default="configs/rest_budget.yaml",
        help="Path to RestBudgetSession YAML configuration",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Override checkpoint path (defaults to <cache-dir>/checkpoint.json)",
    )
    parser.add_argument(
        "--cache-mode",
        default="off",
        help="RestBudgetSession cache mode: off/read/read_write",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume fetching using checkpoint.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan missing requests without performing HTTP calls",
    )
    parser.add_argument(
        "--out",
        default="configs/liquidity_seasonality.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    cache_dir = Path(args.cache_dir)

    fetch_enabled = bool(args.start and args.end)
    if not fetch_enabled and args.dry_run:
        print("[WARN] --dry-run ignored because --start/--end are not provided")
        args.dry_run = False

    symbols = _resolve_symbols(args.symbols)
    intervals = _resolve_intervals(args.intervals)

    rest_cfg = _load_rest_config(args.rest_budget_config)

    if fetch_enabled:
        start_ms = parse_time_to_ms(args.start)
        end_ms = parse_time_to_ms(args.end)
        if end_ms <= start_ms:
            raise SystemExit("end must be greater than start")
        if not symbols:
            raise SystemExit("No symbols provided and default universe is empty")

        tasks, datasets, summary = _prepare_tasks(
            symbols=symbols,
            intervals=intervals,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=max(1, args.limit),
            cache_dir=cache_dir,
        )

        plan_signature = {
            "symbols": list(symbols),
            "intervals": list(intervals),
            "start_ms": int(start_ms),
            "end_ms": int(end_ms),
            "limit": int(max(1, args.limit)),
            "market": args.market,
        }

        if args.dry_run:
            print(_plan_description(summary))
            return

        checkpoint_path = args.checkpoint_path.strip()
        if not checkpoint_path:
            checkpoint_path = str(cache_dir / "checkpoint.json")

        with RestBudgetSession(
            rest_cfg,
            mode=args.cache_mode,
            checkpoint_path=checkpoint_path,
            checkpoint_enabled=bool(checkpoint_path),
            resume_from_checkpoint=args.resume_from_checkpoint,
        ) as session:
            client = BinancePublicClient(session=session)

            def _fetch_single(t: FetchTask) -> pd.DataFrame:
                step_ms = INTERVAL_TO_MS[t.interval]
                end_exclusive = t.start_ms + t.bars * step_ms
                end_param = end_exclusive - 1
                raw = client.get_klines(
                    market=args.market,
                    symbol=t.symbol,
                    interval=t.interval,
                    start_ms=t.start_ms,
                    end_ms=end_param,
                    limit=min(args.limit, t.bars),
                )
                return _raw_to_df(raw, t.symbol, t.interval)

            try:
                checkpoint_payload = (
                    session.load_checkpoint() if args.resume_from_checkpoint else None
                )
                start_index = _determine_start_index(
                    checkpoint_payload,
                    signature=plan_signature,
                    total_tasks=len(tasks),
                )
                batch_pref = int(getattr(session, "batch_size", 0) or 0)
                worker_pref = int(getattr(session, "max_workers", 0) or 0)
                batch_size = max(1, batch_pref or worker_pref or 1)

                _save_checkpoint(
                    session,
                    position=start_index,
                    total=len(tasks),
                    signature=plan_signature,
                )

                idx = start_index
                while idx < len(tasks):
                    batch = tasks[idx : idx + batch_size]
                    futures: List[Tuple[int, FetchTask, Future[pd.DataFrame]]] = []
                    for offset, task in enumerate(batch):
                        absolute = idx + offset
                        _save_checkpoint(
                            session,
                            position=absolute,
                            total=len(tasks),
                            signature=plan_signature,
                            current=task,
                        )
                        future = session.submit(_fetch_single, task)
                        futures.append((absolute, task, future))

                    for absolute, task, future in futures:
                        try:
                            fetched = future.result()
                        except Exception as exc:  # pragma: no cover - network dependent
                            raise RuntimeError(
                                f"Failed to fetch {task.symbol} {task.interval} starting at {task.start_ms}: {exc}"
                            ) from exc
                        key = (task.symbol, task.interval)
                        datasets[key] = _merge_frames(datasets[key], fetched)
                        cache_path = _cache_path(cache_dir, task.symbol, task.interval)
                        _write_dataset(cache_path, datasets[key])
                        _save_checkpoint(
                            session,
                            position=absolute + 1,
                            total=len(tasks),
                            signature=plan_signature,
                            current=task,
                        )

                    idx += max(len(batch), 1)

                _save_checkpoint(
                    session,
                    position=len(tasks),
                    total=len(tasks),
                    signature=plan_signature,
                    completed=True,
                )

            finally:
                client.close()

        df = _combine_datasets(datasets, start_ms=start_ms, end_ms=end_ms)
        if df.empty:
            raise SystemExit("No data fetched; cannot compute multipliers")
        _write_dataset(data_path, df)

    else:
        if not data_path.exists():
            raise SystemExit(
                "Data file not found. Provide --start/--end to fetch history or point --data to existing snapshot."
            )
        df = load_ohlcv(data_path)

    multipliers = compute_multipliers(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "hour_of_week_definition": "0=Monday 00:00 UTC",
        "liquidity": multipliers.tolist(),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    checksum_path = write_checksum(data_path)
    print(f"Saved liquidity seasonality to {args.out}")
    print(f"Input data checksum written to {checksum_path}")


if __name__ == "__main__":
    main()
