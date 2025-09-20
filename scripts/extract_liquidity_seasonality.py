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
import logging
import signal
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


logger = logging.getLogger(__name__)


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
            "end_ms": int(self.end_exclusive()),
        }

    def end_exclusive(self) -> int:
        step_ms = INTERVAL_TO_MS[self.interval]
        return int(self.start_ms + max(self.bars, 0) * step_ms)

    def range_tuple(self) -> Tuple[int, int]:
        return int(self.start_ms), self.end_exclusive()


@dataclass(frozen=True)
class PlannedFetch:
    task: FetchTask
    cached: bool = False


@dataclass
class PlanResult:
    tasks: List[PlannedFetch]
    datasets: Dict[Tuple[str, str], pd.DataFrame]
    summary: Dict[str, Any]


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
        logger.warning("rest budget config not found: %s", path)
        return {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("failed to load rest budget config %s: %s", path, exc)
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


def _infer_chunk_days(step_ms: int, limit: int) -> int:
    bars = max(int(limit), 1)
    try:
        span_ms = int(step_ms) * bars
    except OverflowError:  # pragma: no cover - defensive
        span_ms = step_ms
    approx_days = span_ms // 86_400_000
    return max(int(approx_days), 1)


def _build_kline_request(
    client: BinancePublicClient,
    *,
    market: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> Tuple[str, Dict[str, Any]]:
    if market == "spot":
        base = client.e.spot_base
        path = "/api/v3/klines"
    else:
        base = client.e.futures_base
        path = "/fapi/v1/klines"
    url = f"{base}{path}"
    params: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": int(limit),
    }
    params["startTime"] = int(start_ms)
    params["endTime"] = int(end_ms)
    return url, params


def _plan_fetch_tasks(
    session: RestBudgetSession,
    client: BinancePublicClient,
    *,
    market: str,
    symbols: Sequence[str],
    intervals: Sequence[str],
    start_ms: int,
    end_ms: int,
    limit: int,
    cache_dir: Path,
) -> PlanResult:
    cache_dir.mkdir(parents=True, exist_ok=True)
    planned: List[PlannedFetch] = []
    datasets: Dict[Tuple[str, str], pd.DataFrame] = {}
    summary: Dict[str, Any] = {
        "total_missing": 0,
        "total_tasks": 0,
        "cached_tasks": 0,
        "cached_bars": 0,
        "per_pair": {},
        "symbols": list(symbols),
        "intervals": list(intervals),
        "start_ms": int(start_ms),
        "end_ms": int(end_ms),
        "limit": int(limit),
        "market": market,
    }

    window_ranges: Dict[str, List[Tuple[int, int]]] = {}
    aligned_bounds: Dict[str, Tuple[int, int]] = {}
    for interval in intervals:
        step_ms = INTERVAL_TO_MS[interval]
        aligned_start, aligned_end = _align_range(start_ms, end_ms, step_ms)
        aligned_bounds[interval] = (aligned_start, aligned_end)
        chunk_days = _infer_chunk_days(step_ms, limit)
        windows = split_time_range(aligned_start, aligned_end, chunk_days=chunk_days)
        ranges: List[Tuple[int, int]] = []
        for win_start, win_stop in windows:
            if win_stop <= win_start:
                continue
            inclusive_end = win_stop - step_ms
            if inclusive_end < win_start:
                continue
            ranges.append((win_start, inclusive_end))
        if not ranges:
            candidate_end = aligned_end - step_ms
            if candidate_end >= aligned_start:
                ranges.append((aligned_start, candidate_end))
        window_ranges[interval] = ranges

    for symbol in symbols:
        for interval in intervals:
            step_ms = INTERVAL_TO_MS[interval]
            aligned_start, aligned_end = aligned_bounds[interval]
            cache_path = _cache_path(cache_dir, symbol, interval)
            df = _load_cache(cache_path)
            datasets[(symbol, interval)] = df

            ranges = list(
                _iter_missing_ranges(
                    df, start_ms=aligned_start, end_ms=aligned_end, step_ms=step_ms
                )
            )
            chunked_ranges: List[Tuple[int, int]] = []
            for rng_start, rng_end in ranges:
                for win_start, win_end in window_ranges[interval]:
                    if rng_end < win_start or rng_start > win_end:
                        continue
                    chunk_start = max(rng_start, win_start)
                    chunk_end = min(rng_end, win_end)
                    if chunk_end < chunk_start:
                        continue
                    chunked_ranges.append((chunk_start, chunk_end))

            pair_tasks = _split_ranges_to_tasks(
                chunked_ranges,
                step_ms=step_ms,
                limit=limit,
                symbol=symbol,
                interval=interval,
            )

            pair_missing = sum(task.bars for task in pair_tasks)
            if not df.empty:
                ts_series = pd.to_numeric(df["ts_ms"], errors="coerce").dropna().astype("int64")
                mask = (ts_series >= aligned_start) & (ts_series < aligned_end)
                existing_bars = int(mask.sum())
            else:
                existing_bars = 0

            cached_tasks = 0
            cached_bars = 0
            for task in pair_tasks:
                end_exclusive = task.end_exclusive()
                if end_exclusive <= task.start_ms:
                    planned.append(PlannedFetch(task, cached=False))
                    continue
                end_param = end_exclusive - 1
                url, params = _build_kline_request(
                    client,
                    market=market,
                    symbol=task.symbol,
                    interval=task.interval,
                    start_ms=task.start_ms,
                    end_ms=end_param,
                    limit=min(limit, task.bars),
                )
                cached = False
                try:
                    cached = bool(
                        session.is_cached(
                            url,
                            method="GET",
                            params=params,
                            budget="klines",
                        )
                    )
                except Exception:  # pragma: no cover - cache inspection best effort
                    cached = False
                if cached:
                    cached_tasks += 1
                    cached_bars += task.bars
                planned.append(PlannedFetch(task, cached=cached))

            summary["total_missing"] += pair_missing
            summary["total_tasks"] += len(pair_tasks)
            summary["cached_tasks"] += cached_tasks
            summary["cached_bars"] += cached_bars
            summary["per_pair"][f"{symbol}_{interval}"] = {
                "missing_bars": pair_missing,
                "existing_bars": existing_bars,
                "tasks": len(pair_tasks),
                "cached_tasks": cached_tasks,
                "cached_bars": cached_bars,
            }

    planned.sort(key=lambda item: (item.task.symbol, item.task.interval, item.task.start_ms))
    return PlanResult(planned, datasets, summary)


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
    last_symbol: str | None = None,
    last_range: Tuple[int, int] | None = None,
) -> Dict[str, Any]:
    safe_total = max(int(total), 0)
    safe_position = int(position)
    if safe_total > 0:
        safe_position = max(0, min(safe_position, safe_total))
    else:
        safe_position = max(0, safe_position)

    payload: dict[str, Any] = {
        "task_index": safe_position,
        "tasks_total": safe_total,
        "signature": dict(signature),
    }

    normalized_symbol = last_symbol.strip().upper() if isinstance(last_symbol, str) else None
    normalized_range = tuple(int(x) for x in last_range) if last_range is not None else None

    if current is not None:
        payload["current"] = current.to_checkpoint()
        normalized_symbol = current.symbol.upper()
        normalized_range = current.range_tuple()

    if completed:
        payload["completed"] = True

    if safe_total <= 0:
        progress_pct = 100.0 if completed else None
    else:
        progress_pct = (safe_position / safe_total) * 100.0
        if completed:
            progress_pct = 100.0
    if progress_pct is not None:
        payload["progress_pct"] = progress_pct

    session.save_checkpoint(
        payload,
        last_symbol=normalized_symbol,
        last_range=list(normalized_range) if normalized_range is not None else None,
        progress_pct=progress_pct,
    )

    return {
        "payload": payload,
        "last_symbol": normalized_symbol,
        "last_range": list(normalized_range) if normalized_range is not None else None,
        "progress_pct": progress_pct,
        "completed": completed,
    }


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
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
        logger.warning("--dry-run ignored because --start/--end are not provided")
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

        limit = max(1, int(args.limit))
        checkpoint_path = args.checkpoint_path.strip() or str(cache_dir / "checkpoint.json")

        datasets: Dict[Tuple[str, str], pd.DataFrame] = {}
        session_stats: Dict[str, Any] | None = None

        with RestBudgetSession(
            rest_cfg,
            mode=args.cache_mode,
            checkpoint_path=checkpoint_path,
            checkpoint_enabled=bool(checkpoint_path),
            resume_from_checkpoint=args.resume_from_checkpoint,
        ) as session:
            client = BinancePublicClient(session=session)
            try:
                plan_result = _plan_fetch_tasks(
                    session,
                    client,
                    market=args.market,
                    symbols=symbols,
                    intervals=intervals,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    limit=limit,
                    cache_dir=cache_dir,
                )
                summary = plan_result.summary
                logger.info("Fetch plan summary: %s", _plan_description(summary))

                if args.dry_run:
                    session_stats = session.stats()
                    logger.info(
                        "REST session stats: %s",
                        json.dumps(session_stats, ensure_ascii=False),
                    )
                    return

                datasets = plan_result.datasets
                tasks_with_meta = plan_result.tasks
                tasks = [item.task for item in tasks_with_meta]

                plan_signature = {
                    "symbols": list(symbols),
                    "intervals": list(intervals),
                    "start_ms": int(start_ms),
                    "end_ms": int(end_ms),
                    "limit": int(limit),
                    "market": args.market,
                }

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
                        limit=min(limit, t.bars),
                    )
                    return _raw_to_df(raw, t.symbol, t.interval)

                checkpoint_payload = (
                    session.load_checkpoint() if args.resume_from_checkpoint else None
                )
                start_index = _determine_start_index(
                    checkpoint_payload,
                    signature=plan_signature,
                    total_tasks=len(tasks),
                )

                last_symbol_saved: str | None = None
                last_range_saved: Tuple[int, int] | None = None
                if isinstance(checkpoint_payload, Mapping):
                    symbol_candidate = checkpoint_payload.get("last_symbol")
                    if isinstance(symbol_candidate, str):
                        symbol_text = symbol_candidate.strip().upper()
                        last_symbol_saved = symbol_text or None
                    range_candidate = checkpoint_payload.get("last_range")
                    if isinstance(range_candidate, Sequence) and len(range_candidate) == 2:
                        try:
                            start_range = int(range_candidate[0])
                            end_range = int(range_candidate[1])
                        except (TypeError, ValueError):
                            last_range_saved = None
                        else:
                            last_range_saved = (start_range, end_range)

                checkpoint_state: Dict[str, Any] = {
                    "payload": None,
                    "last_symbol": last_symbol_saved,
                    "last_range": list(last_range_saved) if last_range_saved is not None else None,
                    "progress_pct": None,
                    "completed": False,
                }

                state = _save_checkpoint(
                    session,
                    position=start_index,
                    total=len(tasks),
                    signature=plan_signature,
                    last_symbol=last_symbol_saved,
                    last_range=last_range_saved,
                )
                checkpoint_state.update(state)

                handled_signals: Dict[int, Any] = {}

                def _handle_signal(signum: int, frame: Any | None) -> None:  # pragma: no cover - signal handler
                    payload = checkpoint_state.get("payload")
                    if isinstance(payload, Mapping):
                        session.save_checkpoint(
                            payload,
                            last_symbol=checkpoint_state.get("last_symbol"),
                            last_range=checkpoint_state.get("last_range"),
                            progress_pct=checkpoint_state.get("progress_pct"),
                        )
                    if signum == getattr(signal, "SIGINT", None):
                        raise KeyboardInterrupt
                    raise SystemExit(128 + int(signum))

                for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
                    if sig is None:
                        continue
                    try:
                        handled_signals[sig] = signal.getsignal(sig)
                        signal.signal(sig, _handle_signal)
                    except (ValueError, OSError):  # pragma: no cover - platform dependent
                        handled_signals.pop(sig, None)

                try:
                    batch_pref = int(getattr(session, "batch_size", 0) or 0)
                    worker_pref = int(getattr(session, "max_workers", 0) or 0)
                    batch_size = max(1, batch_pref or worker_pref or 1)

                    idx = start_index
                    last_symbol_seen = last_symbol_saved
                    last_range_seen = last_range_saved

                    while idx < len(tasks):
                        batch = tasks[idx : idx + batch_size]
                        futures: List[Tuple[int, FetchTask, Future[pd.DataFrame]]] = []
                        for offset, task in enumerate(batch):
                            absolute = idx + offset
                            state = _save_checkpoint(
                                session,
                                position=absolute,
                                total=len(tasks),
                                signature=plan_signature,
                                current=task,
                            )
                            checkpoint_state.update(state)
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
                            last_symbol_seen = task.symbol.upper()
                            last_range_seen = task.range_tuple()
                            state = _save_checkpoint(
                                session,
                                position=absolute + 1,
                                total=len(tasks),
                                signature=plan_signature,
                                last_symbol=last_symbol_seen,
                                last_range=last_range_seen,
                            )
                            checkpoint_state.update(state)

                        idx += max(len(batch), 1)

                    final_state = _save_checkpoint(
                        session,
                        position=len(tasks),
                        total=len(tasks),
                        signature=plan_signature,
                        last_symbol=last_symbol_seen,
                        last_range=last_range_seen,
                        completed=True,
                    )
                    checkpoint_state.update(final_state)
                finally:
                    for sig, previous in handled_signals.items():
                        try:
                            signal.signal(sig, previous)
                        except (ValueError, OSError):  # pragma: no cover - platform dependent
                            pass

                session_stats = session.stats()
            finally:
                client.close()

        if session_stats is not None:
            logger.info("REST session stats: %s", json.dumps(session_stats, ensure_ascii=False))

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
    logger.info("Saved liquidity seasonality to %s", args.out)
    logger.info("Input data checksum written to %s", checksum_path)


if __name__ == "__main__":
    main()
