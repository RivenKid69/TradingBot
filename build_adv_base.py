"""Helpers for fetching OHLCV history for ADV dataset builds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import pandas as pd

from binance_public import BinancePublicClient
from services.rest_budget import RestBudgetSession, split_time_range


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
    """Single fetch request covering ``bars`` klines starting at ``start_ms``."""

    symbol: str
    start_ms: int
    bars: int

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "start_ms": int(self.start_ms),
            "bars": int(self.bars),
        }


@dataclass
class BuildAdvConfig:
    """Configuration for :func:`build_adv`."""

    market: str
    interval: str
    start_ms: int
    end_ms: int
    out_path: Path
    cache_dir: Path
    limit: int = 1500
    chunk_days: int = 30
    resume_from_checkpoint: bool = False


@dataclass
class BuildAdvResult:
    """Result of :func:`build_adv` execution."""

    out_path: Path
    rows_written: int
    tasks_total: int
    tasks_completed: int
    bars_fetched: int
    start_ms: int
    end_ms: int
    interval: str
    per_symbol: dict[str, Mapping[str, int]]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "out_path": str(self.out_path),
            "rows_written": int(self.rows_written),
            "tasks_total": int(self.tasks_total),
            "tasks_completed": int(self.tasks_completed),
            "bars_fetched": int(self.bars_fetched),
            "start_ms": int(self.start_ms),
            "end_ms": int(self.end_ms),
            "interval": self.interval,
            "per_symbol": {k: dict(v) for k, v in self.per_symbol.items()},
        }
        return payload


def _align_range(start_ms: int, end_ms: int, step_ms: int) -> tuple[int, int]:
    aligned_start = (int(start_ms) // step_ms) * step_ms
    aligned_end = ((int(end_ms) + step_ms - 1) // step_ms) * step_ms
    return aligned_start, aligned_end


def _extract_ts(existing: pd.DataFrame) -> list[int]:
    if existing.empty or "ts_ms" not in existing.columns:
        return []
    series = pd.to_numeric(existing["ts_ms"], errors="coerce").dropna()
    if series.empty:
        return []
    return [int(v) for v in series.astype("int64")]  # type: ignore[list-item]


def _iter_missing_ranges(
    timestamps: Sequence[int],
    *,
    start_ms: int,
    end_ms: int,
    step_ms: int,
) -> Iterator[tuple[int, int]]:
    if start_ms >= end_ms:
        return
    sorted_ts = sorted(ts for ts in timestamps if start_ms <= ts < end_ms)
    index = 0
    length = len(sorted_ts)
    current_missing: int | None = None
    current = start_ms
    last_valid = end_ms - step_ms
    if last_valid < start_ms:
        return
    while current < end_ms:
        next_existing = sorted_ts[index] if index < length else None
        if next_existing == current:
            if current_missing is not None:
                yield current_missing, current - step_ms
                current_missing = None
            index += 1
            while index < length and sorted_ts[index] == next_existing:
                index += 1
        else:
            if current_missing is None:
                current_missing = current
        current += step_ms
    if current_missing is not None:
        yield current_missing, min(last_valid, end_ms - step_ms)


def _split_ranges_to_tasks(
    ranges: Iterable[tuple[int, int]],
    *,
    step_ms: int,
    limit: int,
    symbol: str,
) -> list[FetchTask]:
    tasks: list[FetchTask] = []
    max_bars = max(1, int(limit))
    for start, end in ranges:
        if end < start:
            continue
        bars_total = int((end - start) // step_ms) + 1
        remaining = bars_total
        cursor = start
        while remaining > 0:
            chunk = min(remaining, max_bars)
            tasks.append(FetchTask(symbol=symbol, start_ms=cursor, bars=chunk))
            cursor += chunk * step_ms
            remaining -= chunk
    return tasks


def _cache_path(cache_dir: Path, symbol: str, interval: str) -> Path:
    safe = symbol.upper()
    return cache_dir / f"{safe}_{interval}.parquet"


def _load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=KLINE_COLUMNS)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_dataset(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _raw_to_df(raw: Sequence[Sequence[Any]], symbol: str) -> pd.DataFrame:
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
            "ts_ms": pd.to_numeric(df["open_time"], errors="coerce").astype("Int64"),
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
    return out[KLINE_COLUMNS]


def _merge_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return incoming.copy()
    if incoming.empty:
        return existing
    merged = (
        pd.concat([existing, incoming], ignore_index=True)
        .drop_duplicates(subset=["symbol", "ts_ms"], keep="last")
        .sort_values(["symbol", "ts_ms"])
        .reset_index(drop=True)
    )
    return merged


def _combine_datasets(
    datasets: Mapping[str, pd.DataFrame],
    *,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, df in datasets.items():
        if df.empty:
            continue
        subset = df.copy()
        subset["ts_ms"] = pd.to_numeric(subset["ts_ms"], errors="coerce")
        subset = subset.dropna(subset=["ts_ms"])
        if subset.empty:
            continue
        subset["ts_ms"] = subset["ts_ms"].astype("int64")
        mask = (subset["ts_ms"] >= int(start_ms)) & (subset["ts_ms"] < int(end_ms))
        subset = subset.loc[mask]
        if subset.empty:
            continue
        subset = subset.assign(symbol=symbol.upper())
        frames.append(subset[KLINE_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.drop_duplicates(subset=["symbol", "ts_ms"], keep="last")
        .sort_values(["symbol", "ts_ms"])
        .reset_index(drop=True)
    )
    return combined


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
    stored_signature = checkpoint.get("signature")
    if stored_signature != dict(signature):
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


def build_adv(
    session: RestBudgetSession,
    symbols: Sequence[str],
    config: BuildAdvConfig,
) -> BuildAdvResult:
    if not symbols:
        raise ValueError("symbol list is empty")

    interval = config.interval.lower()
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"unsupported interval: {config.interval}")

    step_ms = INTERVAL_TO_MS[interval]
    start_aligned, end_aligned = _align_range(config.start_ms, config.end_ms, step_ms)
    if end_aligned <= start_aligned:
        raise ValueError("end must be greater than start")

    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[FetchTask] = []
    datasets: dict[str, pd.DataFrame] = {}
    existing_counts: dict[str, int] = {}

    windows = split_time_range(start_aligned, end_aligned, chunk_days=config.chunk_days)
    if not windows:
        windows = [(start_aligned, end_aligned)]
    window_ranges: list[tuple[int, int]] = []
    for win_start, win_stop in windows:
        if win_stop <= win_start:
            continue
        inclusive_end = win_stop - step_ms
        if inclusive_end < win_start:
            continue
        window_ranges.append((win_start, inclusive_end))
    if not window_ranges:
        window_ranges.append((start_aligned, end_aligned - step_ms))

    for symbol in symbols:
        cache_path = _cache_path(cache_dir, symbol, interval)
        dataset = _load_cache(cache_path)
        datasets[symbol] = dataset
        timestamps = _extract_ts(dataset)
        full_ranges = list(
            _iter_missing_ranges(
                timestamps,
                start_ms=start_aligned,
                end_ms=end_aligned,
                step_ms=step_ms,
            )
        )
        chunked_ranges: list[tuple[int, int]] = []
        for rng_start, rng_end in full_ranges:
            for win_start, win_end in window_ranges:
                if rng_end < win_start or rng_start > win_end:
                    continue
                chunk_start = max(rng_start, win_start)
                chunk_end = min(rng_end, win_end)
                if chunk_end < chunk_start:
                    continue
                chunked_ranges.append((chunk_start, chunk_end))
        symbol_tasks = _split_ranges_to_tasks(
            chunked_ranges,
            step_ms=step_ms,
            limit=config.limit,
            symbol=symbol,
        )
        tasks.extend(symbol_tasks)
        existing = 0
        if dataset is not None and not dataset.empty:
            ts_series = (
                pd.to_numeric(dataset["ts_ms"], errors="coerce").dropna().astype("int64")
            )
            mask = (ts_series >= start_aligned) & (ts_series < end_aligned)
            existing = int(mask.sum())
        existing_counts[symbol] = existing

    tasks.sort(key=lambda t: (t.symbol, t.start_ms))
    plan_signature: dict[str, Any] = {
        "market": config.market,
        "interval": interval,
        "start": int(start_aligned),
        "end": int(end_aligned),
        "symbols": [s.upper() for s in symbols],
        "limit": int(config.limit),
        "chunk_days": int(config.chunk_days),
    }

    start_index = 0
    checkpoint_payload: Mapping[str, Any] | None = None
    if config.resume_from_checkpoint:
        checkpoint_payload = session.load_checkpoint()
        if isinstance(checkpoint_payload, MutableMapping):
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

    fetched_counts: dict[str, int] = {symbol: 0 for symbol in symbols}
    tasks_completed = start_index
    bars_fetched = 0

    client = BinancePublicClient(session=session)
    try:
        idx = start_index
        while idx < len(tasks):
            batch = tasks[idx : idx + batch_size]
            futures: list[tuple[int, FetchTask]] = []
            for offset, task in enumerate(batch):
                absolute = idx + offset
                _save_checkpoint(
                    session,
                    position=absolute,
                    total=len(tasks),
                    signature=plan_signature,
                    current=task,
                )
                futures.append((absolute, task))

            results: list[tuple[int, FetchTask, pd.DataFrame]] = []
            for absolute, task in futures:
                step = step_ms
                window_end = task.start_ms + task.bars * step
                raw = client.get_klines(
                    market=config.market,
                    symbol=task.symbol,
                    interval=interval,
                    start_ms=task.start_ms,
                    end_ms=window_end - 1,
                    limit=min(config.limit, task.bars),
                )
                frame = _raw_to_df(raw, task.symbol)
                if not frame.empty:
                    frame = frame[
                        (pd.to_numeric(frame["ts_ms"], errors="coerce") >= task.start_ms)
                        & (
                            pd.to_numeric(frame["ts_ms"], errors="coerce")
                            < task.start_ms + task.bars * step
                        )
                    ]
                results.append((absolute, task, frame))

            for absolute, task, frame in results:
                key = task.symbol
                dataset = datasets[key]
                if not frame.empty:
                    datasets[key] = dataset = _merge_frames(dataset, frame)
                    bars = int(len(frame))
                    fetched_counts[key] += bars
                    bars_fetched += bars
                    cache_path = _cache_path(cache_dir, key, interval)
                    _write_dataset(cache_path, dataset)
                _save_checkpoint(
                    session,
                    position=absolute + 1,
                    total=len(tasks),
                    signature=plan_signature,
                    current=task,
                )
                tasks_completed = max(tasks_completed, absolute + 1)

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

    combined = _combine_datasets(datasets, start_ms=start_aligned, end_ms=end_aligned)
    combined = combined[
        (combined["ts_ms"] >= start_aligned) & (combined["ts_ms"] < end_aligned)
    ]
    combined = combined.sort_values(["symbol", "ts_ms"]).reset_index(drop=True)

    _write_dataset(config.out_path, combined)

    per_symbol: dict[str, dict[str, int]] = {}
    for symbol in symbols:
        dataset = datasets[symbol]
        if dataset.empty:
            total = 0
        else:
            ts_series = (
                pd.to_numeric(dataset["ts_ms"], errors="coerce").dropna().astype("int64")
            )
            mask = (ts_series >= start_aligned) & (ts_series < end_aligned)
            total = int(mask.sum())
        per_symbol[symbol] = {
            "existing_bars": int(existing_counts.get(symbol, 0)),
            "fetched_bars": int(fetched_counts.get(symbol, 0)),
            "total_bars": total,
        }

    return BuildAdvResult(
        out_path=config.out_path,
        rows_written=int(len(combined)),
        tasks_total=len(tasks),
        tasks_completed=tasks_completed,
        bars_fetched=bars_fetched,
        start_ms=start_aligned,
        end_ms=end_aligned,
        interval=interval,
        per_symbol=per_symbol,
    )


def fetch_klines_for_symbols(
    session: RestBudgetSession,
    symbols: Sequence[str],
    *,
    market: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1500,
) -> Mapping[str, pd.DataFrame]:
    """Fetch kline history for ``symbols`` within the supplied time range.

    The helper returns a mapping of symbol → DataFrame with the same column
    schema as :data:`KLINE_COLUMNS`.  Missing symbols are mapped to empty
    frames.  The range is interpreted as a half-open interval
    ``[start_ms, end_ms)``.
    """

    if not symbols:
        return {}

    interval_key = str(interval).lower()
    if interval_key not in INTERVAL_TO_MS:
        raise ValueError(f"unsupported interval: {interval}")

    step_ms = INTERVAL_TO_MS[interval_key]
    start_aligned, end_aligned = _align_range(start_ms, end_ms, step_ms)
    if end_aligned <= start_aligned:
        return {s.upper(): pd.DataFrame(columns=KLINE_COLUMNS) for s in symbols}

    safe_limit = max(1, int(limit))
    datasets: dict[str, pd.DataFrame] = {}

    client = BinancePublicClient(session=session)
    try:
        for raw_symbol in symbols:
            symbol = str(raw_symbol).strip().upper()
            if not symbol:
                continue
            tasks = _split_ranges_to_tasks(
                [(start_aligned, end_aligned - step_ms)],
                step_ms=step_ms,
                limit=safe_limit,
                symbol=symbol,
            )
            dataset = pd.DataFrame(columns=KLINE_COLUMNS)
            for task in tasks:
                window_end = task.start_ms + task.bars * step_ms
                window_end = min(window_end, end_aligned)
                raw = client.get_klines(
                    market=market,
                    symbol=task.symbol,
                    interval=interval_key,
                    start_ms=task.start_ms,
                    end_ms=window_end - 1,
                    limit=min(safe_limit, task.bars),
                )
                frame = _raw_to_df(raw, task.symbol)
                if not frame.empty:
                    ts_numeric = pd.to_numeric(frame["ts_ms"], errors="coerce")
                    frame = frame.loc[
                        (ts_numeric >= start_aligned) & (ts_numeric < end_aligned)
                    ]
                if not frame.empty:
                    dataset = _merge_frames(dataset, frame)
            if not dataset.empty:
                ts_numeric = pd.to_numeric(dataset["ts_ms"], errors="coerce")
                dataset = dataset.loc[
                    (ts_numeric >= start_aligned) & (ts_numeric < end_aligned)
                ]
                dataset = dataset.sort_values(["symbol", "ts_ms"]).reset_index(drop=True)
            datasets[symbol] = dataset
    finally:
        client.close()

    return datasets


def aggregate_daily_quote_volume(df: pd.DataFrame) -> pd.Series:
    """Aggregate quote asset volumes to daily totals.

    Returns a :class:`pandas.Series` indexed by UTC midnight timestamps with
    daily quote volumes.  Non-numeric entries are ignored.
    """

    if df.empty:
        return pd.Series(dtype="float64")
    required = {"ts_ms", "quote_asset_volume"}
    if not required.issubset(df.columns):
        return pd.Series(dtype="float64")

    ts_numeric = pd.to_numeric(df["ts_ms"], errors="coerce")
    vol_numeric = pd.to_numeric(df["quote_asset_volume"], errors="coerce")
    mask = ts_numeric.notna() & vol_numeric.notna()
    if not mask.any():
        return pd.Series(dtype="float64")

    ts_dt = pd.to_datetime(ts_numeric[mask], unit="ms", utc=True)
    volumes = vol_numeric[mask].astype("float64")
    grouped = volumes.groupby(ts_dt.dt.floor("D")).sum(min_count=1)
    if grouped.empty:
        return pd.Series(dtype="float64")
    grouped = grouped.sort_index()
    return grouped


def compute_adv_quote(
    daily_quote_volume: pd.Series,
    *,
    window_days: int,
    min_days: int = 1,
) -> tuple[float | None, int, int]:
    """Compute average daily quote volume over the specified window.

    Returns a tuple ``(adv_quote, used_days, total_days)`` where
    ``adv_quote`` is ``None`` if insufficient data is available.
    """

    window = max(1, int(window_days))
    minimum = max(1, int(min_days))

    if daily_quote_volume.empty:
        return None, 0, 0

    series = daily_quote_volume.dropna().astype("float64").sort_index()
    total_days = int(len(series))
    if total_days == 0:
        return None, 0, 0

    window_slice = series.tail(window)
    window_slice = window_slice[window_slice > 0.0]
    used_days = int(len(window_slice))
    if used_days < minimum:
        return None, used_days, total_days

    adv_value = float(window_slice.mean())
    if not pd.notna(adv_value) or adv_value <= 0.0:
        return None, used_days, total_days

    return adv_value, used_days, total_days


__all__ = [
    "BuildAdvConfig",
    "BuildAdvResult",
    "FetchTask",
    "aggregate_daily_quote_volume",
    "build_adv",
    "compute_adv_quote",
    "fetch_klines_for_symbols",
]
