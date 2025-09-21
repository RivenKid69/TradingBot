"""Helpers for fetching OHLCV history for ADV dataset builds."""

from __future__ import annotations

import signal
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
    Tuple,
)

import pandas as pd

from binance_public import BinancePublicClient, PublicEndpoints
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
    interval: str

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "start_ms": int(self.start_ms),
            "bars": int(self.bars),
            "end_ms": int(self.end_exclusive()),
        }

    def end_exclusive(self, step_ms: int | None = None) -> int:
        if step_ms is None:
            key = self.interval.lower()
            step = INTERVAL_TO_MS.get(key)
            if step is None:
                raise KeyError(f"Unknown interval for task: {self.interval}")
        else:
            step = int(step_ms)
        bars = max(int(self.bars), 0)
        return int(self.start_ms + bars * step)

    def range_tuple(self, step_ms: int | None = None) -> Tuple[int, int]:
        return int(self.start_ms), self.end_exclusive(step_ms)


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
    dry_run: bool = False


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
    dry_run: bool = False
    plan: Mapping[str, Any] | None = None
    rest_stats: Mapping[str, Any] | None = None

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
        payload["dry_run"] = bool(self.dry_run)
        if self.plan is not None:
            payload["plan"] = dict(self.plan)
        if self.rest_stats is not None:
            payload["rest_stats"] = dict(self.rest_stats)
        return payload


def _estimate_kline_tokens(limit: int) -> float:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = 0
    value = max(value, 1)
    if value <= 100:
        return 1.0
    if value <= 500:
        return 2.0
    if value <= 1000:
        return 5.0
    return 10.0


def _resolve_kline_endpoint(market: str) -> tuple[str, str]:
    endpoints = PublicEndpoints()
    if market.lower() == "spot":
        return endpoints.spot_base, "/api/v3/klines"
    return endpoints.futures_base, "/fapi/v1/klines"


def _task_request_details(
    market: str,
    task: FetchTask,
    *,
    interval: str,
    limit: int,
    end_limit: int,
) -> tuple[str, dict[str, Any]]:
    base, path = _resolve_kline_endpoint(market)
    url = f"{base}{path}"
    end_exclusive = min(task.end_exclusive(), int(end_limit))
    params: dict[str, Any] = {
        "symbol": task.symbol.upper(),
        "interval": interval,
        "limit": int(max(1, int(limit))),
        "startTime": int(task.start_ms),
    }
    end_inclusive = int(end_exclusive - 1)
    if end_inclusive < task.start_ms:
        end_inclusive = int(task.start_ms)
    params["endTime"] = end_inclusive
    return url, params


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
    interval: str,
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
            tasks.append(
                FetchTask(
                    symbol=symbol,
                    start_ms=cursor,
                    bars=chunk,
                    interval=interval,
                )
            )
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
    last_symbol: str | None = None,
    last_range: Tuple[int, int] | None = None,
) -> dict[str, Any]:
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



def build_adv(
    session: RestBudgetSession,
    symbols: Sequence[str],
    config: BuildAdvConfig,
    *,
    stats_path: str | Path | None = None,
) -> BuildAdvResult:
    if not symbols:
        raise ValueError("symbol list is empty")

    interval = config.interval.lower()
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"unsupported interval: {config.interval}")

    stats_target = stats_path

    def _write_stats_safe() -> None:
        if not stats_target:
            return
        try:
            session.write_stats(stats_target)
        except Exception:
            pass

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

    normalized_symbols: list[str] = []
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol:
            continue
        normalized_symbols.append(symbol)
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
            interval=interval,
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

    if not normalized_symbols:
        raise ValueError("symbol list is empty")

    tasks.sort(key=lambda t: (t.symbol, t.start_ms))
    plan_signature: dict[str, Any] = {
        "market": config.market,
        "interval": interval,
        "start": int(start_aligned),
        "end": int(end_aligned),
        "symbols": list(normalized_symbols),
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
                start_val = int(range_candidate[0])
                end_val = int(range_candidate[1])
            except (TypeError, ValueError):
                last_range_saved = None
            else:
                last_range_saved = (start_val, end_val)

    batch_pref = int(getattr(session, "batch_size", 0) or 0)
    worker_pref = int(getattr(session, "max_workers", 0) or 0)
    batch_size = max(1, batch_pref or worker_pref or 1)

    plan_symbol_summary: dict[str, dict[str, int]] = {}
    for sym in normalized_symbols:
        plan_symbol_summary[sym] = {
            "existing_bars": int(existing_counts.get(sym, 0)),
            "pending_tasks": 0,
            "pending_bars": 0,
            "cached_tasks": 0,
            "cached_bars": 0,
        }

    include_details = bool(config.dry_run)
    planned_details: list[dict[str, Any]] = [] if include_details else []
    pending_bars_total = 0
    cached_tasks_total = 0
    cached_bars_total = 0

    plan_callable = getattr(session, "plan_request", None)
    if not callable(plan_callable):
        plan_callable = None
    cache_callable = getattr(session, "is_cached", None)
    if not callable(cache_callable):
        cache_callable = None

    for absolute, task in enumerate(tasks):
        if absolute < start_index:
            continue
        limit_value = min(int(config.limit), int(task.bars))
        if plan_callable is not None:
            try:
                plan_callable("klines", tokens=_estimate_kline_tokens(limit_value))
            except Exception:
                pass
        cached = False
        if cache_callable is not None:
            try:
                url, params = _task_request_details(
                    config.market,
                    task,
                    interval=interval,
                    limit=limit_value,
                    end_limit=end_aligned,
                )
                cached = bool(
                    cache_callable(
                        url,
                        method="GET",
                        params=params,
                        budget="klines",
                    )
                )
            except Exception:
                cached = False
        summary = plan_symbol_summary[task.symbol]
        summary["pending_tasks"] += 1
        summary["pending_bars"] += int(task.bars)
        pending_bars_total += int(task.bars)
        if cached:
            summary["cached_tasks"] += 1
            summary["cached_bars"] += int(task.bars)
            cached_tasks_total += 1
            cached_bars_total += int(task.bars)
        if include_details:
            end_exclusive = min(task.end_exclusive(step_ms), end_aligned)
            planned_details.append(
                {
                    "symbol": task.symbol,
                    "start_ms": int(task.start_ms),
                    "end_ms": int(end_exclusive),
                    "bars": int(task.bars),
                    "limit": int(limit_value),
                    "cached": bool(cached),
                }
            )

    pending_task_count = max(len(tasks) - start_index, 0)
    per_symbol_plan = {
        sym: {
            "existing_bars": int(summary["existing_bars"]),
            "pending_tasks": int(summary["pending_tasks"]),
            "pending_bars": int(summary["pending_bars"]),
            "cached_tasks": int(summary["cached_tasks"]),
            "cached_bars": int(summary["cached_bars"]),
        }
        for sym, summary in plan_symbol_summary.items()
    }
    plan_payload: dict[str, Any] = {
        "resume_from_task": int(start_index),
        "tasks_total": int(len(tasks)),
        "pending_tasks": int(pending_task_count),
        "pending_tasks_remaining": int(pending_task_count),
        "pending_bars": int(pending_bars_total),
        "pending_bars_remaining": int(pending_bars_total),
        "cached_tasks": int(cached_tasks_total),
        "cached_bars": int(cached_bars_total),
        "existing_bars": int(sum(existing_counts.values())),
        "per_symbol": per_symbol_plan,
    }
    if include_details:
        plan_payload["tasks"] = planned_details

    if config.dry_run:
        rest_stats = session.stats()
        per_symbol_result: dict[str, dict[str, int]] = {}
        for sym in normalized_symbols:
            existing = int(existing_counts.get(sym, 0))
            per_symbol_result[sym] = {
                "existing_bars": existing,
                "fetched_bars": 0,
                "total_bars": existing,
            }
        _write_stats_safe()
        return BuildAdvResult(
            out_path=config.out_path,
            rows_written=0,
            tasks_total=len(tasks),
            tasks_completed=start_index,
            bars_fetched=0,
            start_ms=start_aligned,
            end_ms=end_aligned,
            interval=interval,
            per_symbol=per_symbol_result,
            dry_run=True,
            plan=plan_payload,
            rest_stats=rest_stats,
        )

    fetched_counts: dict[str, int] = {sym: 0 for sym in normalized_symbols}
    tasks_completed = start_index
    bars_fetched = 0

    checkpoint_state: dict[str, Any] = {
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
    _write_stats_safe()

    handled_signals: dict[int, Any] = {}

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

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            previous = signal.getsignal(sig)
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            continue
        else:
            handled_signals[sig] = previous

    client = BinancePublicClient(session=session)

    def _fetch_single(fetch_task: FetchTask) -> pd.DataFrame:
        window_end = min(fetch_task.end_exclusive(step_ms), end_aligned)
        raw = client.get_klines(
            market=config.market,
            symbol=fetch_task.symbol,
            interval=interval,
            start_ms=fetch_task.start_ms,
            end_ms=window_end - 1,
            limit=min(int(config.limit), int(fetch_task.bars)),
        )
        frame = _raw_to_df(raw, fetch_task.symbol)
        if frame.empty:
            return frame
        ts_numeric = pd.to_numeric(frame["ts_ms"], errors="coerce")
        mask = (ts_numeric >= fetch_task.start_ms) & (ts_numeric < window_end)
        return frame.loc[mask]

    last_symbol_seen = last_symbol_saved
    last_range_seen = last_range_saved

    try:
        idx = start_index
        while idx < len(tasks):
            batch = tasks[idx : idx + batch_size]
            futures: list[tuple[int, FetchTask, Future[pd.DataFrame]]] = []
            for offset, task in enumerate(batch):
                absolute = idx + offset
                state = _save_checkpoint(
                    session,
                    position=absolute,
                    total=len(tasks),
                    signature=plan_signature,
                    current=task,
                    last_symbol=last_symbol_seen,
                    last_range=last_range_seen,
                )
                checkpoint_state.update(state)
                _write_stats_safe()
                future = session.submit(_fetch_single, task)
                futures.append((absolute, task, future))

            for absolute, task, future in futures:
                try:
                    frame = future.result()
                except Exception as exc:  # pragma: no cover - network dependent
                    raise RuntimeError(
                        f"Failed to fetch {task.symbol} {interval} starting at {task.start_ms}: {exc}"
                    ) from exc
                dataset = datasets[task.symbol]
                if not frame.empty:
                    dataset = _merge_frames(dataset, frame)
                    datasets[task.symbol] = dataset
                    bars = int(len(frame))
                    fetched_counts[task.symbol] += bars
                    bars_fetched += bars
                    cache_path = _cache_path(cache_dir, task.symbol, interval)
                    _write_dataset(cache_path, dataset)
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
                tasks_completed = max(tasks_completed, absolute + 1)
                _write_stats_safe()
            idx += max(len(batch), 1)

        final_state = _save_checkpoint(
            session,
            position=len(tasks),
            total=len(tasks),
            signature=plan_signature,
            completed=True,
            last_symbol=last_symbol_seen,
            last_range=last_range_seen,
        )
        checkpoint_state.update(final_state)
        _write_stats_safe()
    finally:
        for sig, previous in handled_signals.items():
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):
                continue
        try:
            client.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    combined = _combine_datasets(datasets, start_ms=start_aligned, end_ms=end_aligned)
    combined = combined[
        (combined["ts_ms"] >= start_aligned) & (combined["ts_ms"] < end_aligned)
    ]
    combined = combined.sort_values(["symbol", "ts_ms"]).reset_index(drop=True)

    _write_dataset(config.out_path, combined)
    _write_stats_safe()

    per_symbol: dict[str, dict[str, int]] = {}
    for sym in normalized_symbols:
        dataset = datasets[sym]
        if dataset.empty:
            total = 0
        else:
            ts_series = pd.to_numeric(dataset["ts_ms"], errors="coerce").dropna().astype("int64")
            mask = (ts_series >= start_aligned) & (ts_series < end_aligned)
            total = int(mask.sum())
        per_symbol[sym] = {
            "existing_bars": int(existing_counts.get(sym, 0)),
            "fetched_bars": int(fetched_counts.get(sym, 0)),
            "total_bars": total,
        }

    remaining_tasks = max(len(tasks) - tasks_completed, 0)
    remaining_bars = max(pending_bars_total - bars_fetched, 0)
    plan_payload = dict(plan_payload)
    plan_payload["pending_tasks_remaining"] = int(remaining_tasks)
    plan_payload["pending_bars_remaining"] = int(remaining_bars)

    rest_stats = session.stats()

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
        dry_run=False,
        plan=plan_payload,
        rest_stats=rest_stats,
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
    stats_path: str | Path | None = None,
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
    stats_target = stats_path

    def _write_stats_safe() -> None:
        if not stats_target:
            return
        try:
            session.write_stats(stats_target)
        except Exception:
            pass

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
                interval=interval_key,
            )
            dataset = pd.DataFrame(columns=KLINE_COLUMNS)
            for task in tasks:
                window_end = min(task.end_exclusive(step_ms), end_aligned)
                raw = client.get_klines(
                    market=market,
                    symbol=task.symbol,
                    interval=interval_key,
                    start_ms=task.start_ms,
                    end_ms=window_end - 1,
                    limit=min(safe_limit, int(task.bars)),
                )
                frame = _raw_to_df(raw, task.symbol)
                if not frame.empty:
                    ts_numeric = pd.to_numeric(frame["ts_ms"], errors="coerce")
                    frame = frame.loc[
                        (ts_numeric >= start_aligned) & (ts_numeric < end_aligned)
                    ]
                if not frame.empty:
                    dataset = _merge_frames(dataset, frame)
                _write_stats_safe()
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


def aggregate_daily_base_volume(df: pd.DataFrame) -> pd.Series:
    """Aggregate base asset volumes to daily totals."""

    if df.empty or "ts_ms" not in df.columns or "volume" not in df.columns:
        return pd.Series(dtype="float64")

    ts_numeric = pd.to_numeric(df["ts_ms"], errors="coerce")
    base_series = pd.to_numeric(df["volume"], errors="coerce").astype("float64")

    mask = ts_numeric.notna() & base_series.notna()
    if not mask.any():
        return pd.Series(dtype="float64")

    buckets = pd.to_datetime(ts_numeric[mask], unit="ms", utc=True).dt.floor("D")
    grouped = base_series[mask].groupby(buckets).sum(min_count=1)
    if grouped.empty:
        return pd.Series(dtype="float64")
    return grouped.sort_index()


def aggregate_daily_quote_volume(df: pd.DataFrame) -> pd.Series:
    """Aggregate quote asset volumes to daily totals.

    Returns a :class:`pandas.Series` indexed by UTC midnight timestamps with
    daily quote volumes.  Non-numeric entries are ignored.
    """

    if df.empty or "ts_ms" not in df.columns:
        return pd.Series(dtype="float64")

    ts_numeric = pd.to_numeric(df["ts_ms"], errors="coerce")

    quote_series = pd.Series(float("nan"), index=df.index, dtype="float64")
    if "quote_asset_volume" in df.columns:
        quote_series = pd.to_numeric(df["quote_asset_volume"], errors="coerce").astype(
            "float64"
        )

    if "volume" in df.columns and "close" in df.columns:
        base_numeric = pd.to_numeric(df["volume"], errors="coerce")
        close_numeric = pd.to_numeric(df["close"], errors="coerce")
        fallback = (base_numeric * close_numeric).astype("float64")
        quote_series = quote_series.where(quote_series.notna(), fallback)

    mask = ts_numeric.notna() & quote_series.notna()
    if not mask.any():
        return pd.Series(dtype="float64")

    buckets = pd.to_datetime(ts_numeric[mask], unit="ms", utc=True).dt.floor("D")
    grouped = quote_series[mask].groupby(buckets).sum(min_count=1)
    if grouped.empty:
        return pd.Series(dtype="float64")
    grouped = grouped.sort_index()
    return grouped


def _compute_adv_from_series(
    daily_volume: pd.Series,
    *,
    window_days: int,
    min_days: int = 1,
    min_total_days: int | None = None,
    clip_percentiles: tuple[float, float] | None = None,
) -> tuple[float | None, int, int]:
    """Compute average daily volume over the specified window."""

    window = max(1, int(window_days))
    minimum = max(1, int(min_days))
    total_minimum = int(min_total_days) if min_total_days is not None else minimum
    if total_minimum < 0:
        total_minimum = 0

    if daily_volume.empty:
        return None, 0, 0

    numeric = pd.to_numeric(daily_volume, errors="coerce")
    series = numeric.dropna().astype("float64")
    series = series[series > 0.0].sort_index()
    total_days = int(len(series))
    if total_days == 0:
        return None, 0, 0

    if total_minimum and total_days < total_minimum:
        return None, 0, total_days

    window_slice = series.tail(window)
    if clip_percentiles is not None:
        lower, upper = clip_percentiles
        try:
            lower_val = float(lower)
            upper_val = float(upper)
        except (TypeError, ValueError):
            lower_val = float("nan")
            upper_val = float("nan")
        if (
            pd.notna(lower_val)
            and pd.notna(upper_val)
            and 0.0 <= lower_val < upper_val <= 100.0
            and not window_slice.empty
        ):
            lower_clip = window_slice.quantile(lower_val / 100.0, interpolation="linear")
            upper_clip = window_slice.quantile(upper_val / 100.0, interpolation="linear")
            window_slice = window_slice.clip(lower=lower_clip, upper=upper_clip)

    window_slice = window_slice.dropna()
    used_days = int(len(window_slice))
    if used_days < minimum:
        return None, used_days, total_days

    adv_value = float(window_slice.mean())
    if not pd.notna(adv_value) or adv_value <= 0.0:
        return None, used_days, total_days

    return adv_value, used_days, total_days


def compute_adv_quote(
    daily_quote_volume: pd.Series,
    *,
    window_days: int,
    min_days: int = 1,
    min_total_days: int | None = None,
    clip_percentiles: tuple[float, float] | None = None,
) -> tuple[float | None, int, int]:
    """Compute average daily quote volume over the specified window."""

    return _compute_adv_from_series(
        daily_quote_volume,
        window_days=window_days,
        min_days=min_days,
        min_total_days=min_total_days,
        clip_percentiles=clip_percentiles,
    )


def compute_adv_base(
    daily_base_volume: pd.Series,
    *,
    window_days: int,
    min_days: int = 1,
    min_total_days: int | None = None,
    clip_percentiles: tuple[float, float] | None = None,
) -> tuple[float | None, int, int]:
    """Compute average daily base volume over the specified window."""

    return _compute_adv_from_series(
        daily_base_volume,
        window_days=window_days,
        min_days=min_days,
        min_total_days=min_total_days,
        clip_percentiles=clip_percentiles,
    )


__all__ = [
    "BuildAdvConfig",
    "BuildAdvResult",
    "FetchTask",
    "aggregate_daily_base_volume",
    "aggregate_daily_quote_volume",
    "build_adv",
    "compute_adv_base",
    "compute_adv_quote",
    "fetch_klines_for_symbols",
]
