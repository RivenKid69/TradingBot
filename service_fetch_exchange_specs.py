from __future__ import annotations

import json
import math
import os
import random
import signal
import tempfile
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

from services.rest_budget import RestBudgetSession, iter_time_chunks


def _endpoint(market: str) -> str:
    m = str(market).lower().strip()
    if m == "spot":
        return "https://api.binance.com/api/v3/exchangeInfo"
    # по умолчанию возьмём USDT-маржинальные фьючи
    return "https://fapi.binance.com/fapi/v1/exchangeInfo"


def _endpoint_key(market: str) -> str:
    m = str(market).lower().strip()
    if m == "spot":
        return "GET /api/v3/exchangeInfo"
    return "GET /fapi/v1/exchangeInfo"


def _klines_endpoint(market: str) -> str:
    m = str(market).lower().strip()
    if m == "spot":
        return "https://api.binance.com/api/v3/klines"
    return "https://fapi.binance.com/fapi/v1/klines"


def _klines_endpoint_key(market: str) -> str:
    m = str(market).lower().strip()
    if m == "spot":
        return "GET /api/v3/klines"
    return "GET /fapi/v1/klines"


def _estimate_kline_tokens(limit: int) -> float:
    """Return a conservative weight estimate for Binance kline queries."""

    try:
        limit_val = int(limit)
    except (TypeError, ValueError):
        limit_val = 0
    limit_val = max(limit_val, 1)
    if limit_val <= 100:
        return 1.0
    if limit_val <= 500:
        return 2.0
    if limit_val <= 1000:
        return 5.0
    return 10.0


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.fspath(path)) or "."
    os.makedirs(directory, exist_ok=True)


def _write_json_atomic(path: str, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".exchange_specs_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _normalize_symbol_list(raw: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in raw:
        text = str(item).strip().upper()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _iter_symbol_chunks(symbols: Sequence[str], chunk_size: int) -> Iterable[list[str]]:
    step = max(1, int(chunk_size) if chunk_size else 1)
    for idx in range(0, len(symbols), step):
        yield list(symbols[idx : idx + step])


def run(
    market: str = "futures",
    symbols: Sequence[str] | str | None = None,
    out: str = "data/exchange_specs.json",
    volume_threshold: float = 0.0,
    volume_out: str | None = None,
    days: int = 30,
    *,
    shuffle: bool = False,
    session: RestBudgetSession | None = None,
    checkpoint_listener: Callable[[Dict[str, Any]], None] | None = None,
    install_signal_handlers: bool = True,
    chunk_size: int | None = None,
) -> Dict[str, Dict[str, float]]:
    """Fetch Binance exchangeInfo and store minimal specs JSON.

    Additionally computes average daily quote volume over the last ``days``
    for each symbol and optionally filters out symbols whose average falls
    below ``volume_threshold``.  The computed averages can be stored in
    ``volume_out`` for transparency.  When ``shuffle`` is true the symbol
    processing order is randomised (unless restored from checkpoint).
    ``session`` controls HTTP budgeting, caching and checkpoint persistence.
    When provided, ``checkpoint_listener`` is called with every checkpoint
    payload before it is persisted which can be useful for external signal
    handlers.  To manage signals outside this function pass
    ``install_signal_handlers=False``.
    """

    session = session or RestBudgetSession({})

    if isinstance(symbols, str):
        requested = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        requested = [s.strip().upper() for s in (symbols or []) if s.strip()]

    checkpoint = session.load_checkpoint()
    restored_order: list[str] = []
    if isinstance(checkpoint, Mapping):
        saved = checkpoint.get("order") or checkpoint.get("symbols")
        if isinstance(saved, Sequence):
            restored_order = _normalize_symbol_list(saved)

    chunk_size_val = max(1, int(chunk_size)) if chunk_size and chunk_size > 0 else None

    if requested:
        target_symbols = _normalize_symbol_list(requested)
    elif restored_order:
        target_symbols = restored_order
    else:
        target_symbols = []

    url = _endpoint(market)
    endpoint_key = _endpoint_key(market)
    exchange_chunks_total = 1

    symbol_entries: list[Mapping[str, Any]] = []
    if target_symbols and chunk_size_val:
        exchange_chunks_total = max(1, math.ceil(len(target_symbols) / chunk_size_val))
        for chunk in _iter_symbol_chunks(target_symbols, chunk_size_val):
            params: dict[str, Any]
            if len(chunk) == 1:
                params = {"symbol": chunk[0]}
            else:
                params = {"symbols": json.dumps(chunk)}
            cached = session.is_cached(
                url,
                params=params,
                endpoint=endpoint_key,
                budget="exchangeInfo",
            )
            if cached:
                print(f"Using cached exchangeInfo chunk of {len(chunk)} symbols")
            payload = session.get(
                url,
                params=params,
                endpoint=endpoint_key,
                budget="exchangeInfo",
                tokens=10.0,
            )
            if isinstance(payload, Mapping):
                items = payload.get("symbols")
                if isinstance(items, list):
                    symbol_entries.extend(
                        [item for item in items if isinstance(item, Mapping)]
                    )
    else:
        payload = session.get(
            url,
            endpoint=endpoint_key,
            budget="exchangeInfo",
            tokens=10.0,
        )
        if not isinstance(payload, Mapping):
            raise RuntimeError("Unexpected exchangeInfo response")
        items = payload.get("symbols")
        if isinstance(items, list):
            symbol_entries = [item for item in items if isinstance(item, Mapping)]

        # When the full snapshot was fetched chunking degenerates to a single request.
        exchange_chunks_total = 1

    if not symbol_entries:
        raise RuntimeError("No exchangeInfo symbols returned")

    by_symbol: Dict[str, Dict[str, float]] = {}
    for s in symbol_entries:
        sym = str(s.get("symbol", "")).upper()
        if target_symbols and sym not in target_symbols:
            continue
        tick_size = 0.0
        step_size = 0.0
        min_notional = 0.0
        for f in s.get("filters", []):
            typ = str(f.get("filterType", ""))
            if typ == "PRICE_FILTER":
                tick_size = float(f.get("tickSize", 0.0))
            elif typ == "LOT_SIZE":
                step_size = float(f.get("stepSize", 0.0))
            elif typ in ("MIN_NOTIONAL", "NOTIONAL"):
                min_notional = float(
                    f.get("minNotional", f.get("notional", 0.0))
                )
        if not sym:
            continue
        by_symbol[sym] = {
            "tickSize": tick_size,
            "stepSize": step_size,
            "minNotional": min_notional,
        }

    if target_symbols:
        symbols_order = [sym for sym in target_symbols if sym in by_symbol]
    else:
        symbols_order = list(by_symbol.keys())
    avg_quote_vol: Dict[str, float] = {}

    start_index = 0
    if isinstance(checkpoint, dict) and symbols_order:
        saved_order = checkpoint.get("order") or checkpoint.get("symbols")
        if isinstance(saved_order, list):
            normalized = [str(s).upper() for s in saved_order if str(s).strip()]
            if set(normalized) == set(symbols_order):
                symbols_order = normalized
        saved_pos = checkpoint.get("position")
        if isinstance(saved_pos, (int, float)):
            start_index = int(saved_pos)
        saved_vol = checkpoint.get("avg_quote_vol")
        if isinstance(saved_vol, dict):
            for key, value in saved_vol.items():
                sym = str(key).upper()
                if sym in by_symbol:
                    try:
                        avg_quote_vol[sym] = float(value)
                    except (TypeError, ValueError):
                        continue
        start_index = max(0, min(start_index, len(symbols_order)))
        if start_index > 0:
            message = (
                f"Resuming from checkpoint at position {start_index}/{len(symbols_order)}"
            )
            if start_index < len(symbols_order):
                message += f" (next={symbols_order[start_index]})"
            print(message)
    elif shuffle and symbols_order:
        rng = random.Random()
        rng.shuffle(symbols_order)

    end_ms = int(time.time() * 1000)
    window_ms = max(1, int(days)) * 86_400_000
    start_ms = end_ms - window_ms
    limit = max(1, min(int(days), 1500))

    chunk_windows = list(iter_time_chunks(start_ms, end_ms, chunk_days=30))
    if not chunk_windows:
        chunk_windows = [(start_ms, end_ms)]

    handled_signals: dict[int, Any] = {}
    checkpoint_payload: Dict[str, Any] | None = None
    state: Dict[str, Any] = {
        "position": start_index,
        "last_symbol": symbols_order[start_index - 1] if start_index > 0 else None,
    }

    batch_pref = int(getattr(session, "batch_size", 0) or 0)
    worker_pref = int(getattr(session, "max_workers", 0) or 0)
    batch_size = max(1, batch_pref or worker_pref or 1)
    total_symbols = len(symbols_order)
    batches_total = max(1, math.ceil(total_symbols / batch_size)) if total_symbols else 1

    def _save_checkpoint(position: int, *, last_symbol: str | None, completed: bool = False) -> None:
        nonlocal checkpoint_payload
        state["position"] = position
        state["last_symbol"] = last_symbol
        done_pct = 100.0 if completed else (100.0 * position / total_symbols if total_symbols else 0.0)
        batches_completed = 0
        if total_symbols:
            batches_completed = min(batches_total, math.ceil(position / batch_size))
        if completed:
            batches_completed = batches_total
        payload: Dict[str, Any] = {
            "order": symbols_order,
            "position": position,
            "avg_quote_vol": {k: float(v) for k, v in avg_quote_vol.items()},
            "chunks": {
                "completed": batches_completed,
                "total": batches_total,
                "size": batch_size,
            },
            "exchange_info": {
                "chunk_size": chunk_size_val or len(symbols_order) or 0,
                "chunks_total": exchange_chunks_total,
            },
            "done_pct": done_pct,
            "last_symbol": last_symbol,
        }
        if completed:
            payload["completed"] = True
        checkpoint_payload = payload
        session.save_checkpoint(
            payload,
            last_symbol=last_symbol,
            last_range=(start_ms, end_ms),
            progress_pct=done_pct,
        )
        if checkpoint_listener is not None:
            try:
                checkpoint_listener(dict(payload))
            except Exception:  # pragma: no cover - best effort notification
                pass

    def _handle_signal(signum: int, frame: Any | None) -> None:  # pragma: no cover - signal handler
        _save_checkpoint(state["position"], last_symbol=state.get("last_symbol"))
        if signum == getattr(signal, "SIGINT", None):
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    _save_checkpoint(start_index, last_symbol=state.get("last_symbol"), completed=total_symbols == 0)

    if install_signal_handlers:
        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                handled_signals[sig] = signal.getsignal(sig)
                signal.signal(sig, _handle_signal)
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                handled_signals.pop(sig, None)

    def _fetch_symbol_volume(symbol: str) -> float:
        sym = str(symbol).upper()
        seen_opens: set[int] = set()
        volumes: list[float] = []
        for chunk_start, chunk_end in chunk_windows:
            params = {
                "symbol": sym,
                "interval": "1d",
                "startTime": chunk_start,
                "endTime": chunk_end,
                "limit": limit,
            }
            payload = session.get(
                _klines_endpoint(market),
                params=params,
                endpoint=_klines_endpoint_key(market),
                budget="klines",
                tokens=_estimate_kline_tokens(limit),
            )
            if not isinstance(payload, list):
                continue
            for item in payload:
                if not isinstance(item, (list, tuple)) or len(item) < 8:
                    continue
                try:
                    open_ts = int(item[0])
                except (TypeError, ValueError):
                    continue
                if open_ts in seen_opens:
                    continue
                seen_opens.add(open_ts)
                try:
                    quote_volume = float(item[7])
                except (TypeError, ValueError):
                    continue
                volumes.append(quote_volume)
        return sum(volumes) / len(volumes) if volumes else 0.0

    try:
        idx = start_index
        while idx < len(symbols_order):
            batch = symbols_order[idx : idx + batch_size]
            futures: list[tuple[int, str, Any]] = []
            for offset, sym in enumerate(batch):
                absolute = idx + offset
                future = session.submit(_fetch_symbol_volume, sym)
                futures.append((absolute, sym, future))
            last_symbol = None
            last_position = idx
            for absolute, sym, future in futures:
                try:
                    avg_quote_vol[sym] = float(future.result())
                except Exception:
                    avg_quote_vol[sym] = 0.0
                last_symbol = sym
                last_position = absolute + 1
            _save_checkpoint(
                last_position,
                last_symbol=last_symbol,
                completed=last_position >= total_symbols,
            )
            idx += max(len(batch), 1)
    finally:
        if install_signal_handlers:
            for sig, handler in handled_signals.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):  # pragma: no cover - platform dependent
                    pass

    _save_checkpoint(len(symbols_order), last_symbol=symbols_order[-1] if symbols_order else None, completed=True)

    if volume_threshold > 0.0:
        before = len(by_symbol)
        by_symbol = {
            sym: spec
            for sym, spec in by_symbol.items()
            if avg_quote_vol.get(sym, 0.0) >= volume_threshold
        }
        dropped = before - len(by_symbol)
        print(
            f"Dropped {dropped} symbols below volume threshold {volume_threshold}"
        )

    if volume_out:
        _ensure_dir(volume_out)
        _write_json_atomic(volume_out, avg_quote_vol)

    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_dataset": f"binance_exchangeInfo_{market}",
        "version": 1,
    }
    payload = {"metadata": meta, "specs": by_symbol}

    _ensure_dir(out)
    _write_json_atomic(out, payload)
    print(f"Saved {len(by_symbol)} symbols to {out}")
    return by_symbol


__all__ = ["run"]
