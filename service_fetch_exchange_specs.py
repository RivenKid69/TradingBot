from __future__ import annotations

import json
import os
import random
import signal
import time
from datetime import datetime
from typing import Any, Dict, Sequence

from services.rest_budget import RestBudgetSession


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


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d:
        os.makedirs(d, exist_ok=True)


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
) -> Dict[str, Dict[str, float]]:
    """Fetch Binance exchangeInfo and store minimal specs JSON.

    Additionally computes average daily quote volume over the last ``days``
    for each symbol and optionally filters out symbols whose average falls
    below ``volume_threshold``.  The computed averages can be stored in
    ``volume_out`` for transparency.  When ``shuffle`` is true the symbol
    processing order is randomised (unless restored from checkpoint).
    ``session`` controls HTTP budgeting, caching and checkpoint persistence.
    """

    session = session or RestBudgetSession({})
    data = session.get(_endpoint(market), endpoint=_endpoint_key(market))
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected exchangeInfo response")

    if isinstance(symbols, str):
        requested = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        requested = [s.strip().upper() for s in (symbols or []) if s.strip()]

    by_symbol: Dict[str, Dict[str, float]] = {}
    for s in data.get("symbols", []):
        sym = str(s.get("symbol", "")).upper()
        if requested and sym not in requested:
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
        by_symbol[sym] = {
            "tickSize": tick_size,
            "stepSize": step_size,
            "minNotional": min_notional,
        }

    symbols_order = list(by_symbol.keys())
    avg_quote_vol: Dict[str, float] = {}

    checkpoint = session.load_checkpoint()
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

    handled_signals: dict[int, Any] = {}
    checkpoint_payload: Dict[str, Any] = {}

    def _update_checkpoint(position: int, *, symbol: str | None = None, completed: bool = False) -> None:
        nonlocal checkpoint_payload
        payload: Dict[str, Any] = {
            "order": symbols_order,
            "position": position,
            "avg_quote_vol": {k: float(v) for k, v in avg_quote_vol.items()},
        }
        if symbol is not None:
            payload["current_symbol"] = symbol
        if completed:
            payload["completed"] = True
        checkpoint_payload = payload
        session.save_checkpoint(checkpoint_payload)

    def _handle_signal(signum: int, frame: Any | None) -> None:  # pragma: no cover - signal handler
        session.save_checkpoint(checkpoint_payload)
        if signum == getattr(signal, "SIGINT", None):
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    _update_checkpoint(start_index)

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            handled_signals[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            handled_signals.pop(sig, None)

    end_ms = int(time.time() * 1000)
    window_ms = max(1, int(days)) * 86_400_000
    start_ms = end_ms - window_ms
    limit = max(1, min(int(days), 1500))

    try:
        for idx in range(start_index, len(symbols_order)):
            sym = symbols_order[idx]
            _update_checkpoint(idx, symbol=sym)
            params = {
                "symbol": sym,
                "interval": "1d",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            }
            try:
                kl = session.get(
                    _klines_endpoint(market),
                    params=params,
                    endpoint=_klines_endpoint_key(market),
                )
            except Exception:
                avg_quote_vol[sym] = 0.0
            else:
                vols: list[float] = []
                if isinstance(kl, list):
                    for item in kl:
                        try:
                            vols.append(float(item[7]))
                        except (IndexError, TypeError, ValueError):
                            continue
                avg_quote_vol[sym] = sum(vols) / len(vols) if vols else 0.0
            _update_checkpoint(idx + 1, symbol=sym)
    finally:
        for sig, handler in handled_signals.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                pass

    _update_checkpoint(len(symbols_order), completed=True)

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
        with open(volume_out, "w", encoding="utf-8") as f:
            json.dump(avg_quote_vol, f, ensure_ascii=False, indent=2)

    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_dataset": f"binance_exchangeInfo_{market}",
        "version": 1,
    }
    payload = {"metadata": meta, "specs": by_symbol}

    _ensure_dir(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(by_symbol)} symbols to {out}")
    return by_symbol


__all__ = ["run"]
