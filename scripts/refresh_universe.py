from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from services.rest_budget import RestBudgetSession


EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
EXCHANGE_INFO_ENDPOINT = "GET /api/v3/exchangeInfo"
TICKER_24HR_URL = "https://api.binance.com/api/v3/ticker/24hr"
TICKER_24HR_ENDPOINT = "GET /api/v3/ticker/24hr"


def _default_offline_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "offline.yaml"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh Binance USDT spot universe with optional liquidity filter.",
    )
    parser.add_argument(
        "--out",
        default="data/universe/symbols.json",
        help="Destination JSON file for the refreshed universe.",
    )
    parser.add_argument(
        "--liquidity-threshold",
        type=float,
        default=0.0,
        help="Minimum 24h quote volume (USDT) required to keep a symbol.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan requests without saving output or fetching tickers.",
    )
    return parser.parse_args(argv)


def _load_rest_budget_config(path: Path) -> tuple[dict[str, Any], Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.warning("offline config not found at %s, using defaults", path)
        return {}, None
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.warning("failed to load offline config %s: %s", path, exc)
        return {}, None

    if not isinstance(payload, Mapping):
        logging.warning("offline config %s must be a mapping, got %s", path, type(payload))
        return {}, None

    rest_cfg = payload.get("rest_budget", {})
    if not isinstance(rest_cfg, Mapping):
        logging.warning("offline config %s missing rest_budget mapping", path)
        return {}, None
    shuffle_cfg = rest_cfg.get("shuffle_symbols")
    return dict(rest_cfg), shuffle_cfg


def _ensure_directory(path: Path) -> None:
    directory = path.parent
    if directory and not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)


def _write_json_atomic(path: Path, payload: Sequence[str]) -> None:
    _ensure_directory(path)
    directory = path.parent if path.parent.as_posix() else Path(".")
    fd, tmp_path = tempfile.mkstemp(prefix=".refresh_universe_", dir=str(directory))
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


def _extract_symbols(exchange_info: Mapping[str, Any]) -> list[str]:
    symbols: list[str] = []
    for raw in exchange_info.get("symbols", []):
        if not isinstance(raw, Mapping):
            continue
        status = str(raw.get("status", "")).upper()
        quote = str(raw.get("quoteAsset", "")).upper()
        permissions = [str(p).upper() for p in raw.get("permissions", []) if str(p)]
        if (
            status == "TRADING"
            and quote == "USDT"
            and ("SPOT" in permissions or raw.get("isSpotTradingAllowed"))
        ):
            symbol = str(raw.get("symbol", "")).strip().upper()
            if symbol:
                symbols.append(symbol)
    return symbols


def _normalize_order(raw: Iterable[Any], allowed: set[str]) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for item in raw:
        symbol = str(item).strip().upper()
        if not symbol or symbol not in allowed or symbol in seen:
            continue
        order.append(symbol)
        seen.add(symbol)
    return order


def _parse_shuffle_options(raw: Any) -> tuple[bool, int, int | None]:
    enabled = False
    min_size = 0
    seed: int | None = None

    if isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", True))
        threshold = raw.get("min_size", raw.get("min_symbols", raw.get("threshold", 0)))
        try:
            min_size = int(threshold)
        except (TypeError, ValueError):
            min_size = 0
        if "seed" in raw:
            try:
                seed = int(raw["seed"])
            except (TypeError, ValueError):
                seed = None
    elif isinstance(raw, bool):
        enabled = raw
    elif isinstance(raw, (int, float)):
        enabled = True
        min_size = int(raw)
    elif isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"true", "yes", "1"}:
            enabled = True
        elif text in {"false", "no", "0", ""}:
            enabled = False
        else:
            try:
                min_size = int(text)
                enabled = True
            except ValueError:
                enabled = False

    return enabled, max(min_size, 0), seed


def _log_request_stats(stats: Mapping[str, Any]) -> None:
    requests = stats.get("requests", {})
    tokens = stats.get("request_tokens", {})
    try:
        total_requests = sum(int(v) for v in requests.values())
    except Exception:  # pragma: no cover - defensive fallback
        total_requests = 0
    try:
        total_tokens = sum(float(v) for v in tokens.values())
    except Exception:  # pragma: no cover - defensive fallback
        total_tokens = 0.0
    logging.info(
        "HTTP usage: requests_total=%d weight_total=%.3f",
        total_requests,
        total_tokens,
    )


def _fetch_quote_volume(session: RestBudgetSession, symbol: str) -> float:
    payload = session.get(
        TICKER_24HR_URL,
        params={"symbol": symbol},
        endpoint=TICKER_24HR_ENDPOINT,
        budget="ticker24hr",
        tokens=1.0,
    )
    if isinstance(payload, Mapping):
        try:
            volume = float(payload.get("quoteVolume", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return volume if math.isfinite(volume) else 0.0
    return 0.0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config_path = _default_offline_config_path()
    rest_cfg, shuffle_cfg = _load_rest_budget_config(config_path)
    shuffle_enabled, shuffle_min_size, shuffle_seed = _parse_shuffle_options(shuffle_cfg)

    session_cfg = dict(rest_cfg)
    if "shuffle_symbols" in session_cfg:
        session_cfg = dict(session_cfg)
        session_cfg.pop("shuffle_symbols", None)

    liquidity_threshold = float(args.liquidity_threshold or 0.0)
    out_path = Path(args.out)

    with RestBudgetSession(session_cfg) as session:
        exchange_info = session.get(
            EXCHANGE_INFO_URL,
            endpoint=EXCHANGE_INFO_ENDPOINT,
            budget="exchangeInfo",
            tokens=10.0,
        )
        if not isinstance(exchange_info, Mapping):
            raise RuntimeError("Unexpected exchangeInfo response")

        symbols = _extract_symbols(exchange_info)
        symbol_set = set(symbols)
        if not symbols:
            logging.warning("No eligible symbols found in exchangeInfo response")

        order = list(symbols)
        volumes: dict[str, float] = {}
        start_index = 0
        checkpoint_data = session.load_checkpoint()
        resuming = False

        if isinstance(checkpoint_data, Mapping) and symbol_set:
            saved_order_raw = checkpoint_data.get("order") or checkpoint_data.get("symbols")
            if isinstance(saved_order_raw, Iterable):
                normalized_order = _normalize_order(saved_order_raw, symbol_set)
                if set(normalized_order) == symbol_set and len(normalized_order) == len(symbol_set):
                    order = normalized_order
                    resuming = True
            if resuming:
                saved_position = checkpoint_data.get("position")
                if isinstance(saved_position, (int, float)):
                    start_index = max(0, min(int(saved_position), len(order)))
                saved_volumes = checkpoint_data.get("volumes")
                if isinstance(saved_volumes, Mapping):
                    for key, value in saved_volumes.items():
                        sym = str(key).strip().upper()
                        if sym not in symbol_set:
                            continue
                        try:
                            volume = float(value)
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(volume):
                            volumes[sym] = volume
                if checkpoint_data.get("completed"):
                    logging.info("Checkpoint indicates previous run completed; starting fresh")
                    resuming = False
                    start_index = 0
                    volumes.clear()
                    order = list(symbols)
            else:
                if saved_order_raw:
                    logging.info("Ignoring checkpoint order that does not match current symbols")

        if not resuming and shuffle_enabled and len(order) >= max(1, shuffle_min_size):
            rng = random.Random()
            if shuffle_seed is not None:
                rng.seed(shuffle_seed)
            rng.shuffle(order)
            logging.info("Shuffled processing order for %d symbols", len(order))

        if liquidity_threshold <= 0.0:
            planned_ticker_requests = 0
        else:
            planned_ticker_requests = len(order)

        if args.dry_run:
            logging.info(
                "Dry run: would process %d symbols (liquidity_threshold=%.3f) with %d ticker requests",
                len(order),
                liquidity_threshold,
                planned_ticker_requests,
            )
            stats = session.stats()
            _log_request_stats(stats)
            return 0

        def _save_checkpoint(position: int, *, symbol: str | None = None, completed: bool = False) -> None:
            payload: dict[str, Any] = {
                "order": order,
                "position": position,
                "volumes": {k: float(v) for k, v in volumes.items()},
            }
            if symbol is not None:
                payload["current_symbol"] = symbol
            if completed:
                payload["completed"] = True
            session.save_checkpoint(payload)

        if liquidity_threshold > 0.0 and order:
            if start_index >= len(order):
                start_index = len(order)
            else:
                logging.info(
                    "Fetching 24h ticker volumes for %d symbols starting at position %d",
                    len(order),
                    start_index,
                )

            batch_pref = int(getattr(session, "batch_size", 0) or 0)
            worker_pref = int(getattr(session, "max_workers", 0) or 0)
            batch_size = max(1, batch_pref or worker_pref or 1)

            _save_checkpoint(start_index)

            idx = start_index
            while idx < len(order):
                batch = order[idx : idx + batch_size]
                futures: list[tuple[int, str, Any]] = []
                for offset, symbol in enumerate(batch):
                    absolute = idx + offset
                    if symbol in volumes and absolute < start_index:
                        continue
                    _save_checkpoint(absolute, symbol=symbol)
                    future = session.submit(_fetch_quote_volume, session, symbol)
                    futures.append((absolute, symbol, future))
                for absolute, symbol, future in futures:
                    try:
                        volume = float(future.result())
                    except Exception as exc:  # pragma: no cover - network dependent
                        logging.warning("Failed to fetch volume for %s: %s", symbol, exc)
                        volume = 0.0
                    if not math.isfinite(volume):
                        volume = 0.0
                    volumes[symbol] = volume
                    _save_checkpoint(absolute + 1, symbol=symbol)
                idx += max(len(batch), 1)

            _save_checkpoint(len(order), completed=True)

        eligible_symbols = [
            symbol
            for symbol in order
            if liquidity_threshold <= 0.0 or volumes.get(symbol, 0.0) >= liquidity_threshold
        ]
        eligible_symbols = sorted(dict.fromkeys(eligible_symbols))

        _write_json_atomic(out_path, eligible_symbols)
        logging.info(
            "Saved %d symbols to %s (threshold=%.3f)",
            len(eligible_symbols),
            out_path,
            liquidity_threshold,
        )

        stats = session.stats()
        _log_request_stats(stats)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
