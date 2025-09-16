#!/usr/bin/env python3
"""Fetch Binance exchange filters and store them as JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Sequence

import yaml

from binance_public import BinancePublicClient
from services.rest_budget import RestBudgetSession

DEFAULT_CHUNK_SIZE = 100
EXCHANGE_INFO_ENDPOINT = "GET /api/v3/exchangeInfo"


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Binance spot exchange filters and save them as JSON",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Destination JSON file path",
    )
    parser.add_argument(
        "--universe",
        action="store_true",
        help="Load symbols from services.universe.get_symbols()",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan requests without performing HTTP calls",
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Symbols to include (defaults to all when omitted)",
    )
    return parser.parse_args(argv)


def _default_offline_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "offline.yaml"


def _load_offline_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[WARN] offline config not found at {path}, using defaults", file=sys.stderr)
        return {}, {}
    except Exception as exc:  # pragma: no cover - defensive warning
        print(f"[WARN] failed to load offline config {path}: {exc}", file=sys.stderr)
        return {}, {}

    if not isinstance(payload, Mapping):
        print(
            f"[WARN] offline config {path} must be a mapping, got {type(payload).__name__}",
            file=sys.stderr,
        )
        return {}, {}

    rest_cfg_raw = payload.get("rest_budget", {})
    if not isinstance(rest_cfg_raw, Mapping):
        rest_cfg = {}
    else:
        rest_cfg = dict(rest_cfg_raw)

    script_cfg_raw = payload.get("fetch_binance_filters", {})
    if not isinstance(script_cfg_raw, Mapping):
        script_cfg = {}
    else:
        script_cfg = dict(script_cfg_raw)

    return rest_cfg, script_cfg


def _coerce_positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _resolve_chunk_size(config: Mapping[str, Any]) -> int:
    candidates: list[Any] = []
    if "chunk_size" in config:
        candidates.append(config["chunk_size"])
    if "max_symbols_per_request" in config:
        candidates.append(config["max_symbols_per_request"])
    chunk_cfg = config.get("chunk")
    if isinstance(chunk_cfg, Mapping):
        candidates.extend(
            [
                chunk_cfg.get("size"),
                chunk_cfg.get("chunk_size"),
                chunk_cfg.get("max_symbols"),
            ]
        )
    for value in candidates:
        number = _coerce_positive_int(value)
        if number is not None:
            return number
    return DEFAULT_CHUNK_SIZE


def _resolve_checkpoint_threshold(config: Mapping[str, Any], default: int) -> int:
    candidates: list[Any] = []
    if "checkpoint_min_symbols" in config:
        candidates.append(config["checkpoint_min_symbols"])
    checkpoint_cfg = config.get("checkpoint")
    if isinstance(checkpoint_cfg, Mapping):
        candidates.extend(
            [
                checkpoint_cfg.get("min_symbols"),
                checkpoint_cfg.get("min_size"),
                checkpoint_cfg.get("threshold"),
            ]
        )
    for value in candidates:
        number = _coerce_positive_int(value)
        if number is not None:
            return number
    return max(default, 1)


def _normalize_symbols(raw: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for symbol in raw:
        if not symbol:
            continue
        sym = symbol.strip().upper()
        if sym:
            cleaned.append(sym)
    return list(dict.fromkeys(cleaned))


def _load_symbols(args: argparse.Namespace) -> List[str]:
    symbols: List[str] = []
    if args.universe:
        from services.universe import get_symbols as get_universe_symbols

        symbols.extend(get_universe_symbols())
    if args.symbols:
        symbols.extend(args.symbols)
    return _normalize_symbols(symbols)


def _ensure_directory(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _write_json_atomic(path: str, payload: dict) -> None:
    _ensure_directory(path)
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".binance_filters_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(payload, tmp_file, ensure_ascii=False, indent=2, sort_keys=True)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _build_metadata(filters: Mapping[str, Any]) -> dict:
    return {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": "/api/v3/exchangeInfo",
        "symbols_count": len(filters),
    }


def _iter_symbol_chunks(symbols: Sequence[str], chunk_size: int) -> Iterator[list[str]]:
    step = max(1, int(chunk_size))
    for idx in range(0, len(symbols), step):
        yield list(symbols[idx : idx + step])


def _restore_checkpoint(
    session: RestBudgetSession, symbols: List[str], *, enable: bool
) -> tuple[int, dict[str, dict[str, Any]]]:
    if not enable or not symbols:
        return 0, {}

    checkpoint = session.load_checkpoint()
    if not isinstance(checkpoint, Mapping):
        return 0, {}

    saved_order = checkpoint.get("symbols") or checkpoint.get("order")
    if isinstance(saved_order, Iterable):
        normalized_order = _normalize_symbols(saved_order)
        if normalized_order and normalized_order != symbols:
            return 0, {}
    position_raw = checkpoint.get("position")
    try:
        position = int(position_raw)
    except (TypeError, ValueError):
        position = 0
    position = max(0, min(position, len(symbols)))
    if position >= len(symbols):
        return 0, {}

    raw_filters = checkpoint.get("filters")
    restored: dict[str, dict[str, Any]] = {}
    if isinstance(raw_filters, Mapping):
        for key, value in raw_filters.items():
            sym = str(key).strip().upper()
            if not sym or sym not in symbols:
                continue
            if isinstance(value, Mapping):
                restored[sym] = dict(value)
    if not restored:
        return 0, {}

    contiguous = 0
    for sym in symbols:
        if sym in restored:
            contiguous += 1
        else:
            break
    position = max(0, min(position, contiguous))
    return position, restored


def _save_checkpoint(
    session: RestBudgetSession,
    symbols: Sequence[str],
    position: int,
    filters: Mapping[str, Mapping[str, Any]],
    chunk_size: int,
    *,
    completed: bool = False,
) -> None:
    if not symbols:
        return
    limit = max(0, min(int(position), len(symbols)))
    payload: dict[str, Any] = {
        "symbols": list(symbols),
        "position": limit,
        "chunk_size": int(chunk_size),
    }
    if completed:
        payload["completed"] = True
        payload["filters"] = {}
    else:
        stored: dict[str, dict[str, Any]] = {}
        for sym in symbols[:limit]:
            data = filters.get(sym)
            if isinstance(data, Mapping):
                stored[sym] = dict(data)
        payload["filters"] = stored
    session.save_checkpoint(payload)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        symbols = _load_symbols(args)
        config_path = _default_offline_config_path()
        rest_cfg, script_cfg = _load_offline_config(config_path)
        chunk_size = _resolve_chunk_size(script_cfg)
        checkpoint_threshold = _resolve_checkpoint_threshold(script_cfg, chunk_size)

        with RestBudgetSession(rest_cfg) as session:
            client = BinancePublicClient(session=session)
            try:
                symbol_count = len(symbols)
                if symbol_count == 0:
                    chunk_count = 1
                else:
                    chunk_count = (symbol_count + chunk_size - 1) // chunk_size
                session.plan_request(EXCHANGE_INFO_ENDPOINT, count=chunk_count, tokens=1.0)

                if args.dry_run:
                    target = "all available" if symbol_count == 0 else f"{symbol_count}"
                    print(
                        "Dry run: would perform "
                        f"{chunk_count} request(s) to {EXCHANGE_INFO_ENDPOINT} "
                        f"for {target} symbol(s) with chunk_size={chunk_size}",
                    )
                    print(
                        json.dumps(
                            session.stats(), ensure_ascii=False, indent=2, sort_keys=True
                        )
                    )
                    return 0

                should_checkpoint = (
                    symbol_count > 0
                    and chunk_count > 1
                    and symbol_count >= checkpoint_threshold
                )

                filters: dict[str, dict[str, Any]] = {}
                start_index = 0
                if should_checkpoint:
                    start_index, restored = _restore_checkpoint(
                        session, symbols, enable=should_checkpoint
                    )
                    if start_index > 0:
                        print(
                            f"Resuming from symbol index {start_index}",
                            file=sys.stderr,
                        )
                    filters.update(restored)
                    _save_checkpoint(
                        session,
                        symbols,
                        start_index,
                        filters,
                        chunk_size,
                    )

                if symbol_count == 0:
                    filters = client.get_exchange_filters(market="spot", symbols=None)
                else:
                    index = start_index
                    for chunk in _iter_symbol_chunks(symbols[index:], chunk_size):
                        if not chunk:
                            continue
                        chunk_filters = client.get_exchange_filters(
                            market="spot", symbols=chunk
                        )
                        filters.update(chunk_filters)
                        index += len(chunk)
                        if should_checkpoint:
                            _save_checkpoint(
                                session,
                                symbols,
                                index,
                                filters,
                                chunk_size,
                            )
                    if should_checkpoint:
                        _save_checkpoint(
                            session,
                            symbols,
                            len(symbols),
                            filters,
                            chunk_size,
                            completed=True,
                        )

                metadata = _build_metadata(filters)
                payload = {"metadata": metadata, "filters": filters}
                _write_json_atomic(args.out, payload)
                print(
                    f"Fetched {metadata['symbols_count']} symbol filters "
                    f"from {metadata['source']} into {args.out}"
                )
                print(
                    json.dumps(
                        session.stats(), ensure_ascii=False, indent=2, sort_keys=True
                    )
                )
                return 0
            finally:
                client.close()
    except Exception as exc:  # pragma: no cover - CLI error handling
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
