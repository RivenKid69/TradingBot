#!/usr/bin/env python3
"""Fetch Binance exchange filters and store them as JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, MutableMapping, Sequence

import yaml

from binance_public import BinancePublicClient
from offline_config import normalize_dataset_splits
from services.rest_budget import RestBudgetSession

DEFAULT_CHUNK_SIZE = 100
EXCHANGE_INFO_ENDPOINT = "GET /api/v3/exchangeInfo"
DEFAULT_UNIVERSE_PATH = Path(__file__).resolve().parents[1] / "data" / "universe" / "symbols.json"


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Binance spot exchange filters and save them as JSON",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Destination JSON file path",
    )
    parser.add_argument(
        "--universe",
        nargs="?",
        type=Path,
        const=DEFAULT_UNIVERSE_PATH,
        default=DEFAULT_UNIVERSE_PATH,
        metavar="PATH",
        help=(
            "Path to a JSON file containing universe symbols (default: "
            f"{DEFAULT_UNIVERSE_PATH}). Use --universe without PATH to use the "
            "default path, or pass '-' to disable loading a universe file."
        ),
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
    parser.add_argument(
        "--config",
        default=str(_default_offline_config_path()),
        help="Path to offline YAML config containing rest_budget settings.",
    )
    return parser.parse_args(argv)


def _default_offline_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "offline.yaml"


STATS_PATH = Path("logs/offline/fetch_binance_filters_stats.json")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        key: (dict(value) if isinstance(value, Mapping) else value)
        for key, value in base.items()
    }
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = dict(value) if isinstance(value, Mapping) else value
    return merged


def _normalize_endpoint_map(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, Mapping):
            normalized[str(key)] = dict(value)
        else:
            normalized[str(key)] = value
    return normalized


def _normalize_rest_budget_sections(
    rest_cfg: Mapping[str, Any]
) -> tuple[dict[str, Any], Any]:
    sources: list[Mapping[str, Any]] = []
    if isinstance(rest_cfg, Mapping):
        sources.append(rest_cfg)
        for key in ("session", "config"):
            nested = rest_cfg.get(key)
            if isinstance(nested, Mapping):
                sources.append(nested)

    combined: dict[str, Any] = {}
    for source in sources:
        combined = _deep_merge(combined, source)

    session_cfg: dict[str, Any] = {}
    limits_cfg = combined.get("limits")
    if isinstance(limits_cfg, Mapping):
        global_cfg: MutableMapping[str, Any] | None = None
        for alias in ("global", "global_", "default", "defaults"):
            candidate = limits_cfg.get(alias)
            if isinstance(candidate, Mapping):
                global_cfg = dict(candidate)
                break
        if global_cfg is None:
            scalars = {
                key: value for key, value in limits_cfg.items() if not isinstance(value, Mapping)
            }
            if scalars:
                global_cfg = dict(scalars)
        if global_cfg:
            session_cfg["global"] = global_cfg
        endpoints_cfg = limits_cfg.get("endpoints")
        if isinstance(endpoints_cfg, Mapping):
            session_cfg["endpoints"] = _normalize_endpoint_map(endpoints_cfg)

    if "global" not in session_cfg:
        for alias in ("global", "global_", "default_global"):
            candidate = combined.get(alias)
            if isinstance(candidate, Mapping):
                session_cfg["global"] = dict(candidate)
                break

    if "endpoints" not in session_cfg:
        endpoints_candidate = combined.get("endpoints")
        if isinstance(endpoints_candidate, Mapping):
            session_cfg["endpoints"] = _normalize_endpoint_map(endpoints_candidate)

    concurrency_cfg = combined.get("concurrency")
    if isinstance(concurrency_cfg, Mapping):
        session_cfg["concurrency"] = dict(concurrency_cfg)
        for alias in ("batch_size", "queue", "max_in_flight"):
            candidate = concurrency_cfg.get(alias)
            if candidate is not None:
                session_cfg.setdefault("batch_size", candidate)
                break

    batch_candidate = combined.get("batch_size")
    if batch_candidate is not None:
        session_cfg.setdefault("batch_size", batch_candidate)

    for key in (
        "enabled",
        "cache",
        "checkpoint",
        "retry",
        "dynamic_from_headers",
        "timeout",
        "timeout_s",
        "cooldown_s",
        "cooldown_sec",
        "jitter",
        "jitter_ms",
        "cache_mode",
        "cache_dir",
        "cache_ttl_days",
        "cache_controls",
    ):
        value = combined.get(key)
        if value is None:
            continue
        session_cfg[key] = dict(value) if isinstance(value, Mapping) else value

    shuffle_cfg: Any = None
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in ("shuffle", "shuffle_symbols", "shuffleOptions"):
            if key in source:
                shuffle_cfg = source[key]
                break
        if shuffle_cfg is not None:
            break

    return session_cfg, shuffle_cfg


def _load_offline_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
        shuffle_cfg: Any = None
    else:
        rest_cfg, shuffle_cfg = _normalize_rest_budget_sections(rest_cfg_raw)
        if shuffle_cfg is not None:
            rest_cfg.setdefault("shuffle", shuffle_cfg)

    script_cfg_raw = payload.get("fetch_binance_filters", {})
    if not isinstance(script_cfg_raw, Mapping):
        script_cfg = {}
    else:
        script_cfg = dict(script_cfg_raw)

    datasets_raw = payload.get("datasets")
    if not isinstance(datasets_raw, Mapping):
        datasets_raw = payload.get("dataset_splits")

    dataset_splits = normalize_dataset_splits(datasets_raw)

    return rest_cfg, script_cfg, dataset_splits


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


def _load_universe_symbols(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    candidates: Sequence[Any]
    if isinstance(payload, Mapping):
        symbols = payload.get("symbols")
        if not isinstance(symbols, Sequence) or isinstance(symbols, (str, bytes)):
            raise ValueError(
                f"Universe file {path} must contain a sequence under 'symbols'"
            )
        candidates = symbols
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        candidates = payload
    else:
        raise ValueError(f"Universe file {path} must contain a JSON array of symbols")
    return _normalize_symbols(str(item) for item in candidates)


def _load_symbols(args: argparse.Namespace) -> List[str]:
    symbols: List[str] = []
    if args.universe and str(args.universe) != "-":
        symbols.extend(_load_universe_symbols(args.universe))
    if args.symbols:
        symbols.extend(args.symbols)
    return _normalize_symbols(symbols)


def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_directory(path)
    directory = path.parent
    fd, tmp_path = tempfile.mkstemp(prefix=".binance_filters_", dir=str(directory))
    tmp_path_obj = Path(tmp_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(payload, tmp_file, ensure_ascii=False, indent=2, sort_keys=True)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path_obj, path)
    finally:
        try:
            os.unlink(tmp_path_obj)
        except FileNotFoundError:
            pass


def _build_metadata(filters: Mapping[str, Any]) -> dict:
    return {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": EXCHANGE_INFO_ENDPOINT,
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
    last_symbol = symbols[limit - 1] if limit > 0 else None
    progress_pct: float | None = None
    if symbols:
        progress_pct = float(limit) / float(len(symbols)) * 100.0
    session.save_checkpoint(payload, last_symbol=last_symbol, progress_pct=progress_pct)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    symbols = _load_symbols(args)
    config_path = Path(args.config or _default_offline_config_path())
    rest_cfg, script_cfg, _ = _load_offline_config(config_path)
    chunk_size = _resolve_chunk_size(script_cfg)
    checkpoint_threshold = _resolve_checkpoint_threshold(script_cfg, chunk_size)

    stats_path = STATS_PATH
    with RestBudgetSession(rest_cfg) as session:
        try:
            session.write_stats(stats_path)
        except Exception:
            logging.debug("Failed to write initial stats snapshot", exc_info=True)
        with closing(BinancePublicClient(session=session)) as client:
            symbol_count = len(symbols)
            chunk_count = 1 if symbol_count == 0 else (symbol_count + chunk_size - 1) // chunk_size
            session.plan_request(
                EXCHANGE_INFO_ENDPOINT, count=chunk_count, tokens=10.0
            )

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
                try:
                    session.write_stats(stats_path)
                except Exception:
                    logging.debug("Failed to persist stats snapshot", exc_info=True)
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
                try:
                    session.write_stats(stats_path)
                except Exception:
                    logging.debug("Failed to persist stats snapshot", exc_info=True)

            if symbol_count == 0:
                filters = client.get_exchange_filters(market="spot", symbols=None)
                if not filters:
                    raise RuntimeError("No filters returned from Binance exchangeInfo")
                try:
                    session.write_stats(stats_path)
                except Exception:
                    logging.debug("Failed to persist stats snapshot", exc_info=True)
            else:
                index = start_index
                for chunk in _iter_symbol_chunks(symbols[index:], chunk_size):
                    if not chunk:
                        continue
                    chunk_filters = client.get_exchange_filters(
                        market="spot", symbols=chunk
                    )
                    missing = [sym for sym in chunk if sym not in chunk_filters]
                    if missing:
                        raise RuntimeError(
                            "Missing filters for symbol(s): " + ", ".join(missing)
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
                        try:
                            session.write_stats(stats_path)
                        except Exception:
                            logging.debug(
                                "Failed to persist stats snapshot", exc_info=True
                            )
                if len(filters) < symbol_count:
                    missing_total = [sym for sym in symbols if sym not in filters]
                    if missing_total:
                        raise RuntimeError(
                            "Missing filters after completion for symbol(s): "
                            + ", ".join(missing_total)
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
                    try:
                        session.write_stats(stats_path)
                    except Exception:
                        logging.debug("Failed to persist stats snapshot", exc_info=True)

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
            try:
                session.write_stats(stats_path)
            except Exception:
                logging.debug("Failed to persist stats snapshot", exc_info=True)
            return 0


if __name__ == "__main__":
    sys.exit(main())
