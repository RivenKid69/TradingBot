"""CLI for building ADV OHLCV snapshots using Binance public data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import yaml

from build_adv_base import BuildAdvConfig, build_adv
from offline_config import normalize_dataset_splits
from services.rest_budget import RestBudgetSession
from utils_time import parse_time_to_ms


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
    try:
        payload = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
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
        if isinstance(data, dict):
            return _normalize_symbols(data.keys())
        return []
    lines = [line.strip() for line in text.replace(",", "\n").splitlines()]
    return _normalize_symbols(line for line in lines if line)


def _default_symbols() -> list[str]:
    default_path = Path("data/universe/symbols.json")
    try:
        with default_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []
    if isinstance(data, Sequence):
        return _normalize_symbols(data)
    if isinstance(data, dict):
        return _normalize_symbols(data.keys())
    return []


def _resolve_symbols(symbols_arg: str, symbols_file: str) -> list[str]:
    direct = _normalize_symbols(symbols_arg.split(",") if symbols_arg else [])
    if direct:
        return direct
    file_symbols = _load_symbols_file(symbols_file)
    if file_symbols:
        return file_symbols
    return _default_symbols()


def _default_offline_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "offline.yaml"


STATS_PATH = Path("logs/offline/build_adv_stats.json")


def _load_offline_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        print(
            f"[WARN] offline config not found: {path}",
            file=sys.stderr,
        )
        return {}
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] failed to load offline config {path}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(payload, Mapping):
        print(
            f"[WARN] offline config {path} must be a mapping, got {type(payload).__name__}",
            file=sys.stderr,
        )
        return {}
    normalized: dict[str, Any] = dict(payload)
    datasets_raw = payload.get("datasets")
    if not isinstance(datasets_raw, Mapping):
        datasets_raw = payload.get("dataset_splits")
    normalized["datasets"] = normalize_dataset_splits(datasets_raw)
    return normalized


def _merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {key: value for key, value in base.items()}
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _merge_mappings(
                merged[key],
                value,
            )
        else:
            merged[key] = value
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


def _normalize_rest_budget_config(rest_cfg_raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(rest_cfg_raw, Mapping):
        return {}

    sources: list[Mapping[str, Any]] = [rest_cfg_raw]
    for key in ("session", "config"):
        nested = rest_cfg_raw.get(key)
        if isinstance(nested, Mapping):
            sources.append(nested)

    combined: dict[str, Any] = {}
    for source in sources:
        combined = _merge_mappings(combined, dict(source))

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

    return session_cfg


def _first_non_empty_str(*candidates: Any) -> str | None:
    for value in candidates:
        if isinstance(value, (str, Path)):
            text = str(value).strip()
            if text:
                return text
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_rest_config(path: str) -> dict[str, Any]:
    path = path.strip()
    if not path:
        return {}
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        print(f"[WARN] rest budget config not found: {path}", file=sys.stderr)
        return {}
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] failed to load rest budget config {path}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        return {}
    return dict(data)


def _parse_time(value: str, name: str) -> int:
    try:
        return parse_time_to_ms(value)
    except Exception as exc:  # pragma: no cover - validation
        raise SystemExit(f"Invalid {name}: {value!r} ({exc})")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Binance OHLCV history and build ADV dataset parquet.",
    )
    parser.add_argument("--market", choices=["spot", "futures"], default="futures")
    parser.add_argument("--interval", default="1h", help="Kline interval (e.g. 1h,4h,1d)")
    parser.add_argument("--start", required=True, help="Start of history (ISO8601 or unix ms)")
    parser.add_argument("--end", required=True, help="End of history (ISO8601 or unix ms)")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol list")
    parser.add_argument(
        "--symbols-file",
        default="",
        help="Optional path to JSON/TXT with symbols; fallback to data/universe/symbols.json",
    )
    parser.add_argument("--out", default="data/adv/klines.parquet", help="Destination dataset path")
    parser.add_argument(
        "--cache-dir",
        default="data/adv/cache",
        help="Directory for per-symbol parquet cache",
    )
    parser.add_argument(
        "--config",
        default=str(_default_offline_config_path()),
        help="Path to offline YAML config containing rest_budget settings.",
    )
    parser.add_argument("--limit", type=int, default=1500, help="Maximum bars per request")
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=30,
        help="Chunk size in days for planning fetch windows",
    )
    parser.add_argument(
        "--rest-budget-config",
        default="configs/rest_budget.yaml",
        help="Path to RestBudgetSession YAML configuration",
    )
    parser.add_argument(
        "--cache-mode",
        default=None,
        help="RestBudgetSession cache mode override (off/read/read_write)",
    )
    parser.add_argument(
        "--cache-ttl",
        type=float,
        default=None,
        help="RestBudgetSession cache TTL in days",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Override checkpoint path (defaults to <cache-dir>/checkpoint.json)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume from checkpoint if present",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume_from_checkpoint",
        action="store_false",
        help="Ignore checkpoint even if present",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan fetch ranges without performing HTTP requests",
    )
    parser.set_defaults(resume_from_checkpoint=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    start_ms = _parse_time(args.start, "start")
    end_ms = _parse_time(args.end, "end")
    if end_ms <= start_ms:
        raise SystemExit("end must be greater than start")

    symbols = _resolve_symbols(args.symbols, args.symbols_file)
    if not symbols:
        raise SystemExit("No symbols resolved; provide --symbols or --symbols-file")

    out_path = Path(args.out)
    cache_dir = Path(args.cache_dir)

    limit = max(1, int(args.limit))
    chunk_days = max(1, int(args.chunk_days))

    offline_config = _load_offline_config(Path(args.config or _default_offline_config_path()))
    offline_rest_cfg = _normalize_rest_budget_config(
        offline_config.get("rest_budget") if isinstance(offline_config, Mapping) else None
    )

    rest_cfg_loaded = _normalize_rest_budget_config(_load_rest_config(str(args.rest_budget_config)))
    rest_cfg: dict[str, Any]
    if offline_rest_cfg:
        rest_cfg = _merge_mappings(offline_rest_cfg, rest_cfg_loaded)
    else:
        rest_cfg = dict(rest_cfg_loaded)

    cache_cfg_raw = rest_cfg.get("cache") if isinstance(rest_cfg, Mapping) else None
    cache_cfg = dict(cache_cfg_raw) if isinstance(cache_cfg_raw, Mapping) else {}

    http_cache_dir = _first_non_empty_str(
        cache_cfg.get("dir"),
        cache_cfg.get("path"),
        cache_cfg.get("cache_dir"),
        rest_cfg.get("cache_dir") if isinstance(rest_cfg, Mapping) else None,
    )
    if not http_cache_dir:
        http_cache_dir = str(cache_dir)

    if args.cache_mode:
        cache_mode = str(args.cache_mode)
    else:
        cache_mode = _first_non_empty_str(
            cache_cfg.get("mode"),
            cache_cfg.get("cache_mode"),
            rest_cfg.get("cache_mode") if isinstance(rest_cfg, Mapping) else None,
        )

    if args.cache_ttl is not None:
        cache_ttl = float(args.cache_ttl)
    else:
        cache_ttl = _coerce_float(cache_cfg.get("ttl_days"))
        if cache_ttl is None:
            cache_ttl = _coerce_float(cache_cfg.get("ttl"))

    checkpoint_cfg_raw = rest_cfg.get("checkpoint") if isinstance(rest_cfg, Mapping) else None
    checkpoint_cfg = dict(checkpoint_cfg_raw) if isinstance(checkpoint_cfg_raw, Mapping) else {}

    checkpoint_path_override = args.checkpoint_path.strip()
    checkpoint_path = _first_non_empty_str(
        checkpoint_path_override,
        checkpoint_cfg.get("path"),
        checkpoint_cfg.get("file"),
        checkpoint_cfg.get("checkpoint_path"),
    )
    if not checkpoint_path:
        checkpoint_path = str(cache_dir / "checkpoint.json")

    checkpoint_enabled_flag = checkpoint_cfg.get("enabled")
    if checkpoint_enabled_flag is None:
        checkpoint_enabled = bool(checkpoint_path)
    else:
        checkpoint_enabled = bool(checkpoint_enabled_flag) and bool(checkpoint_path)

    resume_flag: bool | None = args.resume_from_checkpoint
    if resume_flag is None:
        resume_flag = bool(checkpoint_enabled)
    else:
        resume_flag = bool(resume_flag)
    if resume_flag and not checkpoint_enabled:
        resume_flag = False

    config = BuildAdvConfig(
        market=args.market,
        interval=args.interval,
        start_ms=start_ms,
        end_ms=end_ms,
        out_path=out_path,
        cache_dir=cache_dir,
        limit=limit,
        chunk_days=chunk_days,
        resume_from_checkpoint=bool(resume_flag),
        dry_run=bool(args.dry_run),
    )

    stats_path = STATS_PATH
    with RestBudgetSession(
        rest_cfg,
        cache_dir=http_cache_dir,
        ttl_days=cache_ttl,
        mode=cache_mode,
        checkpoint_path=checkpoint_path,
        checkpoint_enabled=checkpoint_enabled,
        resume_from_checkpoint=bool(resume_flag),
    ) as session:
        if stats_path:
            try:
                session.write_stats(stats_path)
            except Exception:
                pass
        result = build_adv(session, symbols, config, stats_path=stats_path)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
