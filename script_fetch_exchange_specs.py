from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any, Iterable, Mapping, Tuple

from service_fetch_exchange_specs import _endpoint_key, _klines_endpoint_key, run
from services.rest_budget import RestBudgetSession, iter_time_chunks
import yaml


def _normalize_symbols(items: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for item in items:
        text = str(item).strip().upper()
        if text:
            result.append(text)
    return result


def _resolve_symbols_for_plan(
    args: argparse.Namespace, session: RestBudgetSession
) -> tuple[list[str], str]:
    direct = _normalize_symbols(str(args.symbols or "").split(","))
    if direct:
        return direct, "argument"

    checkpoint_data = session.load_checkpoint()
    if isinstance(checkpoint_data, dict):
        saved = checkpoint_data.get("order") or checkpoint_data.get("symbols")
        if isinstance(saved, list):
            normalized = _normalize_symbols(saved)
            if normalized:
                return normalized, "checkpoint"

    manual_ckpt = str(getattr(args, "checkpoint_path", "") or "").strip()
    if manual_ckpt:
        try:
            with open(manual_ckpt, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError:
            payload = None
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            saved = payload.get("order") or payload.get("symbols")
            if isinstance(saved, list):
                normalized = _normalize_symbols(saved)
                if normalized:
                    return normalized, "checkpoint_file"

    try:
        with open(args.out, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = None
    except (OSError, json.JSONDecodeError):
        existing = None
    if isinstance(existing, dict):
        specs = existing.get("specs")
        if isinstance(specs, dict) and specs:
            normalized = _normalize_symbols(specs.keys())
            if normalized:
                return normalized, "output"

    return [], "unknown"


def _plan_dry_run(
    args: argparse.Namespace,
    session: RestBudgetSession,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    symbols, source = _resolve_symbols_for_plan(args, session)
    now_ms = int(time.time() * 1000)
    window_ms = max(1, int(args.days)) * 86_400_000
    start_ms = now_ms - window_ms
    chunk_windows = list(iter_time_chunks(start_ms, now_ms, chunk_days=30))
    if not chunk_windows:
        chunk_windows = [(start_ms, now_ms)]
    chunk_count = len(chunk_windows)

    exchange_key = _endpoint_key(args.market)
    klines_key = _klines_endpoint_key(args.market)
    chunk_size = max(1, int(chunk_size) if chunk_size else 1)
    symbol_count = len(symbols)
    exchange_calls = 1 if symbol_count == 0 else (symbol_count + chunk_size - 1) // chunk_size
    session.plan_request(exchange_key, count=exchange_calls, tokens=10.0)

    klines_requests = len(symbols) * chunk_count
    if klines_requests > 0:
        session.plan_request(klines_key, count=klines_requests, tokens=1.0)

    plan = {
        "mode": "dry_run",
        "symbol_count": len(symbols),
        "symbol_source": source,
        "chunk_size": chunk_size,
        "exchange_chunks": exchange_calls,
        "chunk_windows": chunk_count,
        "requests": {
            exchange_key: exchange_calls,
            klines_key: klines_requests,
        },
        "total_requests": exchange_calls + klines_requests,
    }
    if not symbols and source == "unknown":
        plan["notes"] = "symbol list unavailable; provide --symbols or checkpoint"
    return plan


def _load_rest_budget_config(path: str) -> Tuple[dict[str, Any], dict[str, Any]]:
    config_path = path.strip()
    if not config_path:
        return {}, {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[WARN] rest budget config not found: {config_path}", file=sys.stderr)
        return {}, {}
    except Exception as exc:  # pragma: no cover - defensive CLI warning
        print(
            f"[WARN] failed to load rest budget config {config_path}: {exc}",
            file=sys.stderr,
        )
        return {}, {}

    if not isinstance(payload, Mapping):
        print(
            f"[WARN] rest budget config {config_path} must be a mapping, got {type(payload).__name__}",
            file=sys.stderr,
        )
        return {}, {}

    rest_cfg_raw: Mapping[str, Any] | None = None
    if isinstance(payload.get("rest_budget"), Mapping):
        rest_cfg_raw = payload["rest_budget"]  # type: ignore[index]
    else:
        rest_cfg_raw = payload

    script_cfg: dict[str, Any] = {}
    if isinstance(payload.get("fetch_exchange_specs"), Mapping):
        script_cfg = dict(payload["fetch_exchange_specs"])  # type: ignore[index]
    elif isinstance(rest_cfg_raw, Mapping) and isinstance(
        rest_cfg_raw.get("fetch_exchange_specs"), Mapping
    ):
        script_cfg = dict(rest_cfg_raw["fetch_exchange_specs"])  # type: ignore[index]

    rest_session_cfg: Mapping[str, Any] | None = None
    if isinstance(rest_cfg_raw, Mapping):
        if isinstance(rest_cfg_raw.get("session"), Mapping):
            rest_session_cfg = rest_cfg_raw["session"]  # type: ignore[index]
        elif isinstance(rest_cfg_raw.get("config"), Mapping):
            rest_session_cfg = rest_cfg_raw["config"]  # type: ignore[index]
        else:
            rest_session_cfg = rest_cfg_raw

    return (dict(rest_session_cfg) if rest_session_cfg else {}), script_cfg


def _coerce_positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _resolve_chunk_size(script_cfg: Mapping[str, Any], rest_cfg: Mapping[str, Any]) -> int:
    candidates: list[Any] = []
    if "chunk_size" in script_cfg:
        candidates.append(script_cfg["chunk_size"])
    if "max_symbols_per_request" in script_cfg:
        candidates.append(script_cfg["max_symbols_per_request"])
    chunk_cfg = script_cfg.get("chunk")
    if isinstance(chunk_cfg, Mapping):
        candidates.extend(
            [
                chunk_cfg.get("size"),
                chunk_cfg.get("chunk_size"),
                chunk_cfg.get("max_symbols"),
            ]
        )

    endpoints_cfg = rest_cfg.get("endpoints")
    if isinstance(endpoints_cfg, Mapping):
        exchange_cfg = endpoints_cfg.get("exchangeInfo")
        if isinstance(exchange_cfg, Mapping):
            candidates.extend(
                [
                    exchange_cfg.get("chunk_size"),
                    exchange_cfg.get("max_symbols"),
                    exchange_cfg.get("max_symbols_per_request"),
                ]
            )

    for value in candidates:
        number = _coerce_positive_int(value)
        if number is not None:
            return number
    return 200


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fetch Binance exchangeInfo and save minimal specs JSON (tickSize/stepSize/minNotional) per symbol.",
    )
    p.add_argument("--market", choices=["spot", "futures"], default="futures", help="Какой рынок опрашивать")
    p.add_argument("--symbols", default="", help="Список символов через запятую; пусто = все")
    p.add_argument("--out", default="data/exchange_specs.json", help="Куда сохранить JSON")
    p.add_argument(
        "--volume-threshold",
        type=float,
        default=float(os.getenv("QUOTE_VOLUME_THRESHOLD", 0.0)),
        help="Минимальный средний quote volume за период",
    )
    p.add_argument(
        "--volume-out",
        default="data/volume_metrics.json",
        help="Куда сохранить средний quote volume по символам",
    )
    p.add_argument("--days", type=int, default=30, help="Число дней для оценки quote volume")
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Перемешать порядок символов перед обращениями к API",
    )
    p.add_argument(
        "--checkpoint-path",
        default="",
        help="Путь к файлу чекпоинта (JSON); пусто = не сохранять прогресс",
    )
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Возобновить обработку из чекпоинта, если он найден",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Игнорировать существующий чекпоинт",
    )
    p.set_defaults(resume=False)
    p.add_argument(
        "--rest-budget-config",
        default="configs/rest_budget.yaml",
        help="Путь к YAML с настройками RestBudgetSession (по умолчанию configs/rest_budget.yaml)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Рассчитать количество HTTP-запросов без фактического обращения к API",
    )
    args = p.parse_args()

    checkpoint_cfg: dict[str, object] = {}
    checkpoint_path = args.checkpoint_path.strip()
    if checkpoint_path:
        resume_flag = bool(args.resume)
        checkpoint_cfg = {
            "path": checkpoint_path,
            "enabled": True,
            "resume_from_checkpoint": resume_flag,
        }
    rest_cfg, script_cfg = _load_rest_budget_config(str(args.rest_budget_config or ""))
    chunk_size = _resolve_chunk_size(script_cfg, rest_cfg)

    if checkpoint_cfg:
        existing = rest_cfg.get("checkpoint")
        merged_checkpoint: dict[str, object] = {}
        if isinstance(existing, dict):
            merged_checkpoint.update(existing)
        merged_checkpoint.update(checkpoint_cfg)
        rest_cfg = dict(rest_cfg)
        rest_cfg["checkpoint"] = merged_checkpoint

    with RestBudgetSession(rest_cfg) as session:
        if args.dry_run:
            plan = _plan_dry_run(args, session, chunk_size=chunk_size)
            stats = session.stats()
            stats["plan"] = plan
            print(json.dumps(stats, ensure_ascii=False))
            return

        handled_signals: dict[int, Any] = {}
        checkpoint_state: dict[str, Any] = {"payload": None}

        def _checkpoint_listener(payload: dict[str, Any]) -> None:
            checkpoint_state["payload"] = payload

        def _handle_signal(signum: int, frame: Any | None) -> None:  # pragma: no cover - signal handler
            payload = checkpoint_state.get("payload")
            if payload is not None:
                session.save_checkpoint(payload)
            if signum == getattr(signal, "SIGINT", None):
                raise KeyboardInterrupt
            raise SystemExit(128 + signum)

        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                handled_signals[sig] = signal.getsignal(sig)
                signal.signal(sig, _handle_signal)
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                handled_signals.pop(sig, None)

        try:
            run(
                market=args.market,
                symbols=args.symbols,
                out=args.out,
                volume_threshold=args.volume_threshold,
                volume_out=args.volume_out,
                days=args.days,
                shuffle=args.shuffle,
                session=session,
                checkpoint_listener=_checkpoint_listener,
                install_signal_handlers=False,
                chunk_size=chunk_size,
            )
        finally:
            for sig, handler in handled_signals.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):  # pragma: no cover - platform dependent
                    pass
            print(json.dumps(session.stats(), ensure_ascii=False))


if __name__ == "__main__":
    main()
