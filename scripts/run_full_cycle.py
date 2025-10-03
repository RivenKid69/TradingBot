#!/usr/bin/env python3
"""High-level orchestration script for historical ingest + inference cycles."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import List, Sequence

# Ensure repository root is on sys.path when executed as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingest_config import (
    FuturesConfig,
    IngestConfig,
    PathsConfig,
    PeriodConfig,
    SlownessConfig,
)
from ingest_orchestrator import run_from_config
from update_and_infer import run_continuous, run_single_cycle


def _split_csv(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if not value:
            continue
        parts = [item.strip() for item in value.split(",") if item.strip()]
        result.extend(parts)
    return result


def _parse_symbols(values: Sequence[str]) -> List[str]:
    symbols = [sym.upper() for sym in _split_csv(values)]
    if not symbols:
        raise ValueError("Список символов не должен быть пустым")
    return symbols


def _parse_intervals(values: Sequence[str]) -> tuple[str, List[str]]:
    intervals = _split_csv(values)
    if not intervals:
        raise ValueError("Необходимо указать хотя бы один интервал")
    base = intervals[0]
    aggregate = intervals[1:]
    return base, aggregate


def _parse_extra_args(raw: str | None) -> List[str]:
    if not raw:
        return []
    return shlex.split(raw)


def build_ingest_config(args: argparse.Namespace) -> IngestConfig:
    symbols = _parse_symbols(args.symbols)
    base_interval, aggregate_to = _parse_intervals(args.interval)

    paths_kwargs = {}
    if args.klines_dir:
        paths_kwargs["klines_dir"] = args.klines_dir
    if args.futures_dir:
        paths_kwargs["futures_dir"] = args.futures_dir
    if args.prices_out:
        paths_kwargs["prices_out"] = args.prices_out
    paths = PathsConfig.from_env(**paths_kwargs)

    futures_kwargs = {}
    if args.mark_interval:
        futures_kwargs["mark_interval"] = args.mark_interval
    futures = FuturesConfig(**futures_kwargs) if futures_kwargs else FuturesConfig()

    slowness_kwargs = {}
    if args.api_limit is not None:
        slowness_kwargs["api_limit"] = args.api_limit
    if args.sleep_ms is not None:
        slowness_kwargs["sleep_ms"] = args.sleep_ms
    slowness = (
        SlownessConfig(**slowness_kwargs)
        if slowness_kwargs
        else SlownessConfig()
    )

    period = PeriodConfig(start=args.start, end=args.end)

    return IngestConfig(
        symbols=symbols,
        market=args.market,
        intervals=[base_interval],
        aggregate_to=aggregate_to,
        period=period,
        paths=paths,
        futures=futures,
        slowness=slowness,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download raw candles, aggregate them and run inference cycle",
    )
    parser.add_argument(
        "--symbols",
        action="append",
        required=True,
        help="Символы (через запятую или несколько флагов)",
    )
    parser.add_argument(
        "--interval",
        action="append",
        required=True,
        help=(
            "Интервалы (первый — базовый для загрузки, остальные будут агрегированы)"
        ),
    )
    parser.add_argument("--start", required=True, help="Начало периода (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Конец периода (YYYY-MM-DD)")
    parser.add_argument("--market", default="futures", choices=["spot", "futures"])
    parser.add_argument("--loop", action="store_true", help="Запустить бесконечный цикл")
    parser.add_argument(
        "--sleep-min",
        type=float,
        default=15.0,
        help="Пауза между циклами (минуты)",
    )
    parser.add_argument(
        "--events-days",
        type=int,
        default=90,
        help="Глубина окна для prepare_events",
    )
    parser.add_argument(
        "--skip-events",
        action="store_true",
        help="Пропустить обновление экономкалендаря",
    )
    parser.add_argument(
        "--prepare-args",
        default="",
        help="Дополнительные аргументы для prepare_*.py (одной строкой)",
    )
    parser.add_argument(
        "--infer-args",
        default="",
        help="Дополнительные аргументы для infer_signals.py (одной строкой)",
    )
    parser.add_argument(
        "--validate-max-age-sec",
        type=int,
        default=3600,
        help=(
            "Максимальный возраст последнего бара для validate_processed (сек). "
            "Значение <=0 отключает проверку."
        ),
    )
    parser.add_argument(
        "--skip-validate-freshness",
        action="store_true",
        help="Полностью пропустить проверку свежести в validate_processed",
    )
    parser.add_argument("--klines-dir", help="Путь для сохранения свечей")
    parser.add_argument("--futures-dir", help="Путь для данных funding/mark")
    parser.add_argument("--prices-out", help="Путь для итогового prices.parquet")
    parser.add_argument(
        "--mark-interval",
        help="Интервал mark price для фьючерсного рынка",
    )
    parser.add_argument(
        "--api-limit",
        type=int,
        help="Ограничение лимита API при загрузке свечей",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        help="Пауза между запросами при загрузке (мс)",
    )

    args = parser.parse_args()

    cfg = build_ingest_config(args)
    run_from_config(cfg)

    prepare_args = _parse_extra_args(args.prepare_args)
    infer_args = _parse_extra_args(args.infer_args)

    if args.loop:
        run_continuous(
            cfg.symbols,
            events_days=args.events_days,
            sleep_minutes=args.sleep_min,
            skip_events=args.skip_events,
            extra_prepare_args=prepare_args,
            extra_infer_args=infer_args,
            validate_max_age_sec=args.validate_max_age_sec,
            skip_validate_freshness=(
                args.skip_validate_freshness or args.validate_max_age_sec <= 0
            ),
        )
    else:
        run_single_cycle(
            cfg.symbols,
            events_days=args.events_days,
            skip_events=args.skip_events,
            extra_prepare_args=prepare_args,
            extra_infer_args=infer_args,
            validate_max_age_sec=args.validate_max_age_sec,
            skip_validate_freshness=(
                args.skip_validate_freshness or args.validate_max_age_sec <= 0
            ),
        )


if __name__ == "__main__":
    main()
