# scripts/ingest_orchestrator.py
from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

from ingest_config import load_config, IngestConfig


_INTERVAL_MS = {
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


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.fspath(path)) or "."
    os.makedirs(directory, exist_ok=True)


def _pick_base_interval(intervals: List[str]) -> str:
    if not intervals:
        return "1m"
    ivals = [i.strip() for i in intervals if i.strip() in _INTERVAL_MS]
    if not ivals:
        return "1m"
    return sorted(ivals, key=lambda x: _INTERVAL_MS[x])[0]


def _is_parquet(path: str) -> bool:
    return path.lower().endswith(".parquet")


_SCRIPT_DIR = Path(__file__).resolve().parent


def _script_path(name: str) -> str:
    return os.fspath(_SCRIPT_DIR / name)


def _run(cmd: List[str]) -> None:
    print(">>", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"Команда завершилась с ошибкой: {' '.join(cmd)}")


def run_from_config(cfg: IngestConfig) -> None:
    symbols: List[str] = [s.upper() for s in cfg.symbols]
    if not symbols:
        raise SystemExit("В конфиге не указаны symbols")

    market: str = cfg.market.lower()
    if market not in ("spot", "futures"):
        raise SystemExit("market должен быть 'spot' или 'futures'")

    intervals: List[str] = cfg.intervals
    aggregate_to: List[str] = cfg.aggregate_to

    start = cfg.period.start
    end = cfg.period.end
    if not start or not end:
        raise SystemExit("period.start и period.end обязательны")

    klines_dir = cfg.paths.klines_dir
    futures_dir = cfg.paths.futures_dir
    prices_out = cfg.paths.prices_out

    mark_interval = cfg.futures.mark_interval

    api_limit = cfg.slowness.api_limit
    sleep_ms = cfg.slowness.sleep_ms

    # 1) Ingest klines для всех символов и всех указанных интервалов
    for sym in symbols:
        for interval in intervals:
            os.makedirs(klines_dir, exist_ok=True)
            cmd = [
                sys.executable,
                _script_path("ingest_klines.py"),
                "--market", market,
                "--symbols", sym,
                "--interval", interval,
                "--start", start,
                "--end", end,
                "--out-dir", klines_dir,
                "--limit", str(api_limit),
                "--sleep-ms", str(sleep_ms),
            ]
            _run(cmd)

    # 2) Агрегация: берём самый мелкий доступный интервал как базу и строим все aggregate_to
    base_interval = _pick_base_interval(intervals)
    for sym in symbols:
        in_path = os.path.join(klines_dir, f"{sym}_{base_interval}.parquet")
        if not os.path.exists(in_path):
            print(f"Предупреждение: нет файла {in_path} — пропускаю агрегирование для {sym}")
            continue
        for target in aggregate_to:
            out_path = os.path.join(klines_dir, f"{sym}_{target}.parquet")
            _ensure_dir(out_path)
            cmd = [
                sys.executable,
                _script_path("agg_klines.py"),
                "--in-path", in_path,
                "--interval", target,
                "--out-path", out_path,
            ]
            _run(cmd)

    # 3) Для фьючей: funding + mark-price
    if market == "futures":
        for sym in symbols:
            os.makedirs(futures_dir, exist_ok=True)
            cmd = [
                sys.executable,
                _script_path("ingest_funding_mark.py"),
                "--symbol", sym,
                "--start", start,
                "--end", end,
                "--mark-interval", mark_interval,
                "--out-dir", futures_dir,
                "--limit", str(api_limit),
                "--sleep-ms", str(sleep_ms),
            ]
            _run(cmd)

    # 4) Нормализовать цены (prices.parquet)
    #    Если символ один — пишем туда, куда указано в prices_out.
    #    Если символов несколько — создаём по файлу на символ: <stem>_<SYM>.parquet
    if len(symbols) == 1:
        sym = symbols[0]
        in_path = os.path.join(klines_dir, f"{sym}_{base_interval}.parquet")
        if not os.path.exists(in_path):
            raise SystemExit(f"Не найдено {in_path} для сборки prices")
        _ensure_dir(prices_out)
        cmd = [
            sys.executable,
            _script_path("make_prices_from_klines.py"),
            "--in-klines", in_path,
            "--symbol", sym,
            "--price-col", "close",
            "--out", prices_out,
        ]
        _run(cmd)
    else:
        stem, ext = os.path.splitext(prices_out)
        for sym in symbols:
            in_path = os.path.join(klines_dir, f"{sym}_{base_interval}.parquet")
            if not os.path.exists(in_path):
                print(f"Предупреждение: нет {in_path}, пропускаю prices для {sym}")
                continue
            out_sym = f"{stem}_{sym}.parquet" if ext.lower() == ".parquet" else os.path.join(prices_out, f"prices_{sym}.parquet")
            _ensure_dir(out_sym)
            cmd = [
                sys.executable,
                _script_path("make_prices_from_klines.py"),
                "--in-klines", in_path,
                "--symbol", sym,
                "--price-col", "close",
                "--out", out_sym,
            ]
            _run(cmd)

    print("Готово: ingest → aggregate → funding/mark (если фьючи) → prices завершены.")


def main():
    parser = argparse.ArgumentParser(description="Orchestrate public Binance ingest (no keys).")
    parser.add_argument("--config", default="configs/ingest.yaml", help="Путь к YAML конфигу")
    args = parser.parse_args()

    cfg: IngestConfig = load_config(args.config)

    run_from_config(cfg)


if __name__ == "__main__":
    main()
