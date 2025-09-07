# scripts/ingest_orchestrator.py
from __future__ import annotations

import argparse
import os
import sys
import subprocess
from typing import Dict, List, Tuple

import yaml


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
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d:
        os.makedirs(d, exist_ok=True)


def _pick_base_interval(intervals: List[str]) -> str:
    if not intervals:
        return "1m"
    ivals = [i.strip() for i in intervals if i.strip() in _INTERVAL_MS]
    if not ivals:
        return "1m"
    return sorted(ivals, key=lambda x: _INTERVAL_MS[x])[0]


def _is_parquet(path: str) -> bool:
    return path.lower().endswith(".parquet")


def _run(cmd: List[str]) -> None:
    print(">>", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"Команда завершилась с ошибкой: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="Orchestrate public Binance ingest (no keys).")
    parser.add_argument("--config", default="configs/ingest.yaml", help="Путь к YAML конфигу")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict = yaml.safe_load(f)

    symbols: List[str] = [str(s).upper() for s in cfg.get("symbols", [])]
    if not symbols:
        raise SystemExit("В конфиге не указаны symbols")

    market: str = str(cfg.get("market", "spot")).lower()
    if market not in ("spot", "futures"):
        raise SystemExit("market должен быть 'spot' или 'futures'")

    intervals: List[str] = [str(i) for i in cfg.get("intervals", ["1m"])]
    aggregate_to: List[str] = [str(i) for i in cfg.get("aggregate_to", [])]

    period = cfg.get("period", {})
    start = str(period.get("start"))
    end = str(period.get("end"))
    if not start or not end:
        raise SystemExit("period.start и period.end обязательны")

    paths = cfg.get("paths", {})
    klines_dir = str(paths.get("klines_dir", "data/klines"))
    futures_dir = str(paths.get("futures_dir", "data/futures"))
    prices_out = str(paths.get("prices_out", "data/prices.parquet"))

    fut_cfg = cfg.get("futures", {})
    mark_interval = str(fut_cfg.get("mark_interval", "1m"))

    slow = cfg.get("slowness", {})
    api_limit = int(slow.get("api_limit", 1500))
    sleep_ms = int(slow.get("sleep_ms", 350))

    # 1) Ingest klines для всех символов и всех указанных интервалов
    for sym in symbols:
        for interval in intervals:
            _ensure_dir(klines_dir)
            cmd = [
                sys.executable,
                "scripts/ingest_klines.py",
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
                "scripts/agg_klines.py",
                "--in-path", in_path,
                "--interval", target,
                "--out-path", out_path,
            ]
            _run(cmd)

    # 3) Для фьючей: funding + mark-price
    if market == "futures":
        for sym in symbols:
            _ensure_dir(futures_dir)
            cmd = [
                sys.executable,
                "scripts/ingest_funding_mark.py",
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
            "scripts/make_prices_from_klines.py",
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
                "scripts/make_prices_from_klines.py",
                "--in-klines", in_path,
                "--symbol", sym,
                "--price-col", "close",
                "--out", out_sym,
            ]
            _run(cmd)

    print("Готово: ingest → aggregate → funding/mark (если фьючи) → prices завершены.")


if __name__ == "__main__":
    main()
