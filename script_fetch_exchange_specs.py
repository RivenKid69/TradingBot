from __future__ import annotations

import argparse
import os

from service_fetch_exchange_specs import run


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
    args = p.parse_args()

    run(
        market=args.market,
        symbols=args.symbols,
        out=args.out,
        volume_threshold=args.volume_threshold,
        volume_out=args.volume_out,
        days=args.days,
    )


if __name__ == "__main__":
    main()
