from __future__ import annotations

import argparse

from service_fetch_exchange_specs import run


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fetch Binance exchangeInfo and save minimal specs JSON (tickSize/stepSize/minNotional) per symbol.",
    )
    p.add_argument("--market", choices=["spot", "futures"], default="futures", help="Какой рынок опрашивать")
    p.add_argument("--symbols", default="", help="Список символов через запятую; пусто = все")
    p.add_argument("--out", default="data/exchange_specs.json", help="Куда сохранить JSON")
    args = p.parse_args()

    run(market=args.market, symbols=args.symbols, out=args.out)


if __name__ == "__main__":
    main()
