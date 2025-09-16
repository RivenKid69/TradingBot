from __future__ import annotations

import argparse
import os

from service_fetch_exchange_specs import run
from services.rest_budget import RestBudgetSession


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
    session = RestBudgetSession({"checkpoint": checkpoint_cfg} if checkpoint_cfg else {})

    run(
        market=args.market,
        symbols=args.symbols,
        out=args.out,
        volume_threshold=args.volume_threshold,
        volume_out=args.volume_out,
        days=args.days,
        shuffle=args.shuffle,
        session=session,
    )


if __name__ == "__main__":
    main()
