# -*- coding: utf-8 -*-
"""
services/service_eval.py
Сервис оценки: читает логи трейдов и equity, считает метрики, сохраняет отчёт.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import os
import time
import pandas as pd
import numpy as np

from services.utils_config import snapshot_config
from services.metrics import calculate_metrics, equity_from_trades, safe_read_csv


@dataclass
class EvalConfig:
    trades_csv: Optional[str] = None          # стандартизированный лог сделок
    equity_csv: Optional[str] = None          # кривая equity (если есть готовая)
    artifacts_dir: str = "artifacts"          # куда сохранить отчёт/копии
    snapshot_config_path: Optional[str] = None

    # имена колонок в логах
    ts_col: str = "ts_ms"
    pnl_col: str = "pnl"
    side_col: str = "side"
    price_col: str = "price"
    qty_col: str = "qty"
    equity_col: str = "equity"


class ServiceEval:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg

    def run(self) -> Dict[str, Any]:
        os.makedirs(self.cfg.artifacts_dir, exist_ok=True)
        if self.cfg.snapshot_config_path:
            snapshot_config(self.cfg.snapshot_config_path, self.cfg.artifacts_dir)

        # загрузка логов
        trades = safe_read_csv(self.cfg.trades_csv) if self.cfg.trades_csv else pd.DataFrame()
        equity = safe_read_csv(self.cfg.equity_csv) if self.cfg.equity_csv else pd.DataFrame()

        # если нет готовой equity — синтезируем из логов pnl
        if equity.empty:
            equity = equity_from_trades(trades, ts_col=self.cfg.ts_col, pnl_col=self.cfg.pnl_col, equity_col=self.cfg.equity_col)

        # метрики
        metrics = calculate_metrics(trades, equity, ts_col=self.cfg.ts_col, pnl_col=self.cfg.pnl_col, equity_col=self.cfg.equity_col)

        # сохраняем отчёт
        ts = int(time.time())
        rep_path = os.path.join(self.cfg.artifacts_dir, f"report_eval_{ts}.json")
        pd.Series(metrics).to_json(rep_path, force_ascii=False)

        return {
            "report_path": rep_path,
            "metrics": metrics,
            "n_trades": int(len(trades)) if not trades.empty else 0,
            "n_equity_points": int(len(equity)) if not equity.empty else 0,
        }
