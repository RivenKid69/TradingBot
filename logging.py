# sim/logging.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pandas as pd


@dataclass
class LogConfig:
    """
    Конфигурация логирования симуляции.
      - enabled: включено/выключено
      - format: "csv" | "parquet"
      - trades_path: путь для трейдов (csv-файл или базовый путь для паркет-частей)
      - reports_path: путь для отчётов (csv-файл или базовый путь для паркет-частей)
      - flush_every: сколько записей буферизовать перед сбросом на диск
    Примечание по parquet:
      Мы пишем «частями»: <path>.part-000001.parquet, <path>.part-000002.parquet, ...
    """
    enabled: bool = True
    format: str = "csv"
    trades_path: str = "logs/log_trades_<runid>.csv"
    reports_path: str = "logs/sim_reports.csv"
    flush_every: int = 1000

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LogConfig":
        return cls(
            enabled=bool(d.get("enabled", True)),
            format=str(d.get("format", "csv")).lower(),
            trades_path=str(d.get("trades_path", "logs/log_trades_<runid>.csv")),
            reports_path=str(d.get("reports_path", "logs/reports.csv")),
            flush_every=int(d.get("flush_every", 1000)),
        )


class LogWriter:
    """
    Простой файловый логгер:
      - append(report, symbol, ts_ms): кладёт трейды и снэпшот отчёта в буфер
      - flush(): сбрасывает буферы на диск
    """
    def __init__(self, cfg: Optional[LogConfig] = None):
        self.cfg = cfg or LogConfig()
        self._trades_buf: List[Dict[str, Any]] = []
        self._reports_buf: List[Dict[str, Any]] = []
        self._part_counter_trades: int = 0
        self._part_counter_reports: int = 0
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        os.makedirs(os.path.dirname(self.cfg.trades_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.cfg.reports_path) or ".", exist_ok=True)

    def append(self, report: Any, *, symbol: str, ts_ms: int) -> None:
        if not self.cfg.enabled:
            return
        # трейды
        for t in report.trades:
            row = dict(t.__dict__)
            row["symbol"] = str(symbol)
            # ts трейда уже в t.ts
            # добавим пару полей из отчёта для удобства
            row["mark_price"] = float(report.mark_price)
            row["equity"] = float(report.equity)
            self._trades_buf.append(row)
        # отчёт (одна строка на шаг)
        rep = {
            "ts_ms": int(ts_ms),
            "symbol": str(symbol),
            "fee_total": float(report.fee_total),
            "position_qty": float(report.position_qty),
            "realized_pnl": float(report.realized_pnl),
            "unrealized_pnl": float(report.unrealized_pnl),
            "equity": float(report.equity),
            "mark_price": float(report.mark_price),
            "risk_paused_until_ms": int(getattr(report, "risk_paused_until_ms", 0)),
        }
        # события риска/фандинга храним как количества (для быстроты сводок)
        rep["risk_events_count"] = int(len(getattr(report, "risk_events", []) or []))
        rep["funding_events_count"] = int(len(getattr(report, "funding_events", []) or []))
        self._reports_buf.append(rep)

        # авто-сброс
        if (len(self._trades_buf) + len(self._reports_buf)) >= max(1, int(self.cfg.flush_every)):
            self.flush()

    def flush(self) -> None:
        if not self.cfg.enabled:
            return
        if self.cfg.format == "csv":
            self._flush_csv()
        else:
            self._flush_parquet()

    def _flush_csv(self) -> None:
        if self._trades_buf:
            df_t = pd.DataFrame(self._trades_buf)
            write_header = not os.path.exists(self.cfg.trades_path)
            df_t.to_csv(self.cfg.trades_path, index=False, mode="a", header=write_header)
            self._trades_buf.clear()
        if self._reports_buf:
            df_r = pd.DataFrame(self._reports_buf)
            write_header = not os.path.exists(self.cfg.reports_path)
            df_r.to_csv(self.cfg.reports_path, index=False, mode="a", header=write_header)
            self._reports_buf.clear()

    def _flush_parquet(self) -> None:
        # Паркет-части: <path>.part-000001.parquet
        if self._trades_buf:
            self._part_counter_trades += 1
            df_t = pd.DataFrame(self._trades_buf)
            out = f"{self.cfg.trades_path}.part-{self._part_counter_trades:06d}.parquet"
            df_t.to_parquet(out, index=False)
            self._trades_buf.clear()
        if self._reports_buf:
            self._part_counter_reports += 1
            df_r = pd.DataFrame(self._reports_buf)
            out = f"{self.cfg.reports_path}.part-{self._part_counter_reports:06d}.parquet"
            df_r.to_parquet(out, index=False)
            self._reports_buf.clear()
