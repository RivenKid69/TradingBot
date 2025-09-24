# sim/sim_logging.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from decimal import Decimal

import pandas as pd

from compat_shims import trade_dict_to_core_exec_report
from core_models import TradeLogRow, EquityPoint, ExecReport, Side, OrderType, Liquidity, ExecStatus


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
    reports_path: str = "logs/report_equity_<runid>.csv"
    flush_every: int = 1000

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LogConfig":
        return cls(
            enabled=bool(d.get("enabled", True)),
            format=str(d.get("format", "csv")).lower(),
            trades_path=str(d.get("trades_path", "logs/log_trades_<runid>.csv")),
            reports_path=str(d.get("reports_path", "logs/report_equity_<runid>.csv")),
            flush_every=int(d.get("flush_every", 1000)),
        )


class LogWriter:
    """
    Простой файловый логгер:
      - append(report, symbol, ts_ms): кладёт трейды и снэпшот отчёта в буфер
      - flush(): сбрасывает буферы на диск
    """
    def __init__(self, cfg: Optional[LogConfig] = None, *, run_id: str = "sim"):
        self.cfg = cfg or LogConfig()
        self._run_id = str(run_id)
        self.cfg.trades_path = self.cfg.trades_path.replace("<runid>", self._run_id)
        self.cfg.reports_path = self.cfg.reports_path.replace("<runid>", self._run_id)
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
        rep_dict = report.to_dict() if hasattr(report, "to_dict") else {}
        regime_value = rep_dict.get("market_regime", getattr(report, "market_regime", None))
        wrote_trade = False
        for t in rep_dict.get("trades", []):
            er = trade_dict_to_core_exec_report(t, parent=rep_dict, symbol=str(symbol), run_id=self._run_id)
            for k in ["slippage_bps", "spread_bps", "latency_ms", "tif", "ttl_steps"]:
                v = t.get(k)
                if v is not None:
                    er.meta[k] = v
            row = TradeLogRow.from_exec(er).to_dict()
            if row.get("slippage_bps") is not None:
                row["slippage_bps"] = float(row["slippage_bps"])
            if row.get("spread_bps") is not None:
                row["spread_bps"] = float(row["spread_bps"])
            if row.get("latency_ms") is not None:
                row["latency_ms"] = int(row["latency_ms"])
            if row.get("ttl_steps") is not None:
                row["ttl_steps"] = int(row["ttl_steps"])
            row["mark_price"] = float(getattr(report, "mark_price", row.get("mark_price", 0.0)))
            row["equity"] = float(getattr(report, "equity", row.get("equity", 0.0)))
            row["drawdown"] = (
                float(getattr(report, "drawdown"))
                if getattr(report, "drawdown", None) is not None
                else (float(row["drawdown"]) if row.get("drawdown") is not None else None)
            )
            row["execution_profile"] = getattr(er, "execution_profile", None)
            if regime_value is not None:
                row["market_regime"] = regime_value
            try:
                row["notional"] = float(row.get("notional", 0.0))
            except Exception:
                row["notional"] = float(
                    Decimal(str(row.get("price", 0.0))) * Decimal(str(row.get("quantity", 0.0)))
                )
            self._trades_buf.append(row)
            wrote_trade = True
        cancel_map = rep_dict.get("cancelled_reasons", {}) or {}
        for cid in rep_dict.get("cancelled_ids", []):
            reason = cancel_map.get(cid) or cancel_map.get(str(cid)) or "OTHER"
            er_cancel = ExecReport(
                ts=int(ts_ms),
                run_id=self._run_id,
                symbol=str(symbol),
                execution_profile=str(getattr(report, "execution_profile", "")) or None,
                side=Side.BUY,
                order_type=OrderType.MARKET,
                price=Decimal("0"),
                quantity=Decimal("0"),
                fee=Decimal("0"),
                fee_asset=None,
                exec_status=ExecStatus.CANCELED,
                liquidity=Liquidity.UNKNOWN,
                client_order_id=str(cid),
                order_id=None,
                trade_id=None,
                pnl=None,
                meta={
                    "mark_price": float(getattr(report, "mark_price", 0.0)),
                    "equity": float(getattr(report, "equity", 0.0)),
                    "drawdown": float(getattr(report, "drawdown", 0.0)) if getattr(report, "drawdown", None) is not None else None,
                    "reason": str(reason),
                },
            )
            row_cancel = TradeLogRow.from_exec(er_cancel).to_dict()
            if regime_value is not None:
                row_cancel["market_regime"] = regime_value
            self._trades_buf.append(row_cancel)
            wrote_trade = True
        if not wrote_trade:
            placeholder_meta = {
                "mark_price": float(getattr(report, "mark_price", 0.0)),
                "equity": float(getattr(report, "equity", 0.0)),
            }
            drawdown_val = getattr(report, "drawdown", None)
            if drawdown_val is not None:
                placeholder_meta["drawdown"] = float(drawdown_val)
            status_val = getattr(report, "status", None)
            if status_val:
                placeholder_meta["status"] = str(status_val)
            synthetic_er = ExecReport(
                ts=int(ts_ms),
                run_id=self._run_id,
                symbol=str(symbol),
                execution_profile=str(getattr(report, "execution_profile", "")) or None,
                side=Side.BUY,
                order_type=OrderType.MARKET,
                price=Decimal(str(getattr(report, "mark_price", 0.0))),
                quantity=Decimal("0"),
                fee=Decimal("0"),
                fee_asset=None,
                exec_status=ExecStatus.NEW,
                liquidity=Liquidity.UNKNOWN,
                client_order_id=None,
                order_id=None,
                trade_id=None,
                pnl=None,
                meta=placeholder_meta,
            )
            synthetic_row = TradeLogRow.from_exec(synthetic_er).to_dict()
            if regime_value is not None:
                synthetic_row["market_regime"] = regime_value
            self._trades_buf.append(synthetic_row)
        eq = EquityPoint(
            ts=int(ts_ms),
            run_id=self._run_id,
            symbol=str(symbol),
            execution_profile=str(getattr(report, "execution_profile", "")) or None,
            fee_total=Decimal(str(getattr(report, "fee_total", 0.0))),
            position_qty=Decimal(str(getattr(report, "position_qty", 0.0))),
            realized_pnl=Decimal(str(getattr(report, "realized_pnl", 0.0))),
            unrealized_pnl=Decimal(str(getattr(report, "unrealized_pnl", 0.0))),
            equity=Decimal(str(getattr(report, "equity", 0.0))),
            mark_price=Decimal(str(getattr(report, "mark_price", 0.0))),
            notional=Decimal(str(getattr(report, "position_qty", 0.0))) * Decimal(str(getattr(report, "mark_price", 0.0))),
            drawdown=Decimal(str(getattr(report, "drawdown", 0.0))) if getattr(report, "drawdown", None) is not None else None,
            risk_paused_until_ms=int(getattr(report, "risk_paused_until_ms", 0)),
            risk_events_count=int(len(getattr(report, "risk_events", []) or [])),
            funding_events_count=int(len(getattr(report, "funding_events", []) or [])),
            funding_cashflow=Decimal(str(getattr(report, "funding_cashflow", 0.0))) if getattr(report, "funding_cashflow", None) is not None else None,
            cash=Decimal(str(getattr(report, "cash", 0.0))) if getattr(report, "cash", None) is not None else None,
        )
        eq_dict = eq.to_dict()
        eq_dict["spread_bps"] = getattr(report, "spread_bps", None)
        eq_dict["vol_factor"] = getattr(report, "vol_factor", None)
        eq_dict["liquidity"] = getattr(report, "liquidity", None)
        eq_dict["latency_p50_ms"] = getattr(report, "latency_p50_ms", None)
        eq_dict["latency_p95_ms"] = getattr(report, "latency_p95_ms", None)
        eq_dict["latency_timeout_ratio"] = getattr(report, "latency_timeout_ratio", None)
        if regime_value is not None:
            eq_dict["market_regime"] = regime_value
        self._reports_buf.append(eq_dict)

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

    def flush_fsync(self) -> None:
        """Flush buffers and fsync output files to disk."""
        self.flush()
        if not self.cfg.enabled:
            return
        for path in (self.cfg.trades_path, self.cfg.reports_path):
            try:
                fd = os.open(path, os.O_RDONLY)
                os.fsync(fd)
                os.close(fd)
            except Exception:
                pass

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
