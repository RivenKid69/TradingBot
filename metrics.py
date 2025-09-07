# reports/metrics.py
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class EquityMetrics:
    start_ts: int
    end_ts: int
    num_points: int
    pnl_total: float
    pnl_mean_step: float
    pnl_std_step: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_dd_start_ts: int
    max_dd_end_ts: int
    avg_step_seconds: float
    turnover: float
    fees_sum: float
    funding_cashflow_sum: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradeMetrics:
    n_trades: int
    n_buy: int
    n_sell: int
    win_rate: float
    avg_pnl: float
    median_pnl: float
    std_pnl: float
    gross_profit: float
    gross_loss: float
    avg_slippage_bps: float
    avg_spread_bps: float
    vwap_price: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe(v: float) -> float:
    try:
        x = float(v)
        if not math.isfinite(x):
            return float("nan")
        return x
    except Exception:
        return float("nan")


def _drawdown(equity: pd.Series) -> Tuple[pd.Series, float, int, int]:
    """
    Возвращает:
      - серию drawdown (equity / rolling_max - 1)
      - max_drawdown (отрицательное число)
      - ts начала максимальной просадки
      - ts конца максимальной просадки
    """
    if equity.empty:
        return pd.Series(dtype=float), 0.0, 0, 0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    i_end = int(dd.idxmin()) if len(dd) else 0
    if len(dd) == 0:
        return dd, 0.0, 0, 0
    max_dd = float(dd.min())
    if i_end not in dd.index:
        return dd, max_dd, 0, 0
    # начало — индекс предыдущего максимума equity до минимума dd
    sub = equity.loc[:i_end]
    if sub.empty:
        return dd, max_dd, 0, int(i_end)
    i_start = int(sub.idxmax())
    return dd, max_dd, i_start, int(i_end)


def _annualize_factor(period_seconds: float) -> float:
    # год ≈ 365 дней
    if period_seconds <= 0:
        return 0.0
    return float((365.0 * 24.0 * 3600.0) / period_seconds)


def _sharpe(returns: np.ndarray, rf_per_step: float = 0.0, ann_factor: float = 1.0) -> float:
    if returns.size == 0:
        return float("nan")
    ex = returns - rf_per_step
    mu = float(np.nanmean(ex))
    sd = float(np.nanstd(ex, ddof=1)) if returns.size > 1 else float("nan")
    if not math.isfinite(sd) or sd == 0.0:
        return float("nan")
    return float(mu / sd * math.sqrt(ann_factor))


def _sortino(returns: np.ndarray, rf_per_step: float = 0.0, ann_factor: float = 1.0) -> float:
    if returns.size == 0:
        return float("nan")
    ex = returns - rf_per_step
    neg = ex.copy()
    neg[neg > 0] = 0.0
    ds = float(np.sqrt(np.nanmean(neg ** 2)))  # downside std
    mu = float(np.nanmean(ex))
    if not math.isfinite(ds) or ds == 0.0:
        return float("nan")
    return float(mu / ds * math.sqrt(ann_factor))


def _vwap(px: pd.Series, qty: pd.Series) -> float:
    w = qty.abs().astype(float)
    if w.sum() <= 0:
        return float("nan")
    return float((px.astype(float) * w).sum() / w.sum())


def compute_trade_metrics(trades: pd.DataFrame) -> TradeMetrics:
    if trades is None or trades.empty:
        return TradeMetrics(
            n_trades=0, n_buy=0, n_sell=0, win_rate=float("nan"), avg_pnl=float("nan"),
            median_pnl=float("nan"), std_pnl=float("nan"), gross_profit=0.0, gross_loss=0.0,
            avg_slippage_bps=float("nan"), avg_spread_bps=float("nan"), vwap_price=float("nan")
        )
    t = trades.copy()
    # если нет pnl на трейд, приблизим как side_sign * (price - ref_price_prev)*qty — но в наших трейдах нет ref_prev,
    # поэтому ограничимся агрегатной статистикой по слиппеджу/спрэду и notional.
    t["side_sign"] = t["side"].map(lambda s: 1.0 if str(s).upper() == "BUY" else -1.0)
    n_trades = int(len(t))
    n_buy = int((t["side_sign"] > 0).sum())
    n_sell = int((t["side_sign"] < 0).sum())

    # Если в логе был явный pnl по трейдам — поддержим, иначе используем fee как прокси издержек для win_rate≈NaN.
    pnl_col = "pnl"
    if pnl_col not in t.columns:
        # нет per-trade pnl — считаем win_rate как NaN
        win_rate = float("nan")
        avg_pnl = float("nan")
        med_pnl = float("nan")
        std_pnl = float("nan")
        gross_profit = float("nan")
        gross_loss = float("nan")
    else:
        pnl = t[pnl_col].astype(float)
        win_rate = float((pnl > 0).mean()) if n_trades > 0 else float("nan")
        avg_pnl = float(pnl.mean()) if n_trades > 0 else float("nan")
        med_pnl = float(pnl.median()) if n_trades > 0 else float("nan")
        std_pnl = float(pnl.std(ddof=1)) if n_trades > 1 else float("nan")
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl < 0].sum())

    avg_slip = float(t["slippage_bps"].mean()) if "slippage_bps" in t.columns and len(t) else float("nan")
    avg_spread = float(t["spread_bps"].mean()) if "spread_bps" in t.columns and len(t) else float("nan")
    vwap = _vwap(t["price"] if "price" in t.columns else pd.Series(dtype=float),
                 t["qty"] if "qty" in t.columns else pd.Series(dtype=float))

    return TradeMetrics(
        n_trades=n_trades,
        n_buy=n_buy,
        n_sell=n_sell,
        win_rate=win_rate,
        avg_pnl=avg_pnl,
        median_pnl=med_pnl,
        std_pnl=std_pnl,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        avg_slippage_bps=avg_slip,
        avg_spread_bps=avg_spread,
        vwap_price=vwap,
    )


def compute_equity_metrics(
    reports: pd.DataFrame,
    *,
    capital_base: float = 10_000.0,
    rf_annual: float = 0.0
) -> EquityMetrics:
    """
    reports: ожидаем хотя бы ts_ms, equity, fee_total, funding_cashflow.
    Возвращаем метрики на базе приращений equity и нормированных «возвратов»:
      r_t = dEquity / capital_base
    """
    if reports is None or reports.empty:
        return EquityMetrics(
            start_ts=0, end_ts=0, num_points=0, pnl_total=0.0, pnl_mean_step=float("nan"),
            pnl_std_step=float("nan"), sharpe=float("nan"), sortino=float("nan"),
            calmar=float("nan"), max_drawdown=0.0, max_dd_start_ts=0, max_dd_end_ts=0,
            avg_step_seconds=float("nan"), turnover=float("nan"), fees_sum=0.0, funding_cashflow_sum=0.0
        )

    r = reports.copy()
    r = r.sort_values("ts_ms").reset_index(drop=True)
    eq = r["equity"].astype(float)
    ts = r["ts_ms"].astype("int64")

    # шаги по времени
    if len(ts) > 1:
        dt_s = (ts.diff().dropna().astype(float) / 1000.0).values
        avg_dt_s = float(np.nanmean(dt_s)) if dt_s.size else float("nan")
    else:
        avg_dt_s = float("nan")

    # приращения PnL
    d_eq = eq.diff().fillna(0.0)
    pnl_total = float(eq.iloc[-1] - eq.iloc[0]) if len(eq) > 0 else 0.0

    # возвраты (на базовый капитал)
    if capital_base <= 0:
        returns = d_eq.values.astype(float)
        ann_factor = 1.0
    else:
        returns = (d_eq.values.astype(float) / float(capital_base))
        # аппроксимируем годовой множитель из среднего шага
        ann_factor = _annualize_factor(avg_dt_s) if math.isfinite(avg_dt_s) and avg_dt_s > 0 else 1.0

    # rf per step
    rf_per_step = 0.0
    if ann_factor > 0 and rf_annual > 0:
        rf_per_step = float(rf_annual) / float(ann_factor)

    sharpe = _sharpe(returns, rf_per_step=rf_per_step, ann_factor=ann_factor)
    sortino = _sortino(returns, rf_per_step=rf_per_step, ann_factor=ann_factor)
    dd_series, max_dd, dd_start, dd_end = _drawdown(eq)

    # Calmar по годовому PnL на capital_base делённому на |maxDD|
    if capital_base > 0 and math.isfinite(max_dd) and max_dd < 0:
        # годовая доходность ≈ mean(returns) * ann_factor
        mu_step = float(np.nanmean(returns)) if returns.size else float("nan")
        if math.isfinite(mu_step):
            calmar = float((mu_step * ann_factor) / abs(max_dd))
        else:
            calmar = float("nan")
    else:
        calmar = float("nan")

    fees_sum = float(r["fee_total"].sum()) if "fee_total" in r.columns else 0.0
    funding_sum = float(r["funding_cashflow"].sum()) if "funding_cashflow" in r.columns else 0.0

    # оборот: сумма абсолютных приращений позиции * mark_price / capital_base,
    # но у нас в reports нет qty_delta — оценим через трейды отдельно в evaluate_performance.
    turnover = float("nan")

    return EquityMetrics(
        start_ts=int(ts.iloc[0]),
        end_ts=int(ts.iloc[-1]),
        num_points=int(len(r)),
        pnl_total=float(pnl_total),
        pnl_mean_step=float(np.nanmean(d_eq.values.astype(float))) if len(d_eq) else float("nan"),
        pnl_std_step=float(np.nanstd(d_eq.values.astype(float), ddof=1)) if len(d_eq) > 1 else float("nan"),
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(calmar),
        max_drawdown=float(max_dd),
        max_dd_start_ts=int(dd_start),
        max_dd_end_ts=int(dd_end),
        avg_step_seconds=float(avg_dt_s),
        turnover=float(turnover),
        fees_sum=float(fees_sum),
        funding_cashflow_sum=float(funding_sum),
    )
