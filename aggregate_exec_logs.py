"""Aggregate execution logs into per-bar and per-day summaries.

Example
-------
```
python aggregate_exec_logs.py \
    --trades 'logs/log_trades_*.csv' \
    --reports 'logs/report_equity_*.csv' \
    --out-bars logs/agg_bars.csv \
    --out-days logs/agg_days.csv \
    --equity-png logs/equity.png \
    --metrics-md logs/metrics.md
```
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from typing import Any, Dict, Tuple

import pandas as pd

from services.metrics import (
    calculate_metrics,
    plot_equity_curve,
    read_any,
)


def _normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize trades to unified schema:
    ts, run_id, symbol, side, order_type, price, quantity, fee, fee_asset, pnl, exec_status, liquidity, client_order_id, order_id, meta_json
    Supports legacy schema: ts, price, volume, side, agent_flag, order_id
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "ts",
                "run_id",
                "symbol",
                "side",
                "order_type",
                "price",
                "quantity",
                "fee",
                "fee_asset",
                "pnl",
                "exec_status",
                "liquidity",
                "client_order_id",
                "order_id",
                "execution_profile",
                "market_regime",
                "meta_json",
            ]
        )

    cols = set(df.columns)

    # Unified already
    if {"ts","run_id","symbol","side","order_type","price","quantity"}.issubset(cols):
        df = df.copy()
        for c in ["fee","pnl"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # ensure required cols exist
        for c in [
            "fee",
            "fee_asset",
            "pnl",
            "exec_status",
            "liquidity",
            "client_order_id",
            "order_id",
            "meta_json",
            "execution_profile",
            "market_regime",
        ]:
            if c not in df.columns:
                df[c] = None
        columns = [
            "ts",
            "run_id",
            "symbol",
            "side",
            "order_type",
            "price",
            "quantity",
            "fee",
            "fee_asset",
            "pnl",
            "exec_status",
            "liquidity",
            "client_order_id",
            "order_id",
            "execution_profile",
            "market_regime",
            "meta_json",
        ]
        if "act_now" in df.columns:
            columns.append("act_now")
        return df[columns]

    # Legacy -> map
    if {"ts","price","volume","side"}.issubset(cols):
        out = pd.DataFrame()
        out["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
        out["run_id"] = ""
        out["symbol"] = "UNKNOWN"
        out["side"] = df["side"].astype(str).str.upper()
        out["order_type"] = "MARKET"
        out["price"] = pd.to_numeric(df["price"], errors="coerce")
        out["quantity"] = pd.to_numeric(df["volume"], errors="coerce")
        out["fee"] = 0.0
        out["fee_asset"] = None
        out["pnl"] = None
        out["exec_status"] = "FILLED"
        out["liquidity"] = "UNKNOWN"
        out["client_order_id"] = None
        out["order_id"] = df["order_id"] if "order_id" in df.columns else None
        out["meta_json"] = "{}"
        out["execution_profile"] = None
        out["market_regime"] = None
        return out

    # Unknown schema -> attempt minimal
    df = df.copy()
    if "ts" not in df.columns:
        df["ts"] = pd.NA
    if "price" not in df.columns:
        df["price"] = pd.NA
    if "quantity" not in df.columns:
        if "volume" in df.columns:
            df["quantity"] = pd.to_numeric(df["volume"], errors="coerce")
        else:
            df["quantity"] = pd.NA
    df["run_id"] = ""
    df["symbol"] = "UNKNOWN"
    df["side"] = df.get("side", "BUY")
    df["order_type"] = df.get("order_type", "MARKET")
    for c in [
        "fee",
        "fee_asset",
        "pnl",
        "exec_status",
        "liquidity",
        "client_order_id",
        "order_id",
        "meta_json",
        "execution_profile",
        "market_regime",
    ]:
        if c not in df.columns:
            df[c] = None
    columns = [
        "ts",
        "run_id",
        "symbol",
        "side",
        "order_type",
        "price",
        "quantity",
        "fee",
        "fee_asset",
        "pnl",
        "exec_status",
        "liquidity",
        "client_order_id",
        "order_id",
        "execution_profile",
        "market_regime",
        "meta_json",
    ]
    if "act_now" in df.columns:
        columns.append("act_now")
    return df[columns]


def _parse_meta(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            return dict(json.loads(text))
        except Exception:
            return {}
    try:
        text = str(value)
    except Exception:
        return {}
    try:
        return dict(json.loads(text))
    except Exception:
        return {}


def _extract_nested_float(meta: Dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """Search ``meta`` recursively for the first numeric value under ``keys``."""

    if not isinstance(meta, Mapping):
        return None

    stack: list[Mapping[str, Any]] = [meta]
    visited: set[int] = set()

    while stack:
        current = stack.pop()
        ident = id(current)
        if ident in visited:
            continue
        visited.add(ident)
        for key in keys:
            if key in current and current[key] is not None:
                try:
                    return float(current[key])
                except (TypeError, ValueError):
                    continue
        for value in current.values():
            if isinstance(value, Mapping):
                stack.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Mapping):
                        stack.append(item)
    return None


def _extract_meta_float(meta: Dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in meta and meta[key] is not None:
            try:
                return float(meta[key])
            except (TypeError, ValueError):
                pass
    decision = meta.get("decision")
    if isinstance(decision, dict):
        for key in keys:
            if key in decision and decision[key] is not None:
                try:
                    return float(decision[key])
                except (TypeError, ValueError):
                    continue
    economics = meta.get("economics")
    if isinstance(economics, Mapping):
        nested = _extract_nested_float(economics, keys)
        if nested is not None:
            return nested
    payload = meta.get("payload")
    if isinstance(payload, Mapping):
        nested = _extract_nested_float(payload, keys)
        if nested is not None:
            return nested
    return None


def _normalize_bool_like(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in {"1", "true", "yes", "y", "on", "t"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "f"}:
            return False
        try:
            numeric = float(normalized)
        except ValueError:
            pass
        else:
            if math.isnan(numeric):
                return None
            return numeric != 0
    try:
        return bool(value)
    except Exception:
        return None


def _extract_meta_bool(meta: Dict[str, Any], keys: tuple[str, ...]) -> bool | None:
    for key in keys:
        if key in meta:
            value = meta[key]
            normalized = _normalize_bool_like(value)
            if normalized is not None:
                return normalized
    decision = meta.get("decision")
    if isinstance(decision, dict):
        return _extract_meta_bool(decision, keys)
    return None


def _infer_execution_mode(meta: Dict[str, Any]) -> str:
    raw = meta.get("execution_mode") or meta.get("mode")
    if isinstance(raw, str):
        mode = raw.strip().lower()
        if mode in {"order", "bar"}:
            return mode
    decision = meta.get("decision")
    if isinstance(decision, dict) and any(
        key in decision for key in ("turnover_usd", "turnover", "notional_usd")
    ):
        return "bar"
    return "order"


def _normalize_reports(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize equity reports to at least ts_ms and equity columns."""
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "ts_ms",
                "symbol",
                "equity",
                "fee_total",
                "funding_cashflow",
                "bid",
                "ask",
                "mtm_price",
                "execution_profile",
                "market_regime",
            ]
        )

    r = df.copy()
    cols = set(r.columns)

    if "ts_ms" not in cols:
        for candidate in ["ts", "timestamp", "time"]:
            if candidate in r.columns:
                r = r.rename(columns={candidate: "ts_ms"})
                break

    if "symbol" not in r.columns:
        r["symbol"] = "UNKNOWN"

    for c in [
        "equity",
        "fee_total",
        "funding_cashflow",
        "bid",
        "ask",
        "mtm_price",
        "execution_profile",
        "market_regime",
    ]:
        if c not in r.columns:
            r[c] = pd.NA

    r["ts_ms"] = pd.to_numeric(r["ts_ms"], errors="coerce").astype("Int64")
    r["equity"] = pd.to_numeric(r["equity"], errors="coerce")
    r["fee_total"] = pd.to_numeric(r["fee_total"], errors="coerce")
    r["funding_cashflow"] = pd.to_numeric(r["funding_cashflow"], errors="coerce")
    r["bid"] = pd.to_numeric(r["bid"], errors="coerce")
    r["ask"] = pd.to_numeric(r["ask"], errors="coerce")
    r["mtm_price"] = pd.to_numeric(r["mtm_price"], errors="coerce")

    return r[
        [
            "ts_ms",
            "symbol",
            "equity",
            "fee_total",
            "funding_cashflow",
            "bid",
            "ask",
            "mtm_price",
            "execution_profile",
            "market_regime",
        ]
    ]


def _bucket_ts_ms(ts_ms: pd.Series, *, bar_seconds: int) -> pd.Series:
    """Floors ms timestamp to bar_seconds buckets."""
    if bar_seconds <= 0:
        raise ValueError(
            f"bar_seconds must be a positive integer (received {bar_seconds!r})"
        )

    step = int(bar_seconds) * 1000
    return (pd.to_numeric(ts_ms, errors="coerce").astype("Int64") // step) * step


def recompute_pnl(trades: pd.DataFrame, reports: pd.DataFrame) -> pd.Series:
    """Recompute total PnL for each report row.

    Trades must contain ``ts``, ``price``, ``quantity`` and ``side`` columns and
    are expected to be in milliseconds. Reports should include ``ts_ms`` along
    with ``bid``, ``ask`` and optional ``mtm_price`` used for mark to market.

    Returns a :class:`pandas.Series` aligned with ``reports`` containing the
    recomputed PnL (realized + unrealized) at each report timestamp.
    """

    if reports is None or reports.empty:
        return pd.Series(dtype=float)

    if trades is None or trades.empty:
        t_ts: list[float] = []
        t_side: list[str] = []
        t_price: list[float] = []
        t_qty: list[float] = []
    else:
        t = trades.sort_values("ts").copy()
        t_ts = t["ts"].values.tolist()
        t_side = t["side"].astype(str).str.upper().values.tolist()
        t_price = t["price"].astype(float).values.tolist()
        t_qty = t["quantity"].astype(float).abs().values.tolist()
    r = reports.sort_values("ts_ms").copy()

    pos = 0.0
    avg = None
    realized = 0.0
    i = 0
    out: list[float] = []

    for _, rep in r.iterrows():
        ts = float(rep["ts_ms"]) if pd.notna(rep["ts_ms"]) else float("inf")
        while i < len(t_ts) and t_ts[i] <= ts:
            price = t_price[i]
            qty = t_qty[i]
            side = t_side[i]
            if side == "BUY":
                if pos < 0.0:
                    close_qty = min(qty, -pos)
                    if avg is not None:
                        realized += (avg - price) * close_qty
                    pos += close_qty
                    qty -= close_qty
                    if qty > 0.0:
                        pos += qty
                        avg = price
                    elif pos == 0.0:
                        avg = None
                else:
                    new_pos = pos + qty
                    avg = (avg * pos + price * qty) / new_pos if pos > 0.0 and avg is not None else price
                    pos = new_pos
            else:  # SELL
                if pos > 0.0:
                    close_qty = min(qty, pos)
                    if avg is not None:
                        realized += (price - avg) * close_qty
                    pos -= close_qty
                    qty -= close_qty
                    if qty > 0.0:
                        pos -= qty
                        avg = price
                    elif pos == 0.0:
                        avg = None
                else:
                    new_pos = pos - qty
                    avg = (avg * (-pos) + price * qty) / (-new_pos) if pos < 0.0 and avg is not None else price
                    pos = new_pos
            i += 1

        bid = rep.get("bid")
        ask = rep.get("ask")
        mark_p = rep.get("mtm_price")
        if pd.isna(mark_p):
            mark_p = None
        if mark_p is None:
            if pos > 0.0 and pd.notna(bid):
                mark_p = float(bid)
            elif pos < 0.0 and pd.notna(ask):
                mark_p = float(ask)
            elif pd.notna(bid) and pd.notna(ask):
                mark_p = float((float(bid) + float(ask)) / 2.0)

        unrealized = 0.0
        if mark_p is not None and avg is not None and pos != 0.0:
            if pos > 0.0:
                unrealized = (float(mark_p) - avg) * pos
            else:
                unrealized = (avg - float(mark_p)) * (-pos)
        out.append(realized + unrealized)

    return pd.Series(out, index=r.index)


def aggregate(
    trades_path: str,
    reports_path: str,
    out_bars: str,
    out_days: str,
    *,
    bar_seconds: int = 60,
    equity_png: str = "",
    metrics_md: str = "",
) -> Tuple[str, str]:
    """
    Aggregates trade logs into per-bar and per-day summaries.
    - trades_path: path or glob to log_trades_*.csv (unified) or legacy trades.csv
    - reports_path: optional path/glob to equity reports (csv/parquet) — if present, we will attach equity at bar ends
    - out_bars, out_days: output CSV paths
    Returns (out_bars, out_days).
    """
    trades_raw = read_any(trades_path)
    trades = _normalize_trades(trades_raw)

    if trades.empty:
        # write empty frames to keep pipeline consistent
        pd.DataFrame().to_csv(out_bars, index=False)
        pd.DataFrame().to_csv(out_days, index=False)
        return out_bars, out_days

    # Ensure numeric types
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce").fillna(0.0)
    trades["ts"] = pd.to_numeric(trades["ts"], errors="coerce").astype("Int64")
    trades["side_sign"] = trades["side"].astype(str).map(lambda s: 1 if s.upper() == "BUY" else -1)

    # Extract execution metadata (supports both order and bar modes)
    meta_series = trades["meta_json"].apply(_parse_meta)
    trades["execution_mode"] = meta_series.apply(_infer_execution_mode)

    turnover_from_meta = meta_series.apply(
        lambda m: _extract_meta_float(m, ("turnover_usd", "turnover", "notional_usd")) or 0.0
    )
    cap_from_meta = meta_series.apply(
        lambda m: _extract_meta_float(m, ("cap_usd", "cap_quote", "daily_notional_cap"))
    )
    adv_from_meta = meta_series.apply(
        lambda m: _extract_meta_float(m, ("adv_quote", "adv_usd"))
    )
    act_now_from_meta = meta_series.apply(
        lambda m: _extract_meta_bool(m, ("act_now", "execute_now"))
    )
    reference_from_meta = meta_series.apply(
        lambda m: _extract_nested_float(
            m,
            (
                "reference_price",
                "ref_price",
                "mid_price",
                "mid",
                "mark_price",
                "quote_mid",
            ),
        )
    )
    modeled_cost_from_meta = meta_series.apply(
        lambda m: _extract_nested_float(
            m,
            (
                "modeled_cost_bps",
                "cost_bps",
                "expected_cost_bps",
                "slippage_bps",
            ),
        )
    )

    if "turnover_usd" in trades.columns:
        trades["turnover_usd"] = (
            pd.to_numeric(trades["turnover_usd"], errors="coerce").fillna(0.0)
        )
    else:
        trades["turnover_usd"] = turnover_from_meta

    if "cap_usd" in trades.columns:
        trades["cap_usd"] = pd.to_numeric(trades["cap_usd"], errors="coerce")
    else:
        trades["cap_usd"] = cap_from_meta

    if "adv_quote" in trades.columns:
        trades["adv_quote"] = pd.to_numeric(trades["adv_quote"], errors="coerce")
    else:
        trades["adv_quote"] = pd.to_numeric(adv_from_meta, errors="coerce")

    if "act_now" in trades.columns:
        trades["act_now_flag"] = trades["act_now"].apply(_normalize_bool_like)
    else:
        trades["act_now_flag"] = act_now_from_meta

    trades["act_now_flag"] = trades["act_now_flag"].astype(object)
    trades.loc[trades["execution_mode"] != "bar", "turnover_usd"] = 0.0
    trades.loc[trades["execution_mode"] != "bar", "cap_usd"] = float("nan")
    trades.loc[trades["execution_mode"] != "bar", "adv_quote"] = float("nan")
    trades.loc[trades["execution_mode"] != "bar", "act_now_flag"] = None

    # Execution cost diagnostics
    trades["modeled_cost_bps"] = pd.to_numeric(modeled_cost_from_meta, errors="coerce")
    reference_prices = pd.to_numeric(reference_from_meta, errors="coerce")
    trade_prices = pd.to_numeric(trades["price"], errors="coerce")
    quantity_source = trades.get("quantity")
    if quantity_source is None:
        quantity_source = pd.Series(index=trades.index, dtype=float)
    quantity_abs = pd.to_numeric(quantity_source, errors="coerce").abs()
    side_sign = trades["side_sign"].astype(float)
    with pd.option_context("mode.use_inf_as_na", True):
        realized = pd.Series(index=trades.index, dtype=float)
        valid_mask = (
            trade_prices.notna()
            & reference_prices.notna()
            & reference_prices.ne(0.0)
            & side_sign.notna()
            & quantity_abs.gt(0.0)
        )
        realized.loc[valid_mask] = (
            (trade_prices.loc[valid_mask] - reference_prices.loc[valid_mask])
            * side_sign.loc[valid_mask]
            / reference_prices.loc[valid_mask]
            * 10_000.0
        )
    # Fall back to existing slippage column when reference price is unavailable
    if "slippage_bps" in trades.columns:
        fallback_slip = pd.to_numeric(trades["slippage_bps"], errors="coerce")
        realized = realized.where(realized.notna(), fallback_slip)
    trades["realized_slippage_bps"] = realized
    trades["cost_bias_bps"] = trades["realized_slippage_bps"] - trades["modeled_cost_bps"]

    trades["bar_decision_count"] = trades["execution_mode"].eq("bar").astype(int)
    act_now_mask = trades["act_now_flag"].apply(
        lambda v: _normalize_bool_like(v) is True
    )
    trades["bar_act_now_count"] = (
        trades["execution_mode"].eq("bar") & act_now_mask
    ).astype(int)

    # Per-bar aggregation
    trades["ts_bucket"] = _bucket_ts_ms(trades["ts"], bar_seconds=bar_seconds)
    g = trades.groupby(["symbol","ts_bucket"], as_index=False)

    def _agg(df: pd.DataFrame) -> pd.Series:
        qty_abs_series = df["quantity"].abs()
        qty_abs = float(qty_abs_series.sum())
        notional = (df["price"] * qty_abs_series).sum()
        vwap = float(notional / qty_abs) if qty_abs and math.isfinite(notional) else float("nan")
        buy_qty = df.loc[df["side_sign"]>0, "quantity"].abs().sum()
        sell_qty = df.loc[df["side_sign"]<0, "quantity"].abs().sum()
        trade_mask = df["quantity"].abs() > 0
        n_trades = int(trade_mask.sum())
        fee_sum = float(pd.to_numeric(df["fee"], errors="coerce").fillna(0.0).sum()) if "fee" in df.columns else 0.0
        bar_decisions = int(df["bar_decision_count"].sum()) if "bar_decision_count" in df.columns else 0
        bar_act_now = int(df["bar_act_now_count"].sum()) if "bar_act_now_count" in df.columns else 0
        turnover_total = float(df.get("turnover_usd", pd.Series(dtype=float)).sum())
        cap_series = pd.to_numeric(
            df.get("cap_usd", pd.Series(dtype=float)), errors="coerce"
        )
        cap_values = cap_series[cap_series > 0]
        cap_value = float(cap_values.iloc[0]) if not cap_values.empty else float("nan")
        ratio = (
            float(turnover_total / cap_value)
            if cap_value > 0 and math.isfinite(cap_value)
            else float("nan")
        )
        adv_series = pd.to_numeric(
            df.get("adv_quote", pd.Series(dtype=float)), errors="coerce"
        )
        adv_values = adv_series[adv_series > 0]
        adv_value = float(adv_values.iloc[0]) if not adv_values.empty else float("nan")
        act_rate = float(bar_act_now / bar_decisions) if bar_decisions > 0 else float("nan")
        realized_series = pd.to_numeric(df.get("realized_slippage_bps"), errors="coerce")
        modeled_series = pd.to_numeric(df.get("modeled_cost_bps"), errors="coerce")

        def _weighted_average(series: pd.Series) -> float:
            mask = series.notna() & qty_abs_series.notna() & (qty_abs_series > 0)
            if not mask.any():
                return float("nan")
            weights = qty_abs_series.loc[mask].astype(float)
            total_weight = float(weights.sum())
            if not total_weight:
                return float("nan")
            return float((series.loc[mask].astype(float) * weights).sum() / total_weight)

        realized_avg = _weighted_average(realized_series)
        modeled_avg = _weighted_average(modeled_series)
        if math.isfinite(realized_avg) and math.isfinite(modeled_avg):
            bias_avg = float(realized_avg - modeled_avg)
        else:
            bias_avg = float("nan")
        return pd.Series({
            "volume": float(qty_abs),
            "buy_qty": float(buy_qty),
            "sell_qty": float(sell_qty),
            "trades": n_trades,
            "vwap": float(vwap),
            "fee_total": fee_sum,
            "bar_decisions": bar_decisions,
            "bar_act_now": bar_act_now,
            "bar_act_now_rate": act_rate,
            "bar_turnover_usd": turnover_total,
            "bar_cap_usd": float(cap_value) if cap_value > 0 else float("nan"),
            "bar_turnover_vs_cap": ratio,
            "bar_adv_quote": float(adv_value) if adv_value > 0 else float("nan"),
            "realized_slippage_bps": float(realized_avg),
            "modeled_cost_bps": float(modeled_avg),
            "cost_bias_bps": float(bias_avg),
        })

    bars = g.apply(_agg)
    bars = bars.rename(columns={"ts_bucket": "ts"})
    bars["ts"] = bars["ts"].astype("Int64")

    # Per-day aggregation (UTC days by ms timestamp)
    day_ms = 24*60*60*1000
    trades["day"] = (trades["ts"].astype("Int64") // day_ms) * day_ms
    gd = trades.groupby(["symbol","day"], as_index=False)
    def _agg_day(df: pd.DataFrame) -> pd.Series:
        qty_abs_series = df["quantity"].abs()
        qty_abs = float(qty_abs_series.sum())
        buy_qty = float(df.loc[df["side_sign"] > 0, "quantity"].abs().sum())
        sell_qty = float(df.loc[df["side_sign"] < 0, "quantity"].abs().sum())
        trades_count = int(len(df))
        notional = (df["price"] * qty_abs_series).sum()
        vwap = float(notional / qty_abs) if qty_abs and math.isfinite(notional) else float("nan")
        cap_series = pd.to_numeric(
            df.get("cap_usd", pd.Series(dtype=float)), errors="coerce"
        )
        cap_values = cap_series[cap_series > 0]
        cap_value = float(cap_values.iloc[0]) if not cap_values.empty else float("nan")
        adv_series = pd.to_numeric(
            df.get("adv_quote", pd.Series(dtype=float)), errors="coerce"
        )
        adv_values = adv_series[adv_series > 0]
        adv_value = float(adv_values.iloc[0]) if not adv_values.empty else float("nan")
        bar_decisions = float(df.get("bar_decision_count", pd.Series(dtype=float)).sum())
        bar_act_now = float(df.get("bar_act_now_count", pd.Series(dtype=float)).sum())
        turnover_total = float(df.get("turnover_usd", pd.Series(dtype=float)).sum())
        realized_series = pd.to_numeric(df.get("realized_slippage_bps"), errors="coerce")
        modeled_series = pd.to_numeric(df.get("modeled_cost_bps"), errors="coerce")

        def _weighted_average(series: pd.Series) -> float:
            mask = series.notna() & qty_abs_series.notna() & (qty_abs_series > 0)
            if not mask.any():
                return float("nan")
            weights = qty_abs_series.loc[mask].astype(float)
            total_weight = float(weights.sum())
            if not total_weight:
                return float("nan")
            return float((series.loc[mask].astype(float) * weights).sum() / total_weight)

        realized_avg = _weighted_average(realized_series)
        modeled_avg = _weighted_average(modeled_series)
        if math.isfinite(realized_avg) and math.isfinite(modeled_avg):
            bias_avg = float(realized_avg - modeled_avg)
        else:
            bias_avg = float("nan")
        return pd.Series({
            "volume": float(qty_abs),
            "trades": trades_count,
            "buy_qty": buy_qty,
            "sell_qty": sell_qty,
            "fee_total": float(pd.to_numeric(df["fee"], errors="coerce").fillna(0.0).sum()) if "fee" in df.columns else 0.0,
            "vwap": float(vwap),
            "bar_decisions": int(bar_decisions),
            "bar_act_now": int(bar_act_now),
            "bar_act_now_rate": float(bar_act_now / bar_decisions) if bar_decisions > 0 else float("nan"),
            "bar_turnover_usd": turnover_total,
            "bar_cap_usd": float(cap_value) if cap_value > 0 else float("nan"),
            "bar_turnover_vs_cap": float(turnover_total / cap_value)
            if cap_value > 0 and math.isfinite(cap_value)
            else float("nan"),
            "bar_adv_quote": float(adv_value) if adv_value > 0 else float("nan"),
            "realized_slippage_bps": float(realized_avg),
            "modeled_cost_bps": float(modeled_avg),
            "cost_bias_bps": float(bias_avg),
        })

    days = gd.apply(_agg_day)
    days = days.rename(columns={"day":"ts"})
    days["ts"] = days["ts"].astype("Int64")

    reports = _normalize_reports(read_any(reports_path)) if reports_path else _normalize_reports(pd.DataFrame())
    if not reports.empty:
        rep_sorted = reports.sort_values(["symbol", "ts_ms"])
        bars = pd.merge_asof(
            bars.sort_values(["symbol", "ts"]),
            rep_sorted,
            left_on="ts",
            right_on="ts_ms",
            by="symbol",
            direction="backward",
        ).drop(columns=["ts_ms"])
        days = pd.merge_asof(
            days.sort_values(["symbol", "ts"]),
            rep_sorted,
            left_on="ts",
            right_on="ts_ms",
            by="symbol",
            direction="backward",
        ).drop(columns=["ts_ms"])
        if equity_png:
            try:
                plot_equity_curve(reports, equity_png)
            except Exception:
                pass
    else:
        if equity_png:
            os.makedirs(os.path.dirname(equity_png) or ".", exist_ok=True)

    cost_summary: Dict[str, float] = {}
    if not trades.empty:
        weight_series = trades["quantity"].abs().astype(float)
        realized_series = pd.to_numeric(trades.get("realized_slippage_bps"), errors="coerce")
        modeled_series = pd.to_numeric(trades.get("modeled_cost_bps"), errors="coerce")

        def _safe_weighted_avg(series: pd.Series) -> float:
            mask = series.notna() & weight_series.notna() & (weight_series > 0)
            if not mask.any():
                return float("nan")
            weights = weight_series.loc[mask]
            total_weight = float(weights.sum())
            if not total_weight:
                return float("nan")
            values = series.loc[mask].astype(float)
            return float((values * weights).sum() / total_weight)

        realized_avg = _safe_weighted_avg(realized_series)
        modeled_avg = _safe_weighted_avg(modeled_series)
        if math.isfinite(realized_avg):
            cost_summary["realized_slippage_bps"] = realized_avg
        if math.isfinite(modeled_avg):
            cost_summary["modeled_cost_bps"] = modeled_avg
        if math.isfinite(realized_avg) and math.isfinite(modeled_avg):
            cost_summary["cost_bias_bps"] = realized_avg - modeled_avg

    if metrics_md:
        trades_for_metrics = trades.rename(columns={"quantity": "qty"})
        metrics = calculate_metrics(trades_for_metrics, reports)
        os.makedirs(os.path.dirname(metrics_md) or ".", exist_ok=True)

        def _iter_metric_blocks(payload: Any) -> list[tuple[str, Dict[str, Any]]]:
            if isinstance(payload, dict) and "equity" in payload and "trades" in payload:
                return [("", payload)]
            if isinstance(payload, dict):
                blocks: list[tuple[str, Dict[str, Any]]] = []
                for name, block in payload.items():
                    if isinstance(block, dict):
                        blocks.append((str(name), block))
                if blocks:
                    return blocks
            return [("", payload if isinstance(payload, dict) else {})]

        metric_blocks = _iter_metric_blocks(metrics)
        with open(metrics_md, "w", encoding="utf-8") as f:
            f.write("# Performance Metrics\n\n")
            for idx, (profile, payload) in enumerate(metric_blocks):
                if profile:
                    f.write(f"## Profile {profile}\n")
                    equity_heading = "### Equity"
                    trades_heading = "### Trades"
                else:
                    equity_heading = "## Equity"
                    trades_heading = "## Trades"
                f.write(f"{equity_heading}\n")
                for k, v in (payload.get("equity") or {}).items():
                    f.write(f"- **{k}**: {v}\n")
                f.write(f"\n{trades_heading}\n")
                for k, v in (payload.get("trades") or {}).items():
                    f.write(f"- **{k}**: {v}\n")
                if idx < len(metric_blocks) - 1:
                    f.write("\n")
            if cost_summary:
                f.write("\n## Execution Costs\n")
                for key, value in cost_summary.items():
                    if math.isfinite(value):
                        f.write(f"- **{key}**: {value:.3f}\n")

    os.makedirs(os.path.dirname(out_bars) or ".", exist_ok=True)
    bars.to_csv(out_bars, index=False)
    os.makedirs(os.path.dirname(out_days) or ".", exist_ok=True)
    days.to_csv(out_days, index=False)
    return out_bars, out_days


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate execution logs into per-bar and per-day summaries.")
    p.add_argument("--trades", required=True, help="Path or glob to unified Exec logs (log_trades_*.csv). Legacy trades.csv is supported but deprecated.")
    p.add_argument(
        "--reports",
        default="",
        help="Optional path or glob to equity reports (report_equity_*.csv)",
    )
    p.add_argument("--out-bars", default="logs/agg_bars.csv", help="Output CSV path for per-bar aggregation")
    p.add_argument("--out-days", default="logs/agg_days.csv", help="Output CSV path for per-day aggregation")
    p.add_argument("--bar-seconds", type=int, default=60, help="Bar length in seconds (default: 60)")
    p.add_argument("--equity-png", default="", help="Optional path to save equity curve PNG")
    p.add_argument("--metrics-md", default="", help="Optional path to save metrics summary in Markdown")
    args = p.parse_args()

    try:
        aggregate(
            args.trades,
            args.reports,
            args.out_bars,
            args.out_days,
            bar_seconds=int(args.bar_seconds),
            equity_png=args.equity_png,
            metrics_md=args.metrics_md,
        )
    except ValueError as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
