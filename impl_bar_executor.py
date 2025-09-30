# -*- coding: utf-8 -*-
"""Bar-based execution helper utilities.

This module provides a light-weight :class:`TradeExecutor` implementation that
operates on bar data instead of full order-book snapshots.  The executor keeps
track of portfolio weights per symbol, accepts high-level rebalance payloads and
produces deterministic schedules describing how the rebalance should be
performed (e.g. single-shot or TWAP style).  No actual fills are emitted –
callers are expected to interpret the returned instructions and account for the
resulting state updates on their end.

Two payload flavours are supported:

``target_weight``
    Absolute target for the next bar (expressed as a fraction of portfolio
    equity).  The executor computes the required delta vs the current weight,
    enforces ``min_rebalance_step`` and emits a single rebalance instruction by
    default.

``delta_weight``
    Relative adjustment to the current weight.  Optional TWAP configuration and
    participation caps can be provided and are mapped onto deterministic child
    instructions.

The helper :func:`decide_spot_trade` encapsulates the simple spot-market cost
model used by the executor.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from api.spot_signals import SpotSignalEconomics, SpotSignalEnvelope
from core_config import (
    ResolvedTurnoverCaps,
    ResolvedTurnoverLimit,
    SpotCostConfig,
    SpotTurnoverCaps,
)
from core_contracts import TradeExecutor
from core_models import (
    Bar,
    ExecReport,
    ExecStatus,
    Liquidity,
    Order,
    OrderType,
    Position,
    Side,
)


logger = logging.getLogger(__name__)


def _as_decimal(value: float | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ensure_cost_config(cost_config: SpotCostConfig | Mapping[str, Any] | None) -> SpotCostConfig:
    if cost_config is None:
        return SpotCostConfig()
    if isinstance(cost_config, SpotCostConfig):
        return cost_config
    try:
        return SpotCostConfig.parse_obj(cost_config)
    except Exception:  # pragma: no cover - defensive fallback
        logger.exception("Failed to parse cost_config, falling back to defaults")
        return SpotCostConfig()


def _ensure_turnover_caps(
    turnover_caps: SpotTurnoverCaps | Mapping[str, Any] | None,
) -> SpotTurnoverCaps:
    if turnover_caps is None:
        return SpotTurnoverCaps()
    if isinstance(turnover_caps, SpotTurnoverCaps):
        return turnover_caps
    try:
        return SpotTurnoverCaps.parse_obj(turnover_caps)
    except Exception:  # pragma: no cover - defensive fallback
        logger.exception("Failed to parse turnover_caps, falling back to defaults")
        return SpotTurnoverCaps()


@dataclass
class PortfolioState:
    symbol: str
    weight: float = 0.0
    equity_usd: float = 0.0
    price: Decimal = Decimal("0")
    ts: int = 0

    def with_bar(self, bar: Bar, price: Decimal) -> "PortfolioState":
        return replace(self, price=price, ts=bar.ts)


@dataclass
class SymbolSpec:
    min_notional: Decimal = Decimal("0")
    step_size: Decimal = Decimal("0")
    tick_size: Decimal = Decimal("0")


@dataclass
class RebalanceInstruction:
    symbol: str
    ts: int
    slice_index: int
    slices_total: int
    target_weight: float
    delta_weight: float
    notional_usd: float
    quantity: Decimal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ts": self.ts,
            "slice_index": self.slice_index,
            "slices_total": self.slices_total,
            "target_weight": self.target_weight,
            "delta_weight": self.delta_weight,
            "notional_usd": self.notional_usd,
            "quantity": float(self.quantity),
        }


def decide_spot_trade(
    signal: Mapping[str, Any],
    portfolio_state: PortfolioState,
    cost_config: SpotCostConfig,
    adv_quote: Optional[float],
    safety_margin_bps: float = 0.0,
) -> SpotSignalEconomics:
    """Evaluate whether the current signal justifies trading.

    Parameters
    ----------
    signal:
        Mapping describing the desired rebalance.  Expected keys include
        ``edge_bps`` and either ``target_weight`` or ``delta_weight``.  When
        both are present, ``target_weight`` takes precedence.
    portfolio_state:
        Current portfolio snapshot for the symbol.
    cost_config:
        Spot execution cost assumptions.
    adv_quote:
        Average daily volume proxy expressed in quote currency.  ``None`` or
        non-positive values disable the participation-based impact model.
    safety_margin_bps:
        Additional buffer subtracted from the signal edge before deciding to
        trade.

    Returns
    -------
    dict
        Dictionary containing ``edge_bps``, ``cost_bps``, ``net_bps``,
        ``turnover_usd`` and ``act_now``.
    """

    edge_bps = float(signal.get("edge_bps") or 0.0)

    if "target_weight" in signal:
        raw_target = float(signal.get("target_weight") or 0.0)
        target_weight = _clamp_01(raw_target)
    elif "delta_weight" in signal:
        delta_weight = float(signal.get("delta_weight") or 0.0)
        target_weight = _clamp_01(portfolio_state.weight + delta_weight)
    else:
        target_weight = portfolio_state.weight

    delta_weight = target_weight - portfolio_state.weight
    turnover_usd = abs(delta_weight) * float(portfolio_state.equity_usd)

    base_cost = float(cost_config.taker_fee_bps) + float(cost_config.half_spread_bps)

    participation = 0.0
    impact = 0.0
    impact_mode = "model"
    if adv_quote is not None and adv_quote > 0.0:
        participation = turnover_usd / float(adv_quote)
        if participation > 0.0:
            sqrt_coeff = float(cost_config.impact.sqrt_coeff)
            linear_coeff = float(cost_config.impact.linear_coeff)
            impact += sqrt_coeff * math.sqrt(participation)
            impact += linear_coeff * participation
            power_coeff = float(
                getattr(cost_config.impact, "power_coefficient", 0.0) or 0.0
            )
            power_exp = float(getattr(cost_config.impact, "power_exponent", 1.0) or 1.0)
            if power_coeff > 0.0 and participation > 0.0:
                if power_exp <= 0.0:
                    power_exp = 1.0
                impact += power_coeff * participation ** power_exp
    else:
        impact_mode = "none"

    cost_bps = base_cost + impact
    net_bps = edge_bps - cost_bps - float(safety_margin_bps)
    act_now = net_bps > 0.0 and turnover_usd > 0.0

    return SpotSignalEconomics(
        edge_bps=edge_bps,
        cost_bps=cost_bps,
        net_bps=net_bps,
        turnover_usd=turnover_usd,
        act_now=act_now,
        impact=impact,
        impact_mode=impact_mode,
    )


class BarExecutor(TradeExecutor):
    """Simple portfolio-weight executor operating on bar data."""

    def __init__(
        self,
        *,
        run_id: str = "bar",
        bar_price: str = "close",
        min_rebalance_step: float = 0.0,
        cost_config: SpotCostConfig | Mapping[str, Any] | None = None,
        safety_margin_bps: float = 0.0,
        max_participation: Optional[float] = None,
        turnover_caps: SpotTurnoverCaps | Mapping[str, Any] | None = None,
        default_equity_usd: float = 0.0,
        initial_weights: Optional[Mapping[str, float]] = None,
        symbol_specs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        self.run_id = run_id
        self.bar_price_field = str(bar_price or "close")
        self.min_rebalance_step = max(0.0, float(min_rebalance_step))
        self.cost_config = _ensure_cost_config(cost_config)
        self.safety_margin_bps = float(safety_margin_bps)
        if max_participation is not None:
            try:
                max_participation = float(max_participation)
            except (TypeError, ValueError):
                max_participation = None
        if max_participation is not None:
            if not math.isfinite(max_participation) or max_participation <= 0.0:
                max_participation = None
        self.max_participation = max_participation
        caps_source: SpotTurnoverCaps | Mapping[str, Any] | None = turnover_caps
        if caps_source is None:
            caps_source = getattr(self.cost_config, "turnover_caps", None)
        self.turnover_caps = _ensure_turnover_caps(caps_source)
        self._resolved_turnover_caps: ResolvedTurnoverCaps = self.turnover_caps.resolve()
        self.default_equity_usd = float(default_equity_usd)
        self._states: Dict[str, PortfolioState] = {}
        self._last_snapshot: Dict[str, Any] = {}
        self._symbol_turnover: Dict[str, Dict[str, Any]] = {}
        self._portfolio_turnover: Dict[str, Any] = {"ts": None, "total": 0.0}
        self._symbol_specs: Dict[str, SymbolSpec] = self._normalize_symbol_specs(
            symbol_specs
        )
        if initial_weights:
            for symbol, weight in initial_weights.items():
                self._states[symbol] = PortfolioState(
                    symbol=symbol,
                    weight=_clamp_01(float(weight)),
                    equity_usd=self.default_equity_usd,
                )

    # ------------------------------------------------------------------
    # TradeExecutor protocol
    # ------------------------------------------------------------------
    def execute(self, order: Order) -> ExecReport:
        symbol = order.symbol
        state = self._states.get(symbol)
        if state is None:
            state = PortfolioState(symbol=symbol, equity_usd=self.default_equity_usd)

        payload = self._extract_payload(order.meta)
        meta_map = order.meta if isinstance(order.meta, Mapping) else {}
        normalized_flag = self._coerce_bool(payload.get("normalized"))
        if not normalized_flag and meta_map:
            normalized_flag = self._coerce_bool(meta_map.get("normalized"))
        normalization_data = self._materialize_mapping(payload.get("normalization"))
        if not normalization_data and meta_map:
            normalization_data = self._materialize_mapping(meta_map.get("normalization"))
        if normalization_data:
            normalized_flag = True
        adv_quote = self._coerce_float(order.meta.get("adv_quote"))
        bar = self._extract_bar(order.meta)

        skip_reason: Optional[str] = None

        if bar is not None:
            price = self._select_bar_price(bar)
            state = state.with_bar(bar, price)
        else:
            price = state.price

        if price is None or price <= Decimal("0"):
            skip_reason = "no_price"

        equity_override = self._coerce_float(order.meta.get("equity_usd"))
        if equity_override is not None:
            state = replace(state, equity_usd=equity_override)

        target_weight, mode, delta_weight = self._resolve_target_weight(state, payload)
        requested_target_weight = target_weight
        requested_delta_weight = delta_weight

        decision_signal: Dict[str, Any] = dict(payload)
        if normalized_flag:
            decision_signal["normalized"] = True
        if normalization_data:
            decision_signal["normalization"] = dict(normalization_data)
        decision_signal.setdefault("target_weight", target_weight)
        metrics = decide_spot_trade(
            decision_signal,
            state,
            self.cost_config,
            adv_quote,
            self.safety_margin_bps,
        )

        if skip_reason is not None:
            if hasattr(metrics, "model_copy"):
                metrics = metrics.model_copy(update={"act_now": False})
            else:  # pragma: no cover - compatibility fallback
                metrics = metrics.copy(update={"act_now": False})

        min_step = self.min_rebalance_step
        skip_due_to_step = False
        raw_turnover_usd = float(getattr(metrics, "turnover_usd", 0.0))
        turnover_usd = raw_turnover_usd
        if min_step > 0.0 and abs(delta_weight) < min_step:
            skip_due_to_step = True
            if hasattr(metrics, "model_copy"):
                metrics = metrics.model_copy(
                    update={"act_now": False, "turnover_usd": 0.0}
                )
            else:  # pragma: no cover - compatibility fallback
                metrics = metrics.copy(update={"act_now": False, "turnover_usd": 0.0})
            turnover_usd = 0.0

        caps_eval = self._evaluate_turnover_caps(symbol, state, bar)
        skip_due_to_cap = False
        effective_cap = caps_eval.get("effective_cap")
        if effective_cap is not None and raw_turnover_usd > float(effective_cap) + 1e-9:
            skip_due_to_cap = True
            if hasattr(metrics, "model_copy"):
                metrics = metrics.model_copy(
                    update={"act_now": False, "turnover_usd": 0.0}
                )
            else:  # pragma: no cover - compatibility fallback
                metrics = metrics.copy(update={"act_now": False, "turnover_usd": 0.0})
            turnover_usd = 0.0

        instructions: List[RebalanceInstruction] = []
        final_state = state

        if (
            metrics.act_now
            and not skip_due_to_step
            and not skip_due_to_cap
            and skip_reason is None
        ):
            (
                instructions,
                executed_target_weight,
                executed_turnover_usd,
                spec_reason,
            ) = self._build_instructions(
                state=state,
                target_weight=target_weight,
                delta_weight=delta_weight,
                payload=payload,
                bar=bar,
                adv_quote=adv_quote,
            )
            if spec_reason is not None:
                skip_reason = spec_reason
                turnover_usd = float(executed_turnover_usd)
                if hasattr(metrics, "model_copy"):
                    metrics = metrics.model_copy(
                        update={"act_now": False, "turnover_usd": turnover_usd}
                    )
                else:  # pragma: no cover - compatibility fallback
                    metrics = metrics.copy(
                        update={"act_now": False, "turnover_usd": turnover_usd}
                    )
                instructions = []
                target_weight = state.weight
                delta_weight = 0.0
            else:
                turnover_usd = float(executed_turnover_usd)
                if hasattr(metrics, "model_copy"):
                    metrics = metrics.model_copy(update={"turnover_usd": turnover_usd})
                else:  # pragma: no cover - compatibility fallback
                    metrics = metrics.copy(update={"turnover_usd": turnover_usd})
                delta_weight = executed_target_weight - state.weight
                target_weight = executed_target_weight
                final_state = replace(state, weight=executed_target_weight)
                self._register_turnover(symbol, caps_eval.get("ts"), turnover_usd)
                if caps_eval.get("symbol_remaining") is not None:
                    caps_eval["symbol_remaining"] = max(
                        0.0, float(caps_eval["symbol_remaining"]) - turnover_usd
                    )
                if caps_eval.get("portfolio_remaining") is not None:
                    caps_eval["portfolio_remaining"] = max(
                        0.0, float(caps_eval["portfolio_remaining"]) - turnover_usd
                    )
                remaining_candidates = [
                    value
                    for value in (
                        caps_eval.get("symbol_remaining"),
                        caps_eval.get("portfolio_remaining"),
                    )
                    if value is not None
                ]
                if remaining_candidates:
                    caps_eval["effective_cap"] = min(remaining_candidates)
        else:
            dump_fn = getattr(metrics, "model_dump", None)
            metrics_data = dump_fn() if callable(dump_fn) else metrics.dict()
            if skip_reason is not None:
                metrics_data["reason"] = skip_reason
            logger.debug(
                "Skipping rebalance for %s (mode=%s, delta=%.6f, metrics=%s)",
                symbol,
                mode,
                delta_weight,
                metrics_data,
            )

        self._states[symbol] = final_state

        dump_fn = getattr(metrics, "model_dump", None)
        decision_data = dump_fn() if callable(dump_fn) else metrics.dict()
        if normalized_flag:
            decision_data["normalized"] = True
        if normalization_data:
            decision_data["normalization"] = dict(normalization_data)
        if skip_reason is not None:
            decision_data["reason"] = skip_reason
        report_meta: Dict[str, Any] = {
            "mode": mode,
            "decision": decision_data,
            "target_weight": target_weight,
            "delta_weight": delta_weight,
            "instructions": [instr.to_dict() for instr in instructions],
        }
        report_meta["requested_target_weight"] = requested_target_weight
        report_meta["requested_delta_weight"] = requested_delta_weight
        if skip_due_to_step:
            report_meta["min_step_enforced"] = True
        if skip_due_to_cap:
            report_meta["turnover_cap_enforced"] = True
        if skip_reason is not None:
            report_meta["reason"] = skip_reason
        if bar is not None:
            report_meta["bar_ts"] = bar.ts
        if price is not None:
            report_meta["reference_price"] = float(price)
        if adv_quote is not None:
            report_meta["adv_quote"] = adv_quote
        cap_effective = caps_eval.get("effective_cap")
        if cap_effective is not None:
            report_meta["cap_usd"] = float(cap_effective)
        if caps_eval.get("symbol_limit") is not None:
            report_meta["symbol_turnover_cap_usd"] = float(caps_eval["symbol_limit"])
        if caps_eval.get("portfolio_limit") is not None:
            report_meta["portfolio_turnover_cap_usd"] = float(
                caps_eval["portfolio_limit"]
            )
        if caps_eval.get("symbol_remaining") is not None:
            report_meta["symbol_turnover_remaining_usd"] = float(
                caps_eval["symbol_remaining"]
            )
        if caps_eval.get("portfolio_remaining") is not None:
            report_meta["portfolio_turnover_remaining_usd"] = float(
                caps_eval["portfolio_remaining"]
            )
        if normalized_flag:
            report_meta["normalized"] = True
        if normalization_data:
            report_meta["normalization"] = dict(normalization_data)

        snapshot: Dict[str, Any] = {
            "execution_mode": "bar",
            "mode": mode,
            "target_weight": target_weight,
            "delta_weight": delta_weight,
            "decision": decision_data,
            "act_now": bool(getattr(metrics, "act_now", False)),
            "turnover_usd": float(getattr(metrics, "turnover_usd", 0.0)),
            "adv_quote": float(adv_quote) if adv_quote is not None else None,
            "min_step_enforced": bool(skip_due_to_step),
        }
        snapshot["impact"] = float(getattr(metrics, "impact", 0.0))
        impact_mode_value = getattr(metrics, "impact_mode", None)
        if impact_mode_value is not None:
            snapshot["impact_mode"] = impact_mode_value
        snapshot["requested_target_weight"] = requested_target_weight
        snapshot["requested_delta_weight"] = requested_delta_weight
        if instructions:
            snapshot["instructions"] = [instr.to_dict() for instr in instructions]
        if bar is not None:
            snapshot["bar_ts"] = int(bar.ts)
        snapshot["normalized"] = bool(normalized_flag)
        if normalization_data:
            snapshot["normalization"] = dict(normalization_data)
        if skip_reason is not None:
            snapshot["reason"] = skip_reason
        if cap_effective is not None:
            snapshot["cap_usd"] = float(cap_effective)
        if caps_eval.get("symbol_limit") is not None:
            snapshot["symbol_cap_usd"] = float(caps_eval["symbol_limit"])
        if caps_eval.get("portfolio_limit") is not None:
            snapshot["portfolio_cap_usd"] = float(caps_eval["portfolio_limit"])
        if caps_eval.get("symbol_remaining") is not None:
            snapshot["symbol_cap_remaining_usd"] = float(caps_eval["symbol_remaining"])
        if caps_eval.get("portfolio_remaining") is not None:
            snapshot["portfolio_cap_remaining_usd"] = float(
                caps_eval["portfolio_remaining"]
            )
        if skip_due_to_cap:
            snapshot["turnover_cap_enforced"] = True
        self._last_snapshot = snapshot

        return ExecReport(
            ts=bar.ts if bar is not None else order.ts,
            run_id=self.run_id,
            symbol=symbol,
            side=order.side if isinstance(order.side, Side) else Side.BUY,
            order_type=order.order_type if isinstance(order.order_type, OrderType) else OrderType.MARKET,
            price=Decimal("0"),
            quantity=Decimal("0"),
            fee=Decimal("0"),
            fee_asset=None,
            exec_status=ExecStatus.CANCELED,
            liquidity=Liquidity.UNKNOWN,
            client_order_id=order.client_order_id,
            meta=report_meta,
        )

    # ------------------------------------------------------------------
    # Monitoring helpers
    def monitoring_snapshot(self) -> Dict[str, Any]:  # pragma: no cover - simple access
        if not self._last_snapshot:
            return {"execution_mode": "bar"}
        return dict(self._last_snapshot)

    def cancel(self, client_order_id: str) -> None:  # pragma: no cover - no-op
        logger.debug("cancel() called on BarExecutor for %s", client_order_id)

    def get_open_positions(self, symbols: Optional[Iterable[str]] = None) -> MutableMapping[str, Position]:
        if symbols is None:
            symbols = self._states.keys()
        result: Dict[str, Position] = {}
        for symbol in symbols:
            state = self._states.get(symbol)
            if state is None:
                continue
            price = state.price if state.price != Decimal("0") else Decimal("0")
            if price != Decimal("0"):
                qty_value = (Decimal(str(state.weight)) * Decimal(str(state.equity_usd))) / price
            else:
                qty_value = Decimal("0")
            position = Position(
                symbol=symbol,
                qty=qty_value,
                avg_entry_price=price if price != Decimal("0") else Decimal("0"),
                realized_pnl=Decimal("0"),
                fee_paid=Decimal("0"),
                ts=state.ts or None,
                meta={
                    "weight": state.weight,
                    "equity_usd": state.equity_usd,
                },
            )
            result[symbol] = position
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_payload(self, meta: Mapping[str, Any]) -> Dict[str, Any]:
        payload = meta.get("payload")
        if isinstance(payload, SpotSignalEnvelope):
            inner = payload.payload
            dump = getattr(inner, "model_dump", None)
            if callable(dump):
                return dict(dump())
            if hasattr(inner, "dict"):
                return dict(inner.dict())
        dump = getattr(payload, "model_dump", None)
        if callable(dump):
            try:
                return dict(dump())
            except Exception:
                pass
        if hasattr(payload, "dict"):
            try:
                return dict(payload.dict())
            except Exception:
                pass
        if isinstance(payload, Mapping):
            return dict(payload)
        rebalance_payload = meta.get("rebalance")
        if isinstance(rebalance_payload, Mapping):
            return dict(rebalance_payload)
        return {}

    def _materialize_mapping(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        for attr in ("model_dump", "dict"):
            getter = getattr(value, attr, None)
            if callable(getter):
                try:
                    data = getter()
                except Exception:
                    continue
                if isinstance(data, Mapping):
                    return dict(data)
        return {}

    def _coerce_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalize_symbol_specs(
        self, payload: Optional[Mapping[str, Mapping[str, Any]]]
    ) -> Dict[str, SymbolSpec]:
        specs: Dict[str, SymbolSpec] = {}
        if not payload:
            return specs
        for raw_symbol, raw_spec in payload.items():
            symbol_text = str(raw_symbol or "").strip().upper()
            if not symbol_text:
                continue
            spec_payload = self._materialize_mapping(raw_spec)
            if not spec_payload:
                continue
            min_notional = self._extract_spec_decimal(
                spec_payload,
                ("min_notional", "minNotional", "MIN_NOTIONAL"),
            )
            step_size = self._extract_spec_decimal(
                spec_payload,
                ("step_size", "stepSize", "lot_step", "LOT_STEP"),
            )
            tick_size = self._extract_spec_decimal(
                spec_payload,
                ("tick_size", "tickSize", "price_tick", "PRICE_TICK"),
            )
            specs[symbol_text] = SymbolSpec(
                min_notional=min_notional,
                step_size=step_size,
                tick_size=tick_size,
            )
        return specs

    def _extract_spec_decimal(
        self, payload: Mapping[str, Any], keys: Iterable[str], depth: int = 0
    ) -> Decimal:
        if depth > 4:
            return Decimal("0")
        key_set = {str(key).strip().lower() for key in keys if str(key).strip()}
        for raw_key, raw_value in payload.items():
            key_text = str(raw_key).strip().lower()
            if key_text in key_set and raw_value is not None:
                try:
                    value = Decimal(str(raw_value))
                except Exception:
                    value = Decimal("0")
                if value < Decimal("0"):
                    return Decimal("0")
                return value
        for raw_value in payload.values():
            if isinstance(raw_value, Mapping):
                value = self._extract_spec_decimal(raw_value, keys, depth + 1)
                if value != Decimal("0"):
                    return value
            elif isinstance(raw_value, (list, tuple, set)):
                for item in raw_value:
                    if isinstance(item, Mapping):
                        value = self._extract_spec_decimal(item, keys, depth + 1)
                        if value != Decimal("0"):
                            return value
        return Decimal("0")

    def _evaluate_turnover_caps(
        self, symbol: str, state: PortfolioState, bar: Optional[Bar]
    ) -> Dict[str, Optional[float]]:
        caps: ResolvedTurnoverCaps = self._resolved_turnover_caps
        equity = float(state.equity_usd)
        symbol_limit = caps.per_symbol.limit_for_equity(equity)
        portfolio_limit = caps.portfolio.limit_for_equity(equity)
        current_ts: Optional[int]
        if bar is not None and getattr(bar, "ts", None) is not None:
            current_ts = int(bar.ts)
        else:
            current_ts = int(state.ts) if state.ts is not None else None
        tracker = self._symbol_turnover.get(symbol)
        if tracker is None or tracker.get("ts") != current_ts:
            tracker = {"ts": current_ts, "total": 0.0}
            self._symbol_turnover[symbol] = tracker
        used_symbol = float(tracker.get("total", 0.0) or 0.0)
        symbol_remaining = (
            None if symbol_limit is None else max(0.0, symbol_limit - used_symbol)
        )
        portfolio_tracker = self._portfolio_turnover
        stored_ts = portfolio_tracker.get("ts")
        if stored_ts != current_ts:
            portfolio_tracker["ts"] = current_ts
            portfolio_tracker["total"] = 0.0
        used_portfolio = float(portfolio_tracker.get("total", 0.0) or 0.0)
        portfolio_remaining = (
            None if portfolio_limit is None else max(0.0, portfolio_limit - used_portfolio)
        )
        candidates = [
            value
            for value in (symbol_remaining, portfolio_remaining)
            if value is not None
        ]
        effective_cap = min(candidates) if candidates else None
        return {
            "ts": current_ts,
            "symbol_limit": symbol_limit,
            "symbol_remaining": symbol_remaining,
            "portfolio_limit": portfolio_limit,
            "portfolio_remaining": portfolio_remaining,
            "effective_cap": effective_cap,
        }

    def _register_turnover(self, symbol: str, ts: Optional[int], turnover_usd: float) -> None:
        if turnover_usd <= 0.0:
            return
        symbol_tracker = self._symbol_turnover.setdefault(
            symbol, {"ts": ts, "total": 0.0}
        )
        if symbol_tracker.get("ts") != ts:
            symbol_tracker["ts"] = ts
            symbol_tracker["total"] = 0.0
        symbol_tracker["total"] = float(symbol_tracker.get("total", 0.0) or 0.0) + float(
            turnover_usd
        )
        portfolio_tracker = self._portfolio_turnover
        if portfolio_tracker.get("ts") != ts:
            portfolio_tracker["ts"] = ts
            portfolio_tracker["total"] = 0.0
        portfolio_tracker["total"] = float(
            portfolio_tracker.get("total", 0.0) or 0.0
        ) + float(turnover_usd)

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, str):
            candidate = value.strip().lower()
            if candidate in {"", "0", "false", "no", "off"}:
                return False
            if candidate in {"1", "true", "yes", "on"}:
                return True
        return bool(value)

    def _extract_bar(self, meta: Mapping[str, Any]) -> Optional[Bar]:
        bar_value = meta.get("bar")
        if isinstance(bar_value, Bar):
            return bar_value
        if isinstance(bar_value, Mapping):
            try:
                return Bar.from_dict(bar_value)
            except Exception:
                logger.exception("Failed to coerce bar payload")
        return None

    def _select_bar_price(self, bar: Bar) -> Decimal:
        price_value = getattr(bar, self.bar_price_field, None)
        if price_value is None:
            logger.debug(
                "Bar does not contain '%s', falling back to close price", self.bar_price_field
            )
            price_value = bar.close
        return _as_decimal(price_value)

    def _resolve_target_weight(
        self, state: PortfolioState, payload: Mapping[str, Any]
    ) -> tuple[float, str, float]:
        current_weight = state.weight
        mode = "none"
        if "target_weight" in payload or "target" in payload or "weight" in payload:
            raw_value = (
                payload.get("target_weight")
                if payload.get("target_weight") is not None
                else payload.get("target")
            )
            if raw_value is None:
                raw_value = payload.get("weight")
            try:
                target_weight = _clamp_01(float(raw_value))
            except (TypeError, ValueError):
                target_weight = current_weight
            mode = "target"
        elif "delta_weight" in payload or "delta" in payload:
            raw_delta = payload.get("delta_weight")
            if raw_delta is None:
                raw_delta = payload.get("delta")
            try:
                delta = float(raw_delta)
            except (TypeError, ValueError):
                delta = 0.0
            unclamped_target = current_weight + delta
            target_weight = _clamp_01(unclamped_target)
            mode = "delta"
        else:
            target_weight = current_weight
        delta_weight = target_weight - current_weight
        return target_weight, mode, delta_weight

    def _build_instructions(
        self,
        *,
        state: PortfolioState,
        target_weight: float,
        delta_weight: float,
        payload: Mapping[str, Any],
        bar: Optional[Bar],
        adv_quote: Optional[float],
    ) -> tuple[List[RebalanceInstruction], float, float, Optional[str]]:
        requested_notional = abs(delta_weight) * float(state.equity_usd)
        if requested_notional <= 0.0:
            return [], float(state.weight), 0.0, None

        spec = self._symbol_specs.get(state.symbol.upper())
        min_notional = spec.min_notional if spec is not None else Decimal("0")
        step_size = spec.step_size if spec is not None else Decimal("0")

        twap_cfg: Mapping[str, Any] = {}
        twap_value = payload.get("twap")
        if isinstance(twap_value, Mapping):
            twap_cfg = twap_value

        parts = 1
        if "parts" in twap_cfg:
            try:
                parts = max(1, int(twap_cfg.get("parts")))
            except (TypeError, ValueError):
                parts = 1
        elif payload.get("twap_parts") is not None:
            try:
                parts = max(1, int(payload.get("twap_parts")))
            except (TypeError, ValueError):
                parts = 1

        max_participation = self._coerce_float(
            twap_cfg.get("max_participation", payload.get("max_participation"))
        )
        if max_participation is None and self.max_participation is not None:
            max_participation = self.max_participation
        if max_participation is not None and max_participation > 0.0 and adv_quote and adv_quote > 0.0:
            max_slice_notional = adv_quote * max_participation
            if max_slice_notional > 0.0:
                required_parts = math.ceil(requested_notional / max_slice_notional)
                parts = max(parts, required_parts)

        if parts <= 0:
            parts = 1

        interval_ms = None
        if "interval_ms" in twap_cfg:
            try:
                interval_ms = int(twap_cfg.get("interval_ms"))
            except (TypeError, ValueError):
                interval_ms = None
        if interval_ms is None and "interval_s" in twap_cfg:
            try:
                interval_ms = int(float(twap_cfg.get("interval_s")) * 1000)
            except (TypeError, ValueError):
                interval_ms = None
        if interval_ms is None and payload.get("twap_interval_ms") is not None:
            try:
                interval_ms = int(payload.get("twap_interval_ms"))
            except (TypeError, ValueError):
                interval_ms = None
        if interval_ms is None and payload.get("twap_interval_s") is not None:
            try:
                interval_ms = int(float(payload.get("twap_interval_s")) * 1000)
            except (TypeError, ValueError):
                interval_ms = None
        if interval_ms is None:
            interval_ms = 0

        instructions: List[RebalanceInstruction] = []
        price = state.price if state.price != Decimal("0") else Decimal("0")
        equity_dec = Decimal(str(state.equity_usd))
        per_weight = delta_weight / float(parts)
        accumulated_weight = float(state.weight)
        total_notional_dec = Decimal("0")
        weight_tolerance = 1e-9

        for idx in range(parts):
            if idx == parts - 1:
                desired_delta = target_weight - accumulated_weight
            else:
                desired_delta = per_weight
            direction = 1 if desired_delta >= 0.0 else -1
            desired_abs = abs(desired_delta)
            if price != Decimal("0") and desired_abs > 0.0:
                desired_notional = Decimal(str(desired_abs)) * equity_dec
                desired_qty = desired_notional / price
            else:
                desired_qty = Decimal("0")
            quantized_qty = desired_qty
            if step_size > Decimal("0") and desired_qty > Decimal("0"):
                try:
                    quantized_qty = (
                        (desired_qty / step_size).to_integral_value(rounding=ROUND_DOWN)
                        * step_size
                    )
                except Exception:
                    quantized_qty = Decimal("0")
                if quantized_qty < Decimal("0"):
                    quantized_qty = Decimal("0")
            executed_notional = Decimal("0")
            if price > Decimal("0") and quantized_qty > Decimal("0"):
                executed_notional = quantized_qty * price
            executed_delta = 0.0
            if equity_dec > Decimal("0") and executed_notional > Decimal("0"):
                executed_fraction = executed_notional / equity_dec
                executed_delta = float(executed_fraction)
            if direction < 0:
                executed_delta = -executed_delta
            prev_weight = accumulated_weight
            candidate_weight = prev_weight + executed_delta
            if candidate_weight < -weight_tolerance:
                return [], float(state.weight), 0.0, "rounded_weight_below_zero"
            if candidate_weight > 1.0 + weight_tolerance:
                return [], float(state.weight), 0.0, "rounded_weight_above_one"
            new_weight = min(max(candidate_weight, 0.0), 1.0)
            adjusted_delta = new_weight - prev_weight
            ts = state.ts if bar is None else bar.ts
            if interval_ms and bar is not None:
                ts = int(bar.ts + idx * interval_ms)
            total_notional_dec += executed_notional
            instructions.append(
                RebalanceInstruction(
                    symbol=state.symbol,
                    ts=ts,
                    slice_index=idx,
                    slices_total=parts,
                    target_weight=new_weight,
                    delta_weight=adjusted_delta,
                    notional_usd=float(executed_notional),
                    quantity=quantized_qty,
                )
            )
            accumulated_weight = new_weight

        total_notional = float(total_notional_dec)
        if total_notional <= 0.0:
            return [], float(state.weight), 0.0, None

        if min_notional > Decimal("0"):
            tolerance = Decimal("1e-9")
            if total_notional_dec + tolerance < min_notional:
                return [], float(state.weight), 0.0, "below_min_notional"

        return instructions, accumulated_weight, total_notional, None


__all__ = ["BarExecutor", "PortfolioState", "RebalanceInstruction", "decide_spot_trade"]

