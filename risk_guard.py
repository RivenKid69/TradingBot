# risk_guard.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import IntEnum
import math
from typing import Optional, Deque, Tuple, Dict, Any, TYPE_CHECKING, Sequence, Callable, Mapping
from collections import deque
from collections.abc import Mapping as MappingABC
from clock import now_ms

if TYPE_CHECKING:
    from core_contracts import RiskGuards

try:
    import event_bus as eb
except Exception:  # на случай отсутствия event_bus в окружении
    class _Stub:
        def configure(self, *a, **k): return ""
        def log_trade(self, *a, **k): pass
        def log_risk(self, *a, **k): pass
        def flush(self): pass
        def run_dir(self): return ""
    eb = _Stub()  # type: ignore

from action_proto import ActionProto, ActionType


class RiskEvent(IntEnum):
    NONE = 0
    POSITION_LIMIT = 1        # превышение лимита по абсолютной позиции (pre/post)
    NOTIONAL_LIMIT = 2        # превышение лимита по ноционалу (post)
    DRAWDOWN = 3              # превышение лимита по дроудауну (post)
    BANKRUPTCY = 4            # cash ниже порога банкротства (post)


@dataclass
class RiskConfig:
    # Прямые жёсткие лимиты
    max_abs_position: float = 1e12
    max_notional: float = 2e12
    max_total_notional: float = 0.0
    max_total_exposure_pct: float = 0.0
    exposure_buffer_frac: float = 0.0

    # Дроудаун/устойчивость
    max_drawdown_pct: float = 1.00        # разрешённая просадка (0.30 => 30%)
    intrabar_dd_pct: float = 0.30         # «жёсткий» интра-барный триггер
    dd_window: int = 500                  # размер окна для оценки пика equity

    # Ликвидация/банкротство
    bankruptcy_cash_th: float = -1e12     # порог банкротства по кэшу

    # Технические опции
    ts_provider: callable = lambda: now_ms()


class RiskGuard:
    """
    Единая точка risk-контроля:
      * on_action_proposed(state, proto) — pre-trade проверка (возможной позиции)
      * on_post_trade(state, mid_price) — post-trade инварианты (ноционал, дроудаун, банкротство)
    Ожидается, что state имеет поля: units (float), cash (float), max_position (float | опционально).
    """

    def __init__(self, cfg: Optional[RiskConfig] = None):
        self.cfg = cfg or RiskConfig()
        self._nw_hist: Deque[Tuple[int, float]] = deque(maxlen=self.cfg.dd_window)  # (ts, net_worth)
        self._peak_nw_window: Deque[float] = deque(maxlen=self.cfg.dd_window)
        self._last_event: RiskEvent = RiskEvent.NONE

    def reset(self) -> None:
        """Reset internal statistics collected during an episode."""
        self._nw_hist.clear()
        self._peak_nw_window.clear()
        self._last_event = RiskEvent.NONE

    # ---------- ВСПОМОГАТЕЛЬНЫЕ РАСЧЁТЫ ----------

    @staticmethod
    def _get_max_position_from_state_or_cfg(state, cfg: RiskConfig) -> float:
        mp = float(getattr(state, "max_position", 0.0) or 0.0)
        if mp > 0.0 and math.isfinite(mp):
            return mp

        cfg_mp = float(cfg.max_abs_position)
        if not math.isfinite(cfg_mp) or cfg_mp <= 0.0:
            cfg_mp = 1.0
        return float(cfg_mp)

    @staticmethod
    def _notional(state, mid_price: float) -> float:
        # Абсолютная экспозиция позиции (в денежных единицах)
        price = float(mid_price)
        if not math.isfinite(price) or price <= 0.0:
            return 0.0
        return abs(float(state.units)) * price

    def _update_equity_windows(self, ts: int, state, mid_price: float) -> Tuple[float, float, float]:
        # Возвращает (nw, peak, dd_pct)
        nw = float(state.cash) + float(state.units) * float(mid_price)
        self._nw_hist.append((ts, nw))
        # поддерживаем окно пиков; если окно пустое — инициализируем пиком=NW
        if not self._peak_nw_window:
            self._peak_nw_window.append(nw)
            peak = nw
        else:
            peak = max(max(self._peak_nw_window, default=nw), nw)
            self._peak_nw_window.append(nw)
        dd_pct = 0.0 if peak <= 0 else max(0.0, (peak - nw) / peak)
        return nw, peak, dd_pct

    # ---------- PRE-TRADE ----------

    def on_action_proposed(self, state, proto: ActionProto) -> RiskEvent:
        """
        Проверяет, не приведёт ли ДЕЙСТВИЕ к нарушению лимита по абсолютной позиции.
        Возвращает RiskEvent (NONE или POSITION_LIMIT).
        """
        cfg = self.cfg
        ts = cfg.ts_provider()

        # предполагаемая позиция после применения volume_frac * max_position
        max_pos = self._get_max_position_from_state_or_cfg(state, cfg)
        # volume_frac ∈ [-1, 1], знак => направление
        delta_units = float(proto.volume_frac) * float(max_pos)

        # политики типа HOLD не изменяют позицию
        if proto.action_type == ActionType.HOLD:
            self._last_event = RiskEvent.NONE
            return self._last_event

        next_units = float(state.units) + delta_units
        if abs(next_units) > cfg.max_abs_position + 1e-12:
            evt = RiskEvent.POSITION_LIMIT
            eb.log_risk({
                "ts": ts,
                "type": "POSITION_LIMIT",
                "stage": "pre_trade",
                "units_curr": float(state.units),
                "units_next": float(next_units),
                "max_abs_position": float(cfg.max_abs_position),
                "proto": {
                    "type": int(proto.action_type),
                    "volume_frac": float(proto.volume_frac),
                    "ttl_steps": int(getattr(proto, "ttl_steps", 0) or 0),
                    "client_order_id": int(getattr(proto, "client_order_id", 0) or 0),
                },
            })
            self._last_event = evt
            return evt

        self._last_event = RiskEvent.NONE
        return self._last_event

    # ---------- POST-TRADE ----------

    def on_post_trade(self, state, mid_price: float) -> RiskEvent:
        """
        Пост-фактум проверки: лимит по ноционалу, интрабарный дроудаун, общий дроудаун и банкротство.
        Возвращает первый сработавший RiskEvent (приоритет: BANKRUPTCY > NOTIONAL_LIMIT > DRAWDOWN > POSITION_LIMIT).
        """
        cfg = self.cfg
        ts = cfg.ts_provider()

        # 1) Банкротство (по кэшу)
        if float(state.cash) < cfg.bankruptcy_cash_th:
            evt = RiskEvent.BANKRUPTCY
            eb.log_risk({
                "ts": ts,
                "type": "BANKRUPTCY",
                "cash": float(state.cash),
                "threshold": float(cfg.bankruptcy_cash_th),
            })
            self._last_event = evt
            return evt

        # 2) Лимит по ноционалу
        notion = self._notional(state, float(mid_price))
        if notion > cfg.max_notional + 1e-9:
            evt = RiskEvent.NOTIONAL_LIMIT
            eb.log_risk({
                "ts": ts,
                "type": "NOTIONAL_LIMIT",
                "notional": float(notion),
                "max_notional": float(cfg.max_notional),
                "units": float(state.units),
                "mid": float(mid_price),
                "cash": float(state.cash),
            })
            self._last_event = evt
            return evt

        # 3) Дроудаун (интрабарный быстрый триггер + оконный)
        nw, peak, dd_pct = self._update_equity_windows(ts, state, float(mid_price))
        if dd_pct >= cfg.intrabar_dd_pct - 1e-12:
            evt = RiskEvent.DRAWDOWN
            eb.log_risk({
                "ts": ts,
                "type": "DRAWDOWN_INTRABAR",
                "drawdown_pct": float(dd_pct),
                "intrabar_dd_pct": float(cfg.intrabar_dd_pct),
                "nw": float(nw),
                "peak": float(peak),
            })
            self._last_event = evt
            return evt

        if dd_pct >= cfg.max_drawdown_pct - 1e-12:
            evt = RiskEvent.DRAWDOWN
            eb.log_risk({
                "ts": ts,
                "type": "DRAWDOWN",
                "drawdown_pct": float(dd_pct),
                "max_drawdown_pct": float(cfg.max_drawdown_pct),
                "nw": float(nw),
                "peak": float(peak),
            })
            self._last_event = evt
            return evt

        # 4) Контроль «на всякий» по абсолютной позиции (post) — на случай внешних модификаций состояния
        if abs(float(state.units)) > cfg.max_abs_position + 1e-12:
            evt = RiskEvent.POSITION_LIMIT
            eb.log_risk({
                "ts": ts,
                "type": "POSITION_LIMIT",
                "stage": "post_trade",
                "units": float(state.units),
                "max_abs_position": float(cfg.max_abs_position),
            })
            self._last_event = evt
            return evt

        self._last_event = RiskEvent.NONE
        return self._last_event

    # ---------- ВСПОМОГАТЕЛЬНОЕ ----------

    def last_event(self) -> RiskEvent:
        return self._last_event

    def snapshot(self) -> Dict[str, Any]:
        """Для отладки/логов."""
        return {
            "cfg": asdict(self.cfg),
            "last_event": int(self._last_event),
            "nw_window_len": len(self._nw_hist),
        }


# ----------- PIPELINE SUPPORT -----------


@dataclass
class _SymbolState:
    """Internal per-symbol bookkeeping for lightweight risk checks."""

    last_ts: int = 0
    exposure: float = 0.0


class SimpleRiskGuard:
    """Minimal per-symbol risk guard used by the pipeline.

    The guard tracks the last processed timestamp and cumulative exposure for
    each symbol.  ``apply`` returns filtered decisions and an optional reason
    string beginning with ``"RISK_"`` if all decisions should be dropped.
    """

    def __init__(self) -> None:
        self._states: Dict[str, _SymbolState] = {}

    def _state(self, symbol: str) -> _SymbolState:
        return self._states.setdefault(symbol, _SymbolState())

    def apply(
        self, ts_ms: int, symbol: str, decisions: Sequence[Any]
    ) -> tuple[Sequence[Any], str | None]:
        st = self._state(symbol)
        if ts_ms <= st.last_ts:
            # Reject stale timestamps outright
            return [], "RISK_STALE_TS"

        def _signal_leg(order: Any) -> str:
            meta = getattr(order, "meta", None)
            if isinstance(meta, MappingABC):
                return str(meta.get("signal_leg") or "").lower()
            if meta is not None:
                getter = getattr(meta, "get", None)
                if callable(getter):
                    try:
                        value = getter("signal_leg")
                    except Exception:
                        value = None
                    else:
                        return str(value or "").lower()
            return ""

        exp = 0.0
        checked: list[Any] = []
        for d in decisions:
            leg = _signal_leg(d)
            if leg == "exit":
                checked.append(d)
                continue
            vol = getattr(d, "volume_frac", getattr(d, "quantity", 0.0)) or 0.0
            try:
                exp += abs(float(vol))
            except Exception:
                continue
            checked.append(d)

        st.last_ts = int(ts_ms)
        st.exposure += exp
        return checked, None


@dataclass
class PortfolioLimitConfig:
    """Configuration for aggregate portfolio exposure limits."""

    max_total_notional: float | None = None
    max_total_exposure_pct: float | None = None
    exposure_buffer_frac: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "PortfolioLimitConfig":
        payload: Dict[str, Any] = {}
        if isinstance(data, Mapping):
            payload.update(data)
        return cls(
            max_total_notional=cls._coerce_float(payload.get("max_total_notional")),
            max_total_exposure_pct=cls._coerce_float(payload.get("max_total_exposure_pct")),
            exposure_buffer_frac=cls._coerce_float(
                payload.get("exposure_buffer_frac"), default=0.0, minimum=0.0
            )
            or 0.0,
        )

    @staticmethod
    def _coerce_float(
        value: Any,
        *,
        default: float | None = None,
        minimum: float | None = None,
    ) -> float | None:
        if value is None:
            return default
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(coerced):
            return default
        if minimum is not None and coerced < minimum:
            return default
        return coerced

    @property
    def enabled(self) -> bool:
        return any(
            v is not None and v > 0.0
            for v in (self.max_total_notional, self.max_total_exposure_pct)
        )


class PortfolioLimitGuard:
    """Guard enforcing aggregate portfolio exposure limits."""

    _EPS = 1e-12

    def __init__(
        self,
        *,
        config: PortfolioLimitConfig,
        get_positions: Callable[[], Mapping[str, Any]],
        get_total_notional: Callable[[], float | None] | None = None,
        get_price: Callable[[str], float | None],
        get_equity: Callable[[], float | None] | None = None,
        leg_getter: Callable[[Any], str] | None = None,
    ) -> None:
        self._cfg = config
        self._get_positions = get_positions
        self._get_total_notional = get_total_notional
        self._get_price = get_price
        self._get_equity = get_equity or (lambda: None)
        self._leg_getter = leg_getter or self._default_leg_getter

    @staticmethod
    def _default_leg_getter(order: Any) -> str:
        meta = getattr(order, "meta", None)
        value: Any = None
        if isinstance(meta, MappingABC):
            value = meta.get("signal_leg")
        elif meta is not None:
            getter = getattr(meta, "get", None)
            if callable(getter):
                try:
                    value = getter("signal_leg")
                except Exception:
                    value = None
        return str(value or "").lower()

    @staticmethod
    def _normalize_symbol(symbol: Any, fallback: str | None = None) -> str:
        if symbol is None and fallback is not None:
            symbol = fallback
        if symbol is None:
            return ""
        return str(symbol).upper()

    @staticmethod
    def _extract_side(order: Any) -> str:
        side = getattr(order, "side", "")
        if hasattr(side, "value"):
            try:
                side = side.value
            except Exception:
                pass
        return str(side or "").upper()

    @staticmethod
    def _extract_quantity(order: Any) -> float | None:
        candidates = (
            getattr(order, "quantity", None),
            getattr(order, "volume", None),
            getattr(order, "size", None),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                value = float(candidate)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
        return None

    def _snapshot_positions(self) -> Dict[str, float]:
        snapshot: Dict[str, float] = {}
        try:
            positions = self._get_positions()
        except Exception:
            positions = {}
        if not isinstance(positions, MappingABC):
            return snapshot
        for sym, qty in positions.items():
            try:
                value = float(qty)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if math.isclose(value, 0.0, abs_tol=self._EPS):
                continue
            snapshot[str(sym).upper()] = value
        return snapshot

    def _base_total_notional(self, positions: Mapping[str, float]) -> float:
        if self._get_total_notional is not None:
            try:
                value = self._get_total_notional()
            except Exception:
                value = None
            else:
                if value is not None and math.isfinite(float(value)):
                    return max(0.0, float(value))
        total = 0.0
        for sym, qty in positions.items():
            price = self._safe_price(sym)
            if price is None:
                continue
            total += abs(float(qty)) * price
        return total

    def _safe_price(self, symbol: str) -> float | None:
        try:
            price = self._get_price(symbol)
        except Exception:
            return None
        if price is None:
            return None
        try:
            value = float(price)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        return value

    def _effective_limits(self) -> tuple[float | None, float | None]:
        def _sanitize(limit: float | None) -> float | None:
            if limit is None:
                return None
            try:
                value = float(limit)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value) or value <= 0.0:
                return None
            return value

        notional_limit = _sanitize(self._cfg.max_total_notional)
        equity_limit: float | None = None
        pct = self._cfg.max_total_exposure_pct
        if pct is not None and pct > 0.0:
            try:
                equity = self._get_equity()
            except Exception:
                equity = None
            if equity is not None:
                try:
                    equity_val = float(equity)
                except (TypeError, ValueError):
                    equity_val = None
                if equity_val is not None and math.isfinite(equity_val) and equity_val > 0.0:
                    equity_limit = _sanitize(equity_val * pct)
        return notional_limit, equity_limit

    def _classify_orders(
        self,
        orders: Sequence[Any],
        default_symbol: str,
    ) -> list[Dict[str, Any]]:
        infos: list[Dict[str, Any]] = []
        for idx, order in enumerate(orders):
            sym = self._normalize_symbol(getattr(order, "symbol", None), default_symbol)
            leg = self._leg_getter(order)
            reduce_only = bool(getattr(order, "reduce_only", False))
            qty = self._extract_quantity(order)
            side = self._extract_side(order)
            delta: float | None
            if qty is None:
                delta = None
            elif side == "BUY":
                delta = qty
            elif side == "SELL":
                delta = -qty
            else:
                delta = None
            infos.append(
                {
                    "index": idx,
                    "order": order,
                    "symbol": sym,
                    "delta": delta,
                    "price": self._safe_price(sym) if sym else None,
                    "leg": leg,
                    "reduce_only": reduce_only,
                }
            )
        return infos

    @staticmethod
    def _is_exit(info: Mapping[str, Any]) -> bool:
        if bool(info.get("reduce_only")):
            return True
        leg = str(info.get("leg") or "").lower()
        return leg == "exit"

    def apply(
        self, ts_ms: int, symbol: str, decisions: Sequence[Any]
    ) -> tuple[list[Any], str | None]:
        orders = list(decisions or [])
        if not orders:
            return [], None
        if not self._cfg.enabled:
            return orders, None
        positions = self._snapshot_positions()
        working_positions = dict(positions)
        current_total = self._base_total_notional(working_positions)
        notional_limit, equity_limit = self._effective_limits()
        if notional_limit is None and equity_limit is None:
            return orders, None
        buffer_raw = float(self._cfg.exposure_buffer_frac or 0.0)
        if not math.isfinite(buffer_raw) or buffer_raw < 0.0:
            buffer_raw = 0.0
        buffer_mult = 1.0 + buffer_raw
        infos = self._classify_orders(orders, symbol)
        accepted: set[int] = set()
        blocked: set[int] = set()

        # Process exits first to free up capacity.
        for info in infos:
            if not self._is_exit(info):
                continue
            idx = int(info["index"])
            accepted.add(idx)
            sym = info.get("symbol") or ""
            delta = info.get("delta")
            if not sym or delta is None:
                continue
            prev = working_positions.get(sym, 0.0)
            new = prev + float(delta)
            working_positions[sym] = new
            price = info.get("price")
            if price is not None:
                current_total += (abs(new) - abs(prev)) * float(price)
                if current_total < 0.0:
                    current_total = 0.0

        def _within_limits(total: float) -> bool:
            if notional_limit is not None and total > notional_limit + self._EPS:
                return False
            if equity_limit is not None and total > equity_limit + self._EPS:
                return False
            return True

        # Process remaining orders.
        for info in infos:
            idx = int(info["index"])
            if idx in accepted or idx in blocked:
                continue
            sym = info.get("symbol") or ""
            delta = info.get("delta")
            price = info.get("price")
            if not sym:
                blocked.add(idx)
                continue
            if delta is None:
                blocked.add(idx)
                continue
            prev = working_positions.get(sym, 0.0)
            new = prev + float(delta)
            exposure_delta = abs(new) - abs(prev)
            if exposure_delta <= self._EPS:
                accepted.add(idx)
                working_positions[sym] = new
                if price is not None:
                    current_total += exposure_delta * float(price)
                    if current_total < 0.0:
                        current_total = 0.0
                continue
            if price is None:
                blocked.add(idx)
                continue
            notional_delta = exposure_delta * float(price)
            if exposure_delta > self._EPS:
                buffered_delta = notional_delta * buffer_mult
            else:
                buffered_delta = notional_delta
            prospective_total = current_total + buffered_delta
            if not _within_limits(prospective_total):
                blocked.add(idx)
                continue
            accepted.add(idx)
            working_positions[sym] = new
            current_total += notional_delta
            if current_total < 0.0:
                current_total = 0.0

        approved: list[Any] = []
        for idx, order in enumerate(orders):
            if idx in accepted:
                approved.append(order)

        if approved:
            return approved, None
        return [], "RISK_PORTFOLIO_LIMIT"


