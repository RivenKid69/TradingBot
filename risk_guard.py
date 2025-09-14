# risk_guard.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import Optional, Deque, Tuple, Dict, Any, TYPE_CHECKING, Sequence
from collections import deque
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
        if mp <= 0.0:
            # если в стейте не задано — хеджируемся конфигом (не строже max_abs_position)
            mp = min(1.0, cfg.max_abs_position)  # «1 контракт» как «минимум», но не выше hard-cap
        return float(mp)

    @staticmethod
    def _notional(state, mid_price: float) -> float:
        # Абсолютная величина net exposure (в денежных единицах)
        # NW = cash + units * mid_price
        return abs(float(state.cash) + float(state.units) * float(mid_price))

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

        exp = 0.0
        checked: list[Any] = []
        for d in decisions:
            vol = getattr(d, "volume_frac", getattr(d, "quantity", 0.0)) or 0.0
            try:
                exp += abs(float(vol))
            except Exception:
                continue
            checked.append(d)

        st.last_ts = int(ts_ms)
        st.exposure += exp
        return checked, None


