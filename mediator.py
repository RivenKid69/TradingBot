from __future__ import annotations

"""
Mediator — координатор между средой, LOB/симулятором исполнения, RiskGuard и EventBus.
Держит TTL ордеров, прокидывает действия агента и обновляет портфельное состояние.

Контракт ожиданий:
- env_ref.state имеет как минимум атрибуты: units: float, cash: float, max_position: float (опционально).
- env_ref.lob (опционально): объект с методами add_limit_order, remove_order, match_market_order.
  Если отсутствует — используется _DummyLOB (ничего не делает, но не ломает пайплайн).
- ExecutionSimulator (если используется) должен предоставлять внутренний SimStepReport (см. execution_sim.py),
  а наружу для логирования и анализа использовать единые core_models.ExecReport через compat_shims/sim_adapter.
"""

from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core_models import Order, ExecReport, Position, TradeLogRow
    from core_contracts import TradeExecutor, RiskGuards

import numpy as np

from core_models import ExecReport, TradeLogRow, Side, OrderType, Liquidity, ExecStatus
from core_events import EventType, OrderEvent, FillEvent
from compat_shims import sim_report_dict_to_core_exec_reports
import event_bus as eb
from impl_latency import LatencyImpl

try:
    import event_bus
except Exception:
    # мягкая деградация при отсутствии event_bus
    class _Stub:
        def configure(self, *a, **k): return ""
        def log_trade(self, *a, **k): pass
        def log_risk(self, *a, **k): pass
        def flush(self): pass
        def run_dir(self): return ""
    event_bus = _Stub()  # type: ignore

from action_proto import ActionProto, ActionType
from risk_guard import RiskConfig, RiskGuard
from compat_shims import sim_report_dict_to_core_exec_reports
from core_models import as_dict
from order_shims import actionproto_to_order, legacy_decision_to_order, OrderContext

# ExecutionSimulator и ExecReport — опциональны; при отсутствии работаем напрямую с LOB.
# ExecutionSimulator и SimStepReport (как внутренний тип отчёта) — опциональны;
# при отсутствии работаем напрямую с LOB. Импортируем SimStepReport как ExecReport
try:
    from execution_sim import SimStepReport as ExecReport, ExecutionSimulator  # type: ignore
    _HAVE_EXEC_SIM = True
except Exception:
    ExecReport = None  # type: ignore
    ExecutionSimulator = None  # type: ignore
    _HAVE_EXEC_SIM = False


# ------------------------------ Вспомогательная заглушка LOB ------------------------------

class _DummyLOB:
    """Минималистичная заглушка для разработки без Cython."""
    _next_id: int

    def __init__(self):
        self._next_id = 1

    def add_limit_order(self, is_buy_side: bool, price_ticks: int, volume: float, timestamp: int,
                        taker_is_agent: bool = True) -> Tuple[int, int]:
        oid = self._next_id
        self._next_id += 1
        # (order_id, fake_queue_position)
        return int(oid), 0

    def remove_order(self, is_buy_side: bool, price_ticks: int, order_id: int) -> bool:
        return True

    def match_market_order(self, is_buy_side: bool, volume: float, timestamp: int,
                           taker_is_agent: bool, out_prices=None, out_volumes=None,
                           out_is_buy=None, out_is_self=None, out_ids=None, max_len: int = 0):
        # Заглушка: не исполняем, возвращаем ноль сделок и нулевую комиссию
        return 0, 0.0


# ------------------------------ Mediator ------------------------------

@dataclass
class _EnvStateView:
    units: float
    cash: float
    max_position: float = 0.0


class Mediator:
    def __init__(self, env_ref: Any, *, event_level: int = 0,
                 use_exec_sim: Optional[bool] = None,
                 latency_steps: int = 0, slip_k: float = 0.0, seed: int = 0,
                 latency_cfg: dict | None = None):
        """
        env_ref — ссылка на «среду» (должна держать .state и, опционально, .lob)
        event_level — уровень логов EventBus (0/1/2)
        use_exec_sim — если None, выбираем автоматически по наличию execution_sim
        latency_steps/slip_k/seed — параметры ExecutionSimulator (если используется)
        latency_cfg — параметры модели латентности для ExecutionSimulator
        """
        self.env = env_ref

        # EventBus
        try:
            event_bus.configure(level=event_level)
        except Exception:
            pass

        # RiskGuard c параметрами из env_ref (если заданы)
        self.risk = RiskGuard(
            RiskConfig(
                max_abs_position=float(getattr(env_ref, "max_abs_position", 1e12)),
                max_notional=float(getattr(env_ref, "max_notional", 2e12)),
                max_drawdown_pct=float(getattr(env_ref, "max_drawdown_pct", 1.0)),
                intrabar_dd_pct=float(getattr(env_ref, "intrabar_dd_pct", 0.30)),
                dd_window=int(getattr(env_ref, "dd_window", 500)),
                bankruptcy_cash_th=float(getattr(env_ref, "bankruptcy_cash_th", -1e12)),
            )
        )

        # LOB (реальный или заглушка)
        self.lob = getattr(env_ref, "lob", None) or _DummyLOB()

        # ExecutionSimulator — опционально
        if use_exec_sim is None:
            use_exec_sim = _HAVE_EXEC_SIM
        self._use_exec = bool(use_exec_sim and _HAVE_EXEC_SIM)

        if latency_cfg is None:
            rc = getattr(env_ref, "run_config", None)
            if rc is not None:
                latency_cfg = getattr(rc, "latency", None)

        self._latency_impl: LatencyImpl | None = None
        if self._use_exec:
            self.exec = ExecutionSimulator(latency_steps=latency_steps, slip_k=slip_k, seed=seed)  # type: ignore
            try:
                l_impl = LatencyImpl.from_dict(latency_cfg or {})
                l_impl.attach_to(self.exec)
                self._latency_impl = l_impl
            except Exception:
                self._latency_impl = None
        else:
            self.exec = None
            self._latency_impl = None

        # TTL-очередь: [(order_id, expire_ts)]
        self._ttl_queue: List[Tuple[int, int]] = []

        # Внутренние «ожидаемые» объёмы по последним операциям (для согласования с отчётом)
        self._pending_buy_volume: float = 0.0
        self._pending_sell_volume: float = 0.0

    # ------------------------------ Служебное ------------------------------

    def reset(self) -> None:
        """Очистить внутреннее состояние посредника (портфельное состояние живёт в env.state)."""
        # логируем накопленную статистику латентности за предыдущий эпизод
        self.on_episode_end()
        self._ttl_queue.clear()
        self._pending_buy_volume = 0.0
        self._pending_sell_volume = 0.0

    def on_episode_end(self) -> None:
        """Запросить и вывести статистику латентности."""
        if self._latency_impl is None:
            return
        try:
            stats = self._latency_impl.get_stats()
        except Exception:
            stats = None
        if stats:
            try:
                event_bus.log_risk({"etype": "LATENCY_STATS", **stats})
            except Exception:
                pass
            try:
                self._latency_impl.reset_stats()
            except Exception:
                pass

    def _state_view(self) -> _EnvStateView:
        st = getattr(self.env, "state", None)
        if st is None:
            # fallback: минимальный стейт
            return _EnvStateView(units=0.0, cash=0.0, max_position=0.0)
        return _EnvStateView(
            units=float(getattr(st, "units", 0.0)),
            cash=float(getattr(st, "cash", 0.0)),
            max_position=float(getattr(st, "max_position", 0.0) or 0.0),
        )

    def _apply_trades_to_state(self, trades: List[Tuple[float, float, bool, bool]]) -> None:
        """
        Обновить env.state по списку сделок: (price, volume, is_buy, maker_is_agent).
        Buy → увеличивает units, уменьшает cash; Sell → уменьшает units, увеличивает cash.
        """
        st = getattr(self.env, "state", None)
        if st is None:
            return
        for price, vol, is_buy, _maker_is_agent in trades:
            if is_buy:
                st.units = float(st.units) + float(vol)
                st.cash = float(st.cash) - float(price) * float(vol)
            else:
                st.units = float(st.units) - float(vol)
                st.cash = float(st.cash) + float(price) * float(vol)

    def _process_ttl_queue(self, now_ts: int) -> None:
        """Отменить просроченные ордера."""
        if not self._ttl_queue:
            return
        keep: List[Tuple[int, int]] = []
        for order_id, exp_ts in self._ttl_queue:
            if now_ts >= exp_ts:
                # Нет информации о стороне/цене в очереди — отменяем «вслепую», игнорируя результат
                try:
                    self.lob.remove_order(True, 0, order_id)
                except Exception:
                    pass
                try:
                    self.lob.remove_order(False, 0, order_id)
                except Exception:
                    pass
            else:
                keep.append((order_id, exp_ts))
        self._ttl_queue = keep

    # ------------------------------ Публичный API ------------------------------

    def add_limit_order(self, *, is_buy_side: bool, price_ticks: int, volume: float,
                        timestamp: int, ttl_steps: int = 0, taker_is_agent: bool = True) -> Tuple[int, int]:
        """
        Разместить лимитный ордер напрямую в LOB.
        Возвращает (order_id, queue_position). При ttl_steps>0 — ордер будет отменён после истечения.
        """
        # pre-trade риск по ожидаемой позиции
        st = self._state_view()
        proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=float(volume) / max(1.0, st.max_position))
        evt = self.risk.on_action_proposed(self.env.state, proto)  # type: ignore[attr-defined]
        if evt.name != "NONE":
            return 0, 0

        order_id, qpos = self.lob.add_limit_order(bool(is_buy_side), int(price_ticks), float(volume), int(timestamp),
                                                  bool(taker_is_agent))
        if int(ttl_steps) > 0:
            ttl_set = False
            if hasattr(self.lob, "set_order_ttl"):
                try:
                    ttl_set = bool(self.lob.set_order_ttl(int(order_id), int(ttl_steps)))
                except Exception:
                    ttl_set = False
            if not ttl_set:
                self._ttl_queue.append((int(order_id), int(timestamp) + int(ttl_steps)))
        # учёт «ожидаемого» объёма
        if is_buy_side:
            self._pending_buy_volume += float(volume)
        else:
            self._pending_sell_volume += float(volume)
        return int(order_id), int(qpos)

    def remove_order(self, *, is_buy_side: bool, price_ticks: int, order_id: int) -> bool:
        """Отменить ордер по ID и цене (грубый контракт, для реального LOB достаточно ID)."""
        ok = False
        try:
            ok = bool(self.lob.remove_order(bool(is_buy_side), int(price_ticks), int(order_id)))
        finally:
            # убрать из TTL-очереди, если там есть
            self._ttl_queue = [(oid, ts) for (oid, ts) in self._ttl_queue if oid != int(order_id)]
        return ok

    def match_market_order(self, *, is_buy_side: bool, volume: float, timestamp: int,
                           taker_is_agent: bool = True) -> List[Tuple[float, float, bool, bool]]:
        """
        Исполнить маркет-заявку через LOB.
        Возвращает список сделок [(price, volume, is_buy, maker_is_agent)].
        """
        # pre-trade риск по ожидаемой позиции
        st = self._state_view()
        proto = ActionProto(action_type=ActionType.MARKET, volume_frac=float(volume) / max(1.0, st.max_position))
        evt = self.risk.on_action_proposed(self.env.state, proto)  # type: ignore[attr-defined]
        if evt.name != "NONE":
            return []

        # Попробуем использовать Cython LOB сигнатуру (с буферами), если она доступна
        trades: List[Tuple[float, float, bool, bool]] = []
        try:
            max_len = 1024
            prices = np.empty(max_len, dtype=np.float64)
            vols = np.empty(max_len, dtype=np.float64)
            is_buy_arr = np.empty(max_len, dtype=np.int32)
            is_self_arr = np.empty(max_len, dtype=np.int32)
            ids = np.empty(max_len, dtype=np.int64)
            n, fee_total = self.lob.match_market_order(bool(is_buy_side), float(volume), int(timestamp),
                                                       bool(taker_is_agent), prices, vols,
                                                       is_buy_arr, is_self_arr, ids, int(max_len))
            for i in range(int(n)):
                trades.append((float(prices[i]), float(vols[i]), bool(is_buy_arr[i]), bool(is_self_arr[i])))
        except Exception:
            # Заглушечный путь: ничего не исполнилось
            trades = []

        # применяем сделки к состоянию и логируем
        if trades:
            self._apply_trades_to_state(trades)
            for (px, vol, is_buy, is_self) in trades:
                try:
                    # формируем ExecReport и логируем единообразно
                    _rid = str(getattr(event_bus, "_STATE").run_id if hasattr(event_bus, "_STATE") else "")
                    _sym = str(getattr(event_bus, "_STATE").default_symbol if hasattr(event_bus, "_STATE") else "UNKNOWN")
                    _er = ExecReport(
                        ts=int(timestamp),
                        run_id=_rid,
                        symbol=_sym,
                        side=Side.BUY if bool(is_buy) else Side.SELL,
                        order_type=OrderType.MARKET,
                        price=Decimal(str(float(px))),
                        quantity=Decimal(str(float(vol))),
                        fee=Decimal("0"),
                        fee_asset=None,
                        exec_status=ExecStatus.FILLED,
                        liquidity=Liquidity.UNKNOWN,
                        client_order_id=None,
                        order_id=None,
                        trade_id=None,
                        pnl=None,
                        meta={},
                    )
                    event_bus.log_trade(_er)
                except Exception:
                    pass

        # post-trade проверки
        mid_for_risk = trades[-1][0] if trades else float(getattr(self.env, "last_mid", 0.0))
        try:
            self.risk.on_post_trade(self.env.state, float(mid_for_risk))  # type: ignore[attr-defined]
        except Exception:
            pass

        return trades

    def step_action(self, proto: ActionProto, *, timestamp: int) -> dict:
        """
        Унифицированная точка для выполнения действия агента.
        Возвращает краткий отчёт dict (совместим с ExecReport.to_dict при наличии execution_sim).
        """
        # локальный буфер событий
        events: list[dict] = []

        now_ts = int(timestamp)
        try:
            self._process_ttl_queue(now_ts)
        except Exception:
            pass
        try:
            decay_fn = getattr(self.lob, "decay_ttl_and_cancel", None)
            if callable(decay_fn):
                decay_fn()
        except Exception:
            pass

        # Сформировать Order из proto (или legacy dict)
        ctx = OrderContext(
            ts_ms=int(timestamp),
            symbol=str(getattr(self.env, "symbol", "UNKNOWN")),
            ref_price=float(getattr(self.env, "last_mid", 0.0)) if hasattr(self.env, "last_mid") else None,
            max_position_abs_base=float(getattr(getattr(self.env, "state", None), "max_position", 0.0) or getattr(self.env, "max_abs_position", 0.0) or 0.0),
            tick_size=None,  # квантование делается ниже по контуру
            price_offset_ticks=int(getattr(proto, "price_offset_ticks", 0)),
            tif=str(getattr(proto, "tif", "GTC")),
            client_tag=str(getattr(proto, "client_tag", "") or ""),
        )

        order_obj: Order | None = None
        try:
            # Если это ActionProto
            order_obj = actionproto_to_order(proto, ctx)
        except Exception:
            # Если пришёл legacy dict
            if isinstance(proto, dict):
                order_obj = legacy_decision_to_order(proto, ctx)

        # публикация факта подачи действия
        try:
            submitted_event = OrderEvent(
                etype=EventType.ORDER_SUBMITTED,
                ts=int(timestamp),
                order=(order_obj.to_dict() if hasattr(order_obj, "to_dict") else None),
                meta={"action": getattr(proto, "to_dict", lambda: {"type": int(getattr(proto, "action_type", 0))})()}
            ).to_dict()
            events.append(submitted_event)
        except Exception:
            pass

        # pre-trade
        evt = self.risk.on_action_proposed(self.env.state, proto)  # type: ignore[attr-defined]
        info: dict = {}
        if evt.name != "NONE":
            info["risk_event"] = evt.name
            return {"trades": [], "cancelled_ids": [], "new_order_ids": [], "fee_total": 0.0,
                    "new_order_pos": [], "info": info, "events": events}

        # если есть ExecutionSimulator — используем его
        if self._use_exec and self.exec is not None:
            try:
                bid = getattr(self.env, "last_bid", None)
                ask = getattr(self.env, "last_ask", None)
                if bid is not None and ask is not None:
                    try:
                        self.exec.set_market_snapshot(bid=bid, ask=ask)  # type: ignore[union-attr]
                    except Exception:
                        pass
                    mid = getattr(self.env, "last_mid", None)
                    if mid is None:
                        mid = (float(bid) + float(ask)) / 2.0
                    try:
                        self.exec.set_ref_price(float(mid))  # type: ignore[union-attr]
                    except Exception:
                        pass
                cli_id = self.exec.submit(proto)  # type: ignore[union-attr]
                # в простом варианте считаем, что latency=0 и сразу «поп» (если latency>0 — поп произойдёт на тик)
                report: ExecReport = self.exec.pop_ready()  # type: ignore  # ExecReport — это alias на SimStepReport
                try:
                    setattr(self.env, "last_bid", float(getattr(report, "bid", getattr(self.env, "last_bid", 0.0))))
                    setattr(self.env, "last_ask", float(getattr(report, "ask", getattr(self.env, "last_ask", 0.0))))
                    setattr(self.env, "last_mid", float(getattr(report, "mtm_price", getattr(self.env, "last_mid", 0.0))))
                except Exception:
                    pass
                # применить и пост-проверки
                self._apply_trades_to_state(report.trades)
                mid_for_risk = float(getattr(report, "mtm_price", getattr(self.env, "last_mid", 0.0)))
                self.risk.on_post_trade(self.env.state, mid_for_risk)  # type: ignore[attr-defined]
                d = report.to_dict()
                try:
                    exec_reports = sim_report_dict_to_core_exec_reports(
                        d,
                        symbol=str(getattr(self.env, "symbol", getattr(self.env, "base_symbol", "UNKNOWN"))),
                        client_order_id=None
                    )
                    d["core_exec_reports"] = [as_dict(er) for er in exec_reports]

                    # добавлено: публикация FillEvent и запись в unified CSV
                    try:
                        lvl = int(getattr(self, "event_level", 0))
                    except Exception:
                        lvl = 0
                    for _er in exec_reports:
                        if lvl >= 2:
                            try:
                                events.append(FillEvent(etype=EventType.EXEC_FILLED, ts=_er.ts, exec_report=_er).to_dict())
                            except Exception:
                                pass
                        # лог в unified-CSV
                        try:
                            run_id_val = getattr(event_bus, "_STATE").run_id if hasattr(event_bus, "_STATE") else ""
                            symbol_val = getattr(self.env, "symbol", getattr(self.env, "base_symbol", "UNKNOWN"))
                            event_bus.log_trade(_er)
                        except Exception:
                            pass
            except Exception:
                d["core_exec_reports"] = []

            # возвращаем также события
            d["events"] = events
            d["info"] = info
            return d
        except Exception:
            # запасной путь — выполнить напрямую по типу действия
            pass

        # иначе — прямое исполнение через LOB
        trades: List[Tuple[float, float, bool, bool]] = []
        new_order_ids: List[int] = []
        new_order_pos: List[int] = []
        cancelled_ids: List[int] = []
        fee_total: float = 0.0

        if proto.action_type == ActionType.HOLD:
            pass
        elif proto.action_type == ActionType.MARKET:
            trades = self.match_market_order(is_buy_side=(proto.volume_frac > 0.0),
                                             volume=abs(proto.volume_frac) * max(1.0, self._state_view().max_position),
                                             timestamp=int(timestamp), taker_is_agent=True)
        elif proto.action_type == ActionType.LIMIT:
            price_ticks = int(getattr(proto, "price_offset_ticks", 0))
            ttl_steps = int(getattr(proto, "ttl_steps", 0))
            vol = abs(proto.volume_frac) * max(1.0, self._state_view().max_position)
            oid, qpos = self.add_limit_order(is_buy_side=(proto.volume_frac > 0.0),
                                             price_ticks=price_ticks, volume=vol,
                                             timestamp=int(timestamp), ttl_steps=ttl_steps, taker_is_agent=True)
            if oid:
                new_order_ids.append(int(oid))
                new_order_pos.append(int(qpos))

        # пост-проверки и отчёт
        mid_for_risk = trades[-1][0] if trades else float(getattr(self.env, "last_mid", 0.0))
        try:
            self.risk.on_post_trade(self.env.state, float(mid_for_risk))  # type: ignore[attr-defined]
        except Exception:
            pass

        for (px, vol, is_buy, is_self) in trades:
            try:
                # формируем ExecReport и логируем единообразно
                _rid = str(getattr(event_bus, "_STATE").run_id if hasattr(event_bus, "_STATE") else "")
                _sym = str(getattr(event_bus, "_STATE").default_symbol if hasattr(event_bus, "_STATE") else "UNKNOWN")
                _er = ExecReport(
                    ts=int(timestamp),
                    run_id=_rid,
                    symbol=_sym,
                    side=Side.BUY if bool(is_buy) else Side.SELL,
                    order_type=OrderType.MARKET,
                    price=Decimal(str(float(px))),
                    quantity=Decimal(str(float(vol))),
                    fee=Decimal("0"),
                    fee_asset=None,
                    exec_status=ExecStatus.FILLED,
                    liquidity=Liquidity.UNKNOWN,
                    client_order_id=None,
                    order_id=None,
                    trade_id=None,
                    pnl=None,
                    meta={},
                )
                event_bus.log_trade(_er)
            except Exception:
                pass

        return {
            "trades": trades,
            "cancelled_ids": cancelled_ids,
            "new_order_ids": new_order_ids,
            "fee_total": float(fee_total),
            "new_order_pos": new_order_pos,
            "info": info,
            "events": events,
        }
