# realtime/signal_runner.py
from __future__ import annotations

import csv
import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING


from feature_pipe import FeatureConfig, FeaturePipe
from calibration import BaseCalibrator

# Контракты подключаем только для типизации, чтобы не менять рантайм-поведение
if TYPE_CHECKING:
    from core_contracts import FeaturePipe as FeaturePipeProtocol, SignalPolicy, RiskGuards, MarketDataSource, TradeExecutor

# Контракты только для типизации (не влияют на рантайм и поведение):
if TYPE_CHECKING:
    from core_contracts import FeaturePipe, SignalPolicy, RiskGuards, MarketDataSource, TradeExecutor


@dataclass
class GuardsConfig:
    min_history_bars: int = 0
    gap_cooldown_bars: int = 0
    gap_threshold_ms: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GuardsConfig":
        return cls(
            min_history_bars=int(d.get("min_history_bars", 0)),
            gap_cooldown_bars=int(d.get("gap_cooldown_bars", 0)),
            gap_threshold_ms=int(d["gap_threshold_ms"]) if d.get("gap_threshold_ms") is not None else None,
        )


@dataclass
class NoTradeConfig:
    funding_buffer_min: int = 0
    daily_utc: List[str] = None
    custom_ms: List[Dict[str, int]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NoTradeConfig":
        return cls(
            funding_buffer_min=int(d.get("funding_buffer_min", 0)),
            daily_utc=list(d.get("daily_utc", []) or []),
            custom_ms=list(d.get("custom_ms", []) or []),
        )


class SignalRunner:
    """
    Потоковый сигналер:
      - на вход: закрытые 1m klines (dict с полями: symbol, close, close_time)
      - применяет FeaturePipe (общий с оффлайном)
      - вызывает стратегию
      - гварды: холодный старт и пауза после гэпа
      - чёрные окна: daily UTC, буфер вокруг funding (00:00/08:00/16:00), кастомные окна по ts_ms
      - частотный кулдаун
      - пишет сигналы в CSV (если разрешено конфигом)
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = dict(cfg or {})

        # фичи (онлайн) через общий код
        fparams = self.cfg.get("features", {}) or {}
        self.pipe = FeaturePipe(FeatureConfig(
            lookbacks_prices=list(fparams.get("lookbacks_prices", [5, 15, 60])),
            rsi_period=int(fparams.get("rsi_period", 14)),
        ))

        # стратегия (динамический импорт)
        s = self.cfg.get("strategy", {}) or {}
        module = str(s.get("module", "strategies.momentum"))
        clsname = str(s.get("class", "MomentumStrategy"))
        params = s.get("params", {}) or {}
        self.strategy = getattr(importlib.import_module(module), clsname)(**params)

        # гварды
        self.guards = GuardsConfig.from_dict(self.cfg.get("guards", {}) or {})
        self._hist_bars: Dict[str, int] = {}
        self._cooldown_left: Dict[str, int] = {}
        self._last_ts: Dict[str, int] = {}

        # чёрные окна
        self.no_trade = NoTradeConfig.from_dict(self.cfg.get("no_trade", {}) or {})
        self._daily_windows_min: List[tuple] = self._parse_daily_windows(self.no_trade.daily_utc or [])

        # частотный кулдаун
        self.min_signal_gap_s = int(self.cfg.get("min_signal_gap_s", 0))
        self._last_signal_ts: Dict[str, int] = {}
        
        # калибратор вероятностей (опционально)
        cal_cfg = dict(self.cfg.get("calibrator", {}) or {})
        self._calibrator_enabled: bool = bool(cal_cfg.get("enabled", False))
        self._calibrator_apply_to: str = str(cal_cfg.get("apply_to", "score"))
        self._calibrator_out_field: str = str(cal_cfg.get("out_field", "score_calibrated"))
        self._calibrator = None
        if self._calibrator_enabled:
            try:
                path = str(cal_cfg.get("path", "models/calibrator.json"))
                if os.path.exists(path):
                    self._calibrator = BaseCalibrator.load_json(path)
                else:
                    self._calibrator_enabled = False
            except Exception:
                self._calibrator_enabled = False

        # вывод
        self.out_csv = str(self.cfg.get("out_csv", "logs/signals.csv"))
        if self.out_csv:
            os.makedirs(os.path.dirname(self.out_csv) or ".", exist_ok=True)
            if not os.path.exists(self.out_csv):
                with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["ts_ms", "symbol", "side", "score", "volume_frac", "ref_price", "reason"])

    # --------------------- helpers: daily windows / funding / custom ---------------------

    @staticmethod
    def _parse_daily_windows(windows: List[str]) -> List[tuple]:
        out: List[tuple] = []
        for w in windows:
            try:
                a, b = str(w).strip().split("-")
                sh, sm = a.split(":")
                eh, em = b.split(":")
                smin = int(sh) * 60 + int(sm)
                emin = int(eh) * 60 + int(em)
                if 0 <= smin <= 1440 and 0 <= emin <= 1440 and smin <= emin:
                    out.append((smin, emin))
            except Exception:
                continue
        return out

    def _in_daily_window(self, ts_ms: int) -> bool:
        if not self._daily_windows_min:
            return False
        mins = int((ts_ms // 60000) % 1440)
        for smin, emin in self._daily_windows_min:
            if smin <= mins < emin:
                return True
        return False

    def _in_funding_buffer(self, ts_ms: int) -> bool:
        buf_min = int(self.no_trade.funding_buffer_min or 0)
        if buf_min <= 0:
            return False
        sec_day = int((ts_ms // 1000) % 86400)
        marks = [0, 8 * 3600, 16 * 3600]
        for m in marks:
            if abs(sec_day - m) <= buf_min * 60:
                return True
        return False

    def _in_custom_window(self, ts_ms: int) -> bool:
        for w in (self.no_trade.custom_ms or []):
            try:
                s = int(w.get("start_ts_ms"))
                e = int(w.get("end_ts_ms"))
                if s <= int(ts_ms) <= e:
                    return True
            except Exception:
                continue
        return False

    def _no_trade_block(self, ts_ms: int) -> bool:
        return self._in_daily_window(ts_ms) or self._in_funding_buffer(ts_ms) or self._in_custom_window(ts_ms)

    # --------------------- guards & cooldown ---------------------

    def _guards_allow(self, sym: str, ts_ms: int) -> bool:
        h = self._hist_bars.get(sym, 0)
        cd = self._cooldown_left.get(sym, 0)
        last = self._last_ts.get(sym)

        if last is not None:
            dt = int(ts_ms) - int(last)
            thr = self.guards.gap_threshold_ms if self.guards.gap_threshold_ms is not None else 90000
            if dt > max(0, int(thr)):
                self._cooldown_left[sym] = int(self.guards.gap_cooldown_bars or 0)

        self._hist_bars[sym] = h + 1
        self._last_ts[sym] = int(ts_ms)

        if self._hist_bars[sym] < int(self.guards.min_history_bars or 0):
            return False
        if self._cooldown_left.get(sym, 0) > 0:
            self._cooldown_left[sym] = max(0, self._cooldown_left[sym] - 1)
            return False
        return True

    # --------------------- основной обработчик ---------------------

    def on_kline(self, kline: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        kline: словарь минимум с полями:
            symbol: "BTCUSDT"
            close: float
            close_time: int (ts_ms конца свечи)
        """
        sym = str(kline["symbol"]).upper()
        ts_ms = int(kline["close_time"])
        close = float(kline["close"])

        feats = self.pipe.on_kline({"symbol": sym, "close": close, "close_time": ts_ms})
        if feats is None:
            return None

        allowed = self._guards_allow(sym, ts_ms)
        if allowed and self._no_trade_block(ts_ms):
            allowed = False

        self.strategy.on_features({**feats})
        decisions = self.strategy.decide({"ts_ms": ts_ms, "symbol": sym, "ref_price": feats["ref_price"], "features": feats})

        if self.min_signal_gap_s > 0 and decisions:
            last_sig = self._last_signal_ts.get(sym)
            if last_sig is not None and (ts_ms - last_sig) < self.min_signal_gap_s * 1000:
                decisions = []

        if not allowed:
            decisions = []

        if not decisions:
            return None

        d = decisions[0].__dict__
        side = d.get("side", "")
        score = float(d.get("score", 0.0))
        volume_frac = float(d.get("volume_frac", d.get("size", 0.0)))
        ref_price = float(feats["ref_price"])

        self._last_signal_ts[sym] = ts_ms

        # стандартизированный выход стратегии: OrderIntent для первого решения
        try:
            from order_shims import OrderContext, decisions_to_order_intents
            _ctx = OrderContext(
                ts_ms=int(ts_ms),
                symbol=str(sym),
                ref_price=float(ref_price),
                max_position_abs_base=1.0,  # здесь только намерение; объём разрешит исполнитель
                tick_size=None,
                price_offset_ticks=int(d.get("price_offset_ticks", 0)) if decisions else 0,
                tif=str(d.get("tif", "GTC")) if decisions else "GTC",
                client_tag=None,
                round_qty_fn=None,
            )
            _intents = decisions_to_order_intents([d], _ctx) if decisions else []
            core_order_intent = (_intents[0].to_dict() if _intents else None)
        except Exception:
            core_order_intent = None

        row = {
            "ts_ms": ts_ms,
            "symbol": sym,
            "side": side,
            "score": score,
            "volume_frac": volume_frac,
            "ref_price": ref_price,
            "core_order_intent": core_order_intent,
            "reason": reason,
        }

        # применим калибратор (если включен) и добавим поле
        if self._calibrator_enabled and self._calibrator is not None:
            try:
                if self._calibrator_apply_to in d:
                    raw = float(d.get(self._calibrator_apply_to, score))
                else:
                    raw = float(score)
                p = float(self._calibrator.predict_proba([raw])[0])
                row[self._calibrator_out_field] = p
            except Exception:
                # в случае ошибки просто пропустим калибровку
                pass

        if self.out_csv:
            with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([row["ts_ms"], row["symbol"], row["side"], row["score"], row["volume_frac"], row["ref_price"], row["reason"]])

        return row
