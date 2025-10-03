# event_bus.py
# Лёгкая шина событий с потокобезопасной записью:
# - log_trades_<runid>.csv (UNIFIED CSV: ts,run_id,symbol,side,order_type,price,quantity,fee,fee_asset,pnl,exec_status,liquidity,client_order_id,order_id,meta_json)
# - risk.jsonl (по строке JSON на событие риска)
# Поддерживает много процессов через multiprocessing.Lock.
from __future__ import annotations

import atexit
import csv
import json
import os
import time
from dataclasses import asdict, is_dataclass
from multiprocessing import Lock
from utils.prometheus import Counter


class Topics:
    """Common event topics used across trading components."""

    RISK = "risk"


__all__ = [
    "Topics",
    "configure",
    "set_defaults",
    "run_dir",
    "log_signal_metric",
    "log_trade",
    "log_trade_row",
    "log_trade_exec",
    "log_risk",
    "flush",
    "close",
]

# Уровень событий: NONE=0, SUMMARY=1, FULL=2
class EventLevel(int):
    NONE = 0
    SUMMARY = 1
    FULL = 2

# Глобальное состояние шины
class _BusState:
    __slots__ = (
        "level",
        "root",
        "run_dir",
        "risk_path",
        "risk_file",
        "unified_path",
        "unified_file",
        "unified_writer",
        "run_id",
        "default_symbol",
        "lock",
        "initialized",
    )

    def __init__(self):
        self.level = EventLevel.NONE
        self.root = "logs"
        self.run_dir = None
        self.risk_path = None
        self.risk_file = None
        self.unified_path = None
        self.unified_file = None
        self.unified_writer = None
        self.run_id = ""
        self.default_symbol = None
        self.lock = Lock()
        self.initialized = False

_STATE = _BusState()

# Prometheus counters for signal rate limiting
_SIGNALS_TOTAL = Counter(
    "signals_total",
    "Total number of outbound trading signals",
)
_SIGNALS_DELAYED = Counter(
    "signals_delayed",
    "Signals delayed due to rate limiting",
)
_SIGNALS_REJECTED = Counter(
    "signals_rejected",
    "Signals rejected due to rate limiting",
)


def log_signal_metric(status: str) -> None:
    """Emit signal rate limiter metrics and optional risk events."""
    try:
        _SIGNALS_TOTAL.inc()
        if status == "delayed":
            _SIGNALS_DELAYED.inc()
            try:
                log_risk({"etype": "SIGNAL_DELAYED"})
            except Exception:
                pass
        elif status == "rejected":
            _SIGNALS_REJECTED.inc()
            try:
                log_risk({"etype": "SIGNAL_REJECTED"})
            except Exception:
                pass
    except Exception:
        pass

def _ensure_open():
    """Открывает файлы, если ещё не открыты. Потокобезопасно."""
    if _STATE.initialized:
        return
    with _STATE.lock:
        if _STATE.initialized:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        _STATE.run_dir = os.path.join(_STATE.root, f"run_{ts}")
        os.makedirs(_STATE.run_dir, exist_ok=True)
        _STATE.risk_path = os.path.join(_STATE.run_dir, "risk.jsonl")

        # unified log path / defaults
        _STATE.unified_path = os.path.join(_STATE.run_dir, f"log_trades_{ts}.csv")
        _STATE.run_id = ts
        _STATE.default_symbol = getattr(_STATE, "default_symbol", None)

        # unified CSV c заголовком
        if not hasattr(_STATE, "unified_file"):
            _STATE.unified_file = open(_STATE.unified_path, "a", newline="", encoding="utf-8")
            _STATE.unified_writer = csv.writer(_STATE.unified_file)
            _STATE.unified_writer.writerow([
                "ts","run_id","symbol","side","order_type","price","quantity","fee","fee_asset",
                "pnl","exec_status","liquidity","client_order_id","order_id","meta_json"
            ])

        # risk.jsonl
        _STATE.risk_file = open(_STATE.risk_path, "w")

        _STATE.initialized = True

def configure(
    level: int = EventLevel.NONE,
    *,
    root: str = "logs",
    run_id: str | None = None,
    default_symbol: str | None = None,
) -> str:
    """
    Настройка уровня логирования и корневой директории.
    Возвращает путь к директории текущего запуска.
    """
    _STATE.level = int(level)
    _STATE.root = str(root)
    _STATE.run_id = str(run_id) if run_id is not None else ""
    _STATE.default_symbol = str(default_symbol) if default_symbol is not None else None

def set_defaults(*, run_id: str | None = None, default_symbol: str | None = None) -> None:
    if run_id is not None:
        _STATE.run_id = run_id
    if default_symbol is not None:
        _STATE.default_symbol = default_symbol

def run_dir() -> str:
    """Путь к директории текущего запуска (создаётся лениво)."""
    _ensure_open()
    return _STATE.run_dir

def log_trade(ts: int, price: float, volume: float, is_buy: bool, agent_flag: bool, order_id: int | None = None):
    """
    Логирует трейд. Если уровень NONE — запись пропускается (но формат сохраняем для совместимости).
    side: 'B' или 'S'
    """
    if _STATE.level <= EventLevel.NONE:
        return
    _ensure_open()
    side = "B" if is_buy else "S"
    with _STATE.lock:
        # Запись в unified-CSV (best-effort маппинг)
        sym = _STATE.default_symbol or "UNKNOWN"
        side_str = "BUY" if is_buy else "SELL"
        try:
            _STATE.unified_writer.writerow([
                int(ts),
                getattr(_STATE, "run_id", ""),
                sym,
                side_str,
                "MARKET",
                float(price),
                float(volume),
                0.0,               # fee (неизвестно на этом уровне)
                None,              # fee_asset
                None,              # pnl
                "FILLED",          # exec_status
                "UNKNOWN",         # liquidity
                None,              # client_order_id
                None,              # order_id
                "{}",              # meta_json
            ])
            _STATE.unified_file.flush()
        except Exception:
            pass

def log_trade_row(*, ts: int, run_id: str, symbol: str, side: str, order_type: str,
                  price: float, quantity: float, fee: float = 0.0, fee_asset: str | None = None,
                  pnl: float | None = None, exec_status: str = "FILLED", liquidity: str = "UNKNOWN",
                  client_order_id: str | None = None, order_id: str | None = None, meta_json: str = "{}") -> None:
    if _STATE.level <= EventLevel.NONE:
        return
    _ensure_open()
    with _STATE.lock:
        try:
            _STATE.unified_writer.writerow([
                int(ts), run_id, symbol, side, order_type, float(price), float(quantity),
                float(fee), fee_asset, (None if pnl is None else float(pnl)),
                exec_status, liquidity, client_order_id, order_id, meta_json
            ])
            _STATE.unified_file.flush()
        except Exception:
            pass
def log_trade_exec(er) -> None:
    """
    Удобная запись ExecReport/TradeLogRow/словаря в unified-CSV.
    Формат er: core_models.ExecReport | core_models.TradeLogRow | dict совместимой схемы.
    """
    if _STATE.level <= EventLevel.NONE:
        return
    _ensure_open()
    try:
        # ленивые импорты, чтобы не ломать окружение
        from core_models import ExecReport as _ER, TradeLogRow as _TLR
        import json as _json
    except Exception:
        _ER = None  # type: ignore
        _TLR = None  # type: ignore
        import json as _json  # type: ignore

    if hasattr(er, "to_dict"):
        d = er.to_dict()
    elif isinstance(er, dict):
        d = dict(er)
    else:
        # неподдерживаемый тип — пропускаем
        return

    # если это ExecReport — приводим к TradeLogRow совместимой схеме
    if _ER is not None and isinstance(er, _ER):
        # подчистим возможные Decimal/Enum уже в to_dict(); ключи в event_bus фиксируем вручную
        d = {
            "ts": d.get("ts"), "run_id": d.get("run_id"), "symbol": d.get("symbol"),
            "side": d.get("side"), "order_type": d.get("order_type"),
            "price": d.get("price"), "quantity": d.get("quantity"),
            "fee": d.get("fee"), "fee_asset": d.get("fee_asset"),
            "pnl": d.get("pnl"), "exec_status": d.get("exec_status"),
            "liquidity": d.get("liquidity"), "client_order_id": d.get("client_order_id"),
            "order_id": d.get("order_id"), "meta_json": _json.dumps(d.get("meta") or {}, ensure_ascii=False)
        }

    # значения по умолчанию
    run_id = getattr(_STATE, "run_id", "")
    symbol = getattr(_STATE, "default_symbol", "UNKNOWN")
    ts = int(d.get("ts", int(time.time() * 1000)))
    side = str(d.get("side") or "BUY")
    order_type = str(d.get("order_type") or "MARKET")
    price = float(d.get("price") or 0.0)
    quantity = float(d.get("quantity") or 0.0)
    fee = float(d.get("fee") or 0.0)
    fee_asset = d.get("fee_asset")
    pnl = (None if d.get("pnl") is None else float(d.get("pnl")))
    exec_status = str(d.get("exec_status") or "FILLED")
    liquidity = str(d.get("liquidity") or "UNKNOWN")
    client_order_id = d.get("client_order_id")
    order_id = d.get("order_id")
    meta_json = d.get("meta_json") or "{}"

    # запись одной строки unified-CSV
    with _STATE.lock:
        try:
            _STATE.unified_writer.writerow([
                ts, run_id, symbol, side, order_type, price, quantity,
                fee, fee_asset, pnl, exec_status, liquidity, client_order_id, order_id, meta_json
            ])
            _STATE.unified_file.flush()
        except Exception:
            pass

def log_risk(obj):
    """
    Логирует событие риска в risk.jsonl.
    Принимает dict/датакласс/любой JSON-сериализуемый объект.
    """
    if _STATE.level <= EventLevel.NONE:
        return
    _ensure_open()
    if is_dataclass(obj):
        obj = asdict(obj)
    elif not isinstance(obj, dict):
        obj = {"message": str(obj)}
    # нормализуем тип (best-effort, не ломаем существующие поля)
    try:
        if "etype" not in obj:
            obj["etype"] = "RISK_EVENT"
    except Exception:
        pass
    with _STATE.lock:
        _STATE.risk_file.write(json.dumps(obj, ensure_ascii=False) + "\n")
        _STATE.risk_file.flush()

def flush():
    """Принудительный flush всех файлов."""
    if not _STATE.initialized:
        return
    with _STATE.lock:
        try:
            _STATE.risk_file.flush()
        except Exception:
            pass
        try:
            _STATE.unified_file.flush()
        except Exception:
            pass

def close():
    """Закрывает файлы (используется при завершении процесса)."""
    if not _STATE.initialized:
        return
    with _STATE.lock:
        try:
            _STATE.risk_file.close()
        except Exception:
            pass
        try:
            _STATE.unified_file.close()
        except Exception:
            pass
        _STATE.initialized = False

@atexit.register
def _on_exit():
    try:
        flush()
    finally:
        close()
