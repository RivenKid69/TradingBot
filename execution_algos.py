# sim/execution_algos.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class MarketChild:
    ts_offset_ms: int
    qty: float
    liquidity_hint: Optional[float] = None  # если хотим переопределить ликвидность на шаге


class BaseExecutor:
    def plan_market(self, *, now_ts_ms: int, side: str, target_qty: float, snapshot: Dict[str, Any]) -> List[MarketChild]:
        raise NotImplementedError


class TakerExecutor(BaseExecutor):
    def plan_market(self, *, now_ts_ms: int, side: str, target_qty: float, snapshot: Dict[str, Any]) -> List[MarketChild]:
        q = float(abs(target_qty))
        if q <= 0.0:
            return []
        return [MarketChild(ts_offset_ms=0, qty=q, liquidity_hint=None)]


class TWAPExecutor(BaseExecutor):
    def __init__(self, *, parts: int = 6, child_interval_s: int = 600):
        self.parts = max(1, int(parts))
        self.child_interval_ms = int(child_interval_s) * 1000

    def plan_market(self, *, now_ts_ms: int, side: str, target_qty: float, snapshot: Dict[str, Any]) -> List[MarketChild]:
        q_total = float(abs(target_qty))
        if q_total <= 0.0:
            return []
        per = q_total / float(self.parts)
        plan: List[MarketChild] = []
        for i in range(self.parts):
            plan.append(MarketChild(ts_offset_ms=i * self.child_interval_ms, qty=per, liquidity_hint=None))
        # скорректируем последнего из-за накопленных округлений
        if plan:
            acc = sum(c.qty for c in plan[:-1])
            plan[-1].qty = max(0.0, q_total - acc)
        return plan


class POVExecutor(BaseExecutor):
    def __init__(self, *, participation: float = 0.1, child_interval_s: int = 60, min_child_notional: float = 20.0):
        self.participation = max(0.0, float(participation))
        self.child_interval_ms = int(child_interval_s) * 1000
        self.min_child_notional = float(min_child_notional)

    def plan_market(self, *, now_ts_ms: int, side: str, target_qty: float, snapshot: Dict[str, Any]) -> List[MarketChild]:
        q_total = float(abs(target_qty))
        if q_total <= 0.0:
            return []

        liq = float(snapshot.get("liquidity") or 0.0)  # прокси «штук за интервал»
        price = float(snapshot.get("ref_price") or snapshot.get("mid") or 0.0)

        # если нет ликвидности, сведём к одному такеру
        if liq <= 0.0 or price <= 0.0 or self.participation <= 0.0:
            return [MarketChild(ts_offset_ms=0, qty=q_total, liquidity_hint=None)]

        per_child_qty = self.participation * liq
        if per_child_qty <= 0.0:
            return [MarketChild(ts_offset_ms=0, qty=q_total, liquidity_hint=None)]

        # обеспечим минимальный notional на ребёнка
        min_qty_by_notional = self.min_child_notional / max(1e-12, price)
        per_child_qty = max(per_child_qty, min_qty_by_notional)

        plan: List[MarketChild] = []
        produced = 0.0
        i = 0
        # ограничим разумно – не более 10k «детей» в план
        while produced + 1e-12 < q_total and i < 10000:
            left = q_total - produced
            q = min(per_child_qty, left)
            plan.append(MarketChild(ts_offset_ms=i * self.child_interval_ms, qty=q, liquidity_hint=liq))
            produced += q
            i += 1
        return plan
