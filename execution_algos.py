# sim/execution_algos.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class MarketChild:
    ts_offset_ms: int
    qty: float
    liquidity_hint: Optional[float] = (
        None  # если хотим переопределить ликвидность на шаге
    )


class BaseExecutor:
    def plan_market(
        self, *, now_ts_ms: int, side: str, target_qty: float, snapshot: Dict[str, Any]
    ) -> List[MarketChild]:
        raise NotImplementedError


class TakerExecutor(BaseExecutor):
    def plan_market(
        self, *, now_ts_ms: int, side: str, target_qty: float, snapshot: Dict[str, Any]
    ) -> List[MarketChild]:
        q = float(abs(target_qty))
        if q <= 0.0:
            return []
        return [MarketChild(ts_offset_ms=0, qty=q, liquidity_hint=None)]


class TWAPExecutor(BaseExecutor):
    """Time-weighted execution with deterministic schedule.

    For a given timestamp and target quantity the resulting child orders are
    always identical.  No randomness is used in planning the schedule.
    """

    def __init__(self, *, parts: int = 6, child_interval_s: int = 600):
        self.parts = max(1, int(parts))
        self.child_interval_ms = int(child_interval_s) * 1000

    def plan_market(
        self, *, now_ts_ms: int, side: str, target_qty: float, snapshot: Dict[str, Any]
    ) -> List[MarketChild]:
        q_total = float(abs(target_qty))
        if q_total <= 0.0:
            return []
        per = q_total / float(self.parts)
        plan: List[MarketChild] = []
        for i in range(self.parts):
            plan.append(
                MarketChild(
                    ts_offset_ms=i * self.child_interval_ms,
                    qty=per,
                    liquidity_hint=None,
                )
            )
        # скорректируем последнего из-за накопленных округлений
        if plan:
            acc = sum(c.qty for c in plan[:-1])
            plan[-1].qty = max(0.0, q_total - acc)
        return plan


class POVExecutor(BaseExecutor):
    """Participation-of-volume execution with deterministic planning.

    The plan depends only on the provided timestamp, liquidity hint and target
    quantity; repeated calls with identical inputs produce identical child
    trajectories.
    """

    def __init__(
        self,
        *,
        participation: float = 0.1,
        child_interval_s: int = 60,
        min_child_notional: float = 20.0,
    ):
        self.participation = max(0.0, float(participation))
        self.child_interval_ms = int(child_interval_s) * 1000
        self.min_child_notional = float(min_child_notional)

    def plan_market(
        self, *, now_ts_ms: int, side: str, target_qty: float, snapshot: Dict[str, Any]
    ) -> List[MarketChild]:
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
            plan.append(
                MarketChild(
                    ts_offset_ms=i * self.child_interval_ms, qty=q, liquidity_hint=liq
                )
            )
            produced += q
            i += 1
        return plan


class MarketOpenH1Executor(BaseExecutor):
    def plan_market(
        self,
        *,
        now_ts_ms: int,
        side: str,
        target_qty: float,
        snapshot: Dict[str, Any],
    ) -> List[MarketChild]:
        q = float(abs(target_qty))
        if q <= 0.0:
            return []
        hour_ms = 3_600_000
        next_open = ((now_ts_ms // hour_ms) + 1) * hour_ms
        offset = int(max(0, next_open - now_ts_ms))
        return [MarketChild(ts_offset_ms=offset, qty=q, liquidity_hint=None)]


class VWAPExecutor(BaseExecutor):
    def plan_market(
        self,
        *,
        now_ts_ms: int,
        side: str,
        target_qty: float,
        snapshot: Dict[str, Any],
    ) -> List[MarketChild]:
        q = float(abs(target_qty))
        if q <= 0.0:
            return []
        hour_ms = 3_600_000
        end = ((now_ts_ms // hour_ms) + 1) * hour_ms
        offset = int(max(0, end - now_ts_ms))
        return [MarketChild(ts_offset_ms=offset, qty=q, liquidity_hint=None)]


class MidOffsetLimitExecutor(BaseExecutor):
    """Generate a single limit order around the mid price.

    The executor does **not** plan market child orders like the other
    executors above.  Instead it builds an ``ActionProto`` (or a compatible
    dictionary) describing a LIMIT order placed at ``mid*(1±offset)`` where the
    sign of the offset depends on the order side.

    Parameters
    ----------
    offset_bps: float
        Offset from mid in basis points.  Positive values move buys above the
        mid and sells below the mid.
    ttl_steps: int
        Optional TTL in simulation steps.
    tif: str
        Time-in-force policy: ``"GTC"``, ``"IOC"`` or ``"FOK"``.
    """

    def __init__(
        self, *, offset_bps: float = 0.0, ttl_steps: int = 0, tif: str = "GTC"
    ):
        self.offset_bps = float(offset_bps)
        self.ttl_steps = int(ttl_steps)
        self.tif = str(tif)

    def build_action(self, *, side: str, qty: float, snapshot: Dict[str, Any]):
        """Return a limit ``ActionProto`` based on the snapshot mid price.

        If the ``action_proto`` module is unavailable, a dictionary with
        equivalent fields is returned.
        """
        mid = snapshot.get("mid")
        q = float(abs(qty))
        if mid is None or q <= 0.0:
            return None

        offset = self.offset_bps / 10_000.0
        if str(side).upper() == "BUY":
            price = float(mid) * (1.0 + offset)
            vol = q
        else:
            price = float(mid) * (1.0 - offset)
            vol = -q

        try:  # попытаться вернуть настоящий ActionProto
            from action_proto import ActionProto, ActionType  # type: ignore

            return ActionProto(
                action_type=ActionType.LIMIT,
                volume_frac=vol,
                ttl_steps=self.ttl_steps,
                abs_price=float(price),
                tif=str(self.tif),
            )
        except Exception:  # pragma: no cover - fallback для минимальных окружений
            return {
                "action_type": 2,  # LIMIT
                "volume_frac": vol,
                "ttl_steps": self.ttl_steps,
                "abs_price": float(price),
                "tif": str(self.tif),
            }


def make_executor(algo: str, cfg: Dict[str, Any] | None = None) -> BaseExecutor:
    """Factory helper for building execution algos.

    Parameters
    ----------
    algo:
        Algorithm name (e.g. ``"TAKER"``, ``"TWAP"`` or ``"POV"``).
    cfg:
        Optional configuration mapping used to extract algorithm-specific
        parameters.  For ``TWAP`` and ``POV`` the helper looks for ``"twap"``
        and ``"pov"`` sub-dictionaries respectively.
    """

    cfg = dict(cfg or {})
    a = str(algo).upper()
    if a == "TWAP":
        tw = dict(cfg.get("twap", {}))
        parts = int(tw.get("parts", 6))
        interval = int(tw.get("child_interval_s", 600))
        return TWAPExecutor(parts=parts, child_interval_s=interval)
    if a == "POV":
        pv = dict(cfg.get("pov", {}))
        part = float(pv.get("participation", 0.10))
        interval = int(pv.get("child_interval_s", 60))
        min_not = float(pv.get("min_child_notional", 20.0))
        return POVExecutor(
            participation=part, child_interval_s=interval, min_child_notional=min_not
        )
    if a == "VWAP":
        return VWAPExecutor()
    return TakerExecutor()
