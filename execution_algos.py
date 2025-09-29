# sim/execution_algos.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


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


class _BarWindowAware:
    """Mixin providing helpers for executors using bar timeframe metadata."""

    def __init__(self) -> None:
        self._last_bar_timeframe_ms: Optional[int] = None
        self._last_bar_start_ts: Optional[int] = None
        self._last_bar_end_ts: Optional[int] = None

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            iv = int(value)
        except (TypeError, ValueError):
            return None
        return iv

    def _resolve_bar_window(
        self, now_ts_ms: int, snapshot: Mapping[str, Any]
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
        timeframe = self._coerce_int(
            snapshot.get("bar_timeframe_ms")
            or snapshot.get("timeframe_ms")
            or snapshot.get("intrabar_timeframe_ms")
        )
        if timeframe is not None and timeframe <= 0:
            timeframe = None
        if timeframe is not None:
            self._last_bar_timeframe_ms = timeframe
        else:
            timeframe = self._last_bar_timeframe_ms

        start = self._coerce_int(
            snapshot.get("bar_start_ts")
            or snapshot.get("intrabar_start_ts")
            or snapshot.get("bar_start_ts_ms")
        )
        if start is not None:
            self._last_bar_start_ts = start
        else:
            start = self._last_bar_start_ts

        end = self._coerce_int(
            snapshot.get("bar_end_ts")
            or snapshot.get("intrabar_end_ts")
            or snapshot.get("bar_end_ts_ms")
        )
        if end is not None:
            self._last_bar_end_ts = end
        else:
            end = self._last_bar_end_ts

        if timeframe is not None:
            if start is None and end is not None:
                start = end - timeframe
            if end is None and start is not None:
                end = start + timeframe

        if timeframe is None and start is not None and end is not None:
            diff = end - start
            if diff > 0:
                timeframe = diff

        if timeframe is not None and timeframe > 0:
            if start is None:
                start = (now_ts_ms // timeframe) * timeframe
            if end is None and start is not None:
                end = start + timeframe
        else:
            timeframe = None

        if timeframe is not None and timeframe > 0:
            self._last_bar_timeframe_ms = timeframe
        if start is not None:
            self._last_bar_start_ts = start
        if end is not None:
            self._last_bar_end_ts = end
        return timeframe, start, end


class TWAPExecutor(_BarWindowAware, BaseExecutor):
    """Time-weighted execution with deterministic schedule.

    For a given timestamp and target quantity the resulting child orders are
    always identical.  No randomness is used in planning the schedule.
    """

    def __init__(self, *, parts: int = 6, child_interval_s: int = 600):
        super().__init__()
        self.parts = max(1, int(parts))
        self.child_interval_ms = int(child_interval_s) * 1000

    def plan_market(
        self, *, now_ts_ms: int, side: str, target_qty: float, snapshot: Dict[str, Any]
    ) -> List[MarketChild]:
        q_total = float(abs(target_qty))
        if q_total <= 0.0:
            return []
        per = q_total / float(self.parts)
        timeframe_ms, bar_start, bar_end = self._resolve_bar_window(
            now_ts_ms, snapshot
        )
        plan: List[MarketChild] = []
        if timeframe_ms is not None and timeframe_ms > 0:
            denominator = max(1, self.parts - 1)
            try:
                anchor = int(bar_start if bar_start is not None else now_ts_ms)
            except (TypeError, ValueError):
                anchor = int(now_ts_ms)
            for i in range(self.parts):
                frac = float(i) / float(denominator) if denominator else 0.0
                offset_abs = anchor + int(round(frac * timeframe_ms))
                if bar_end is not None and offset_abs > bar_end:
                    offset_abs = int(bar_end)
                offset = max(0, int(offset_abs - now_ts_ms))
                plan.append(
                    MarketChild(
                        ts_offset_ms=offset,
                        qty=per,
                        liquidity_hint=None,
                    )
                )
        else:
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


class POVExecutor(_BarWindowAware, BaseExecutor):
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
        super().__init__()
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
        timeframe_ms, bar_start, bar_end = self._resolve_bar_window(
            now_ts_ms, snapshot
        )
        if timeframe_ms is not None and timeframe_ms > 0:
            total_children = int(math.ceil(q_total / per_child_qty))
            total_children = max(1, min(total_children, 10000))
            denominator = max(1, total_children - 1)
            try:
                anchor = int(bar_start if bar_start is not None else now_ts_ms)
            except (TypeError, ValueError):
                anchor = int(now_ts_ms)
            offsets: List[int] = []
            for i in range(total_children):
                frac = float(i) / float(denominator) if denominator else 0.0
                offset_abs = anchor + int(round(frac * timeframe_ms))
                if bar_end is not None and offset_abs > bar_end:
                    offset_abs = int(bar_end)
                offsets.append(max(0, int(offset_abs - now_ts_ms)))
            for offset in offsets:
                if produced + 1e-12 >= q_total:
                    break
                left = q_total - produced
                q = min(per_child_qty, left)
                plan.append(
                    MarketChild(
                        ts_offset_ms=offset, qty=q, liquidity_hint=liq
                    )
                )
                produced += q
        else:
            i = 0
            while produced + 1e-12 < q_total and i < 10000:
                left = q_total - produced
                q = min(per_child_qty, left)
                plan.append(
                    MarketChild(
                        ts_offset_ms=i * self.child_interval_ms,
                        qty=q,
                        liquidity_hint=liq,
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
    """Volume-weighted execution using intrabar volume profile when available."""

    def __init__(self, *, fallback_parts: int = 6) -> None:
        self.fallback_parts = max(1, int(fallback_parts))

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            iv = int(value)
        except (TypeError, ValueError):
            return None
        return iv

    @staticmethod
    def _extract_profile_entry(
        entry: Any,
        *,
        bar_start: Optional[int],
        timeframe: Optional[int],
    ) -> Optional[tuple[int, float]]:
        ts_raw: Any = None
        vol_raw: Any = None
        if isinstance(entry, Mapping):
            data = dict(entry)
            ts_raw = (
                data.get("ts")
                or data.get("timestamp")
                or data.get("ts_ms")
                or data.get("time")
            )
            if ts_raw is None and bar_start is not None:
                offset_raw = data.get("offset_ms")
                if offset_raw is not None:
                    try:
                        ts_raw = int(bar_start + float(offset_raw))
                    except (TypeError, ValueError):
                        ts_raw = None
            if ts_raw is None and timeframe is not None and timeframe > 0:
                frac_raw = data.get("fraction")
                if frac_raw is not None and bar_start is not None:
                    try:
                        ts_raw = int(bar_start + float(frac_raw) * float(timeframe))
                    except (TypeError, ValueError):
                        ts_raw = None
            vol_raw = (
                data.get("volume")
                or data.get("qty")
                or data.get("quantity")
                or data.get("vol")
            )
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            seq = list(entry)
            if seq:
                ts_raw = seq[0]
            if len(seq) > 1:
                vol_raw = seq[1]
        else:
            return None

        try:
            vol = float(vol_raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(vol) or vol <= 0.0:
            return None

        ts_val: Optional[int] = None
        if isinstance(ts_raw, (int, float)) and not isinstance(ts_raw, bool):
            if math.isfinite(float(ts_raw)):
                ts_val = int(round(float(ts_raw)))
        elif isinstance(ts_raw, str):
            try:
                ts_val = int(round(float(ts_raw.strip())))
            except (TypeError, ValueError):
                ts_val = None
        if ts_val is None:
            return None
        return ts_val, float(vol)

    def _fallback_plan(
        self,
        *,
        now_ts_ms: int,
        total_qty: float,
        bar_end_ts: Optional[int],
        timeframe_ms: Optional[int],
    ) -> List[MarketChild]:
        if total_qty <= 0.0:
            return []
        horizon_ms: Optional[int] = None
        if bar_end_ts is not None and bar_end_ts > now_ts_ms:
            horizon_ms = bar_end_ts - now_ts_ms
        elif timeframe_ms is not None and timeframe_ms > 0:
            horizon_ms = timeframe_ms
        if horizon_ms is None or horizon_ms < 0:
            horizon_ms = 0

        parts = int(self.fallback_parts)
        if parts <= 1:
            return [MarketChild(ts_offset_ms=0, qty=total_qty, liquidity_hint=None)]

        denominator = max(1, parts - 1)
        offsets: List[int] = []
        for i in range(parts):
            if horizon_ms > 0:
                frac = float(i) / float(denominator)
                offsets.append(int(round(frac * horizon_ms)))
            else:
                offsets.append(0)

        per = total_qty / float(parts)
        plan: List[MarketChild] = []
        produced = 0.0
        for i, offset in enumerate(offsets):
            qty = per
            if i == parts - 1:
                qty = max(0.0, total_qty - produced)
            plan.append(
                MarketChild(
                    ts_offset_ms=max(0, int(offset)),
                    qty=qty,
                    liquidity_hint=None,
                )
            )
            produced += qty
        return [child for child in plan if child.qty > 0.0]

    def plan_market(
        self,
        *,
        now_ts_ms: int,
        side: str,
        target_qty: float,
        snapshot: Dict[str, Any],
    ) -> List[MarketChild]:
        total_qty = float(abs(target_qty))
        if total_qty <= 0.0:
            return []

        timeframe_ms = self._coerce_int(
            snapshot.get("bar_timeframe_ms")
            or snapshot.get("timeframe_ms")
            or snapshot.get("intrabar_timeframe_ms")
        )
        bar_start = self._coerce_int(
            snapshot.get("bar_start_ts")
            or snapshot.get("intrabar_start_ts")
            or snapshot.get("bar_start_ts_ms")
        )
        bar_end = self._coerce_int(
            snapshot.get("bar_end_ts")
            or snapshot.get("intrabar_end_ts")
            or snapshot.get("bar_end_ts_ms")
        )

        if bar_start is not None and bar_end is None and timeframe_ms:
            bar_end = bar_start + timeframe_ms
        if bar_end is not None and bar_start is None and timeframe_ms:
            bar_start = bar_end - timeframe_ms

        if timeframe_ms is None and bar_start is not None and bar_end is not None:
            timeframe_ms = max(0, bar_end - bar_start)

        if timeframe_ms is None or timeframe_ms <= 0:
            hour_ms = 3_600_000
            timeframe_ms = hour_ms
        if bar_start is None:
            bar_start = (now_ts_ms // timeframe_ms) * timeframe_ms
        if bar_end is None:
            bar_end = bar_start + timeframe_ms

        profile_raw = snapshot.get("intrabar_volume_profile") or []
        entries: List[tuple[int, float]] = []
        if isinstance(profile_raw, Sequence) and not isinstance(profile_raw, (str, bytes)):
            for node in profile_raw:
                extracted = self._extract_profile_entry(
                    node, bar_start=bar_start, timeframe=timeframe_ms
                )
                if extracted is None:
                    continue
                ts_val, vol_val = extracted
                if bar_start is not None and ts_val < bar_start:
                    ts_val = bar_start
                if bar_end is not None and ts_val > bar_end:
                    ts_val = bar_end
                entries.append((ts_val, vol_val))

        if not entries:
            return self._fallback_plan(
                now_ts_ms=now_ts_ms,
                total_qty=total_qty,
                bar_end_ts=bar_end,
                timeframe_ms=timeframe_ms,
            )

        aggregated: Dict[int, float] = {}
        for ts_val, vol_val in entries:
            if ts_val < now_ts_ms:
                continue
            aggregated[ts_val] = aggregated.get(ts_val, 0.0) + vol_val

        if not aggregated:
            return self._fallback_plan(
                now_ts_ms=now_ts_ms,
                total_qty=total_qty,
                bar_end_ts=bar_end,
                timeframe_ms=timeframe_ms,
            )

        timestamps = sorted(aggregated.keys())
        total_volume = sum(aggregated[ts_val] for ts_val in timestamps)
        if total_volume <= 0.0:
            return self._fallback_plan(
                now_ts_ms=now_ts_ms,
                total_qty=total_qty,
                bar_end_ts=bar_end,
                timeframe_ms=timeframe_ms,
            )

        plan: List[MarketChild] = []
        produced_qty = 0.0
        for idx, ts_val in enumerate(timestamps):
            vol_share = aggregated[ts_val] / total_volume
            qty = total_qty * vol_share
            if idx == len(timestamps) - 1:
                qty = max(0.0, total_qty - produced_qty)
            offset = ts_val - now_ts_ms
            if offset < 0:
                offset = 0
            plan.append(
                MarketChild(
                    ts_offset_ms=int(offset),
                    qty=qty,
                    liquidity_hint=qty,
                )
            )
            produced_qty += qty

        return [child for child in plan if child.qty > 0.0]


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
