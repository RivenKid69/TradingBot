import importlib.util
import json
import os
import pathlib
import sys
import tempfile

import pytest

base = pathlib.Path(__file__).resolve().parents[1]

# Load execution simulator
spec_exec = importlib.util.spec_from_file_location("execution_sim", base / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec_exec)
sys.modules["execution_sim"] = exec_mod
spec_exec.loader.exec_module(exec_mod)
ActionProto = exec_mod.ActionProto
ActionType = exec_mod.ActionType
ExecutionSimulator = exec_mod.ExecutionSimulator

# Load quantizer and constants
spec_quant = importlib.util.spec_from_file_location("quantizer", base / "quantizer.py")
quant_mod = importlib.util.module_from_spec(spec_quant)
sys.modules["quantizer"] = quant_mod
spec_quant.loader.exec_module(quant_mod)
Quantizer = quant_mod.Quantizer

spec_impl = importlib.util.spec_from_file_location("impl_quantizer", base / "impl_quantizer.py")
impl_mod = importlib.util.module_from_spec(spec_impl)
sys.modules["impl_quantizer"] = impl_mod
spec_impl.loader.exec_module(impl_mod)
QuantizerImpl = impl_mod.QuantizerImpl

spec_const = importlib.util.spec_from_file_location("core_constants", base / "core_constants.py")
const_mod = importlib.util.module_from_spec(spec_const)
sys.modules["core_constants"] = const_mod
spec_const.loader.exec_module(const_mod)
PRICE_SCALE = const_mod.PRICE_SCALE

from fast_lob import CythonLOB

# Shared filters for tests
filters = {
    "BTCUSDT": {
        "PRICE_FILTER": {"minPrice": "0", "maxPrice": "1000000", "tickSize": "0.5"},
        "LOT_SIZE": {"minQty": "0.1", "maxQty": "1000", "stepSize": "0.1"},
        "MIN_NOTIONAL": {"minNotional": "5"},
        "PERCENT_PRICE_BY_SIDE": {
            "multiplierUp": "1000",
            "multiplierDown": "0",
            "bidMultiplierUp": "1000",
            "bidMultiplierDown": "0",
            "askMultiplierUp": "1000",
            "askMultiplierDown": "0",
        },
    }
}


def make_sim(strict: bool) -> ExecutionSimulator:
    sim = ExecutionSimulator(filters_path=None)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"filters": filters}, fh)
        temp_path = fh.name
    try:
        cfg = {
            "path": temp_path,
            "filters_path": temp_path,
            "strict_filters": bool(strict),
            "enforce_percent_price_by_side": True,
        }
        impl = QuantizerImpl.from_dict(cfg)
        sim.attach_quantizer(impl=impl)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    return sim


class _FakeRisk:
    def __init__(self, adjusted_qty: float):
        self.qty = float(adjusted_qty)
        self._events: list = []
        self.paused_until_ms = 0

    def pre_trade_adjust(self, **_kwargs):
        return self.qty

    def pop_events(self):
        events = list(self._events)
        self._events.clear()
        return events

    def can_send_order(self, *_args, **_kwargs) -> bool:
        return True

    def on_new_order(self, *_args, **_kwargs) -> None:
        return None

    def _emit(self, *_args, **_kwargs) -> None:
        return None

    def on_mark(self, **_kwargs) -> None:
        return None


def add_limit_with_filters(lob: CythonLOB, is_buy: bool, price: float, qty: float, q: Quantizer):
    p_abs = q.quantize_price("BTCUSDT", price)
    q_qty = q.quantize_qty("BTCUSDT", qty)
    q_qty = q.clamp_notional("BTCUSDT", p_abs if p_abs > 0 else price, q_qty)
    p_ticks = int(round(p_abs * PRICE_SCALE))
    if p_ticks != int(round(price * PRICE_SCALE)) or abs(q_qty - qty) > 1e-12:
        return 0, 0
    return lob.add_limit_order(is_buy, p_ticks, q_qty, 0, True)


class _ClampNotionalQuantizer:
    def __init__(self, forced_qty: float):
        self._forced_qty = float(forced_qty)

    def quantize_price(self, _symbol: str, price: float) -> float:
        return float(price)

    def quantize_qty(self, _symbol: str, qty: float) -> float:
        return float(qty)

    def clamp_notional(self, _symbol: str, _ref_price: float, qty: float) -> float:
        return max(float(qty), self._forced_qty)


class _LegacyQuantizerWithoutQuantizeOrder:
    def __init__(self, filters_payload):
        self._filters_raw = filters_payload
        self._delegate = Quantizer(filters_payload, strict=True)
        self._filters = self._delegate._filters
        self.strict = True

    def raw_filters(self):
        return self._filters_raw

    def quantize_price(self, symbol: str, price: float) -> float:
        return float(self._delegate.quantize_price(symbol, price))

    def quantize_qty(self, symbol: str, qty: float) -> float:
        return float(self._delegate.quantize_qty(symbol, qty))

    def clamp_notional(self, symbol: str, ref_price: float, qty: float) -> float:
        return float(self._delegate.clamp_notional(symbol, ref_price, qty))

    def check_percent_price_by_side(
        self, symbol: str, side: str, price: float, ref_price: float
    ) -> bool:
        return bool(
            self._delegate.check_percent_price_by_side(symbol, side, price, ref_price)
        )


class _LegacyQuantizerWithoutPercentPriceCheck:
    def __init__(self, filters_payload):
        self._filters_raw = filters_payload
        self._delegate = Quantizer(filters_payload, strict=True)
        self._filters = self._delegate._filters
        self.strict = True

    def raw_filters(self):
        return self._filters_raw

    def quantize_price(self, symbol: str, price: float) -> float:
        return float(self._delegate.quantize_price(symbol, price))

    def quantize_qty(self, symbol: str, qty: float) -> float:
        return float(self._delegate.quantize_qty(symbol, qty))

    def clamp_notional(self, symbol: str, ref_price: float, qty: float) -> float:
        return float(self._delegate.clamp_notional(symbol, ref_price, qty))


# --- Python ExecutionSimulator tests ---

def test_unquantized_limit_executes_permissive():
    sim = make_sim(strict=False)
    proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=0.05, abs_price=100.3)
    oid = sim.submit(proto)
    report = sim.pop_ready(ref_price=100.0)

    assert report.cancelled_ids == []
    assert report.new_order_ids == []
    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.client_order_id == oid
    expected_price = sim.quantizer.quantize_price("BTCUSDT", proto.abs_price)
    expected_qty = sim.quantizer.quantize_qty("BTCUSDT", abs(proto.volume_frac))
    expected_qty = sim.quantizer.clamp_notional(
        "BTCUSDT",
        expected_price if expected_price > 0 else proto.abs_price,
        expected_qty,
    )
    assert trade.price == pytest.approx(expected_price)
    assert trade.qty == pytest.approx(expected_qty)
    assert sim._last_bid is None and sim._last_ask is None
    assert sim.strict_filters is False


def test_unquantized_limit_rejected_strict():
    sim = make_sim(strict=True)
    assert sim.strict_filters is True
    proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=0.02, abs_price=101.0)
    oid = sim.submit(proto)
    report = sim.pop_ready(ref_price=100.0)

    assert report.cancelled_ids == []
    assert report.new_order_ids == []
    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.client_order_id == oid
    expected_price = sim.quantizer.quantize_price("BTCUSDT", proto.abs_price)
    expected_qty = sim.quantizer.quantize_qty("BTCUSDT", abs(proto.volume_frac))
    expected_qty = sim.quantizer.clamp_notional(
        "BTCUSDT",
        expected_price if expected_price > 0 else proto.abs_price,
        expected_qty,
    )
    assert trade.price == pytest.approx(expected_price)
    assert trade.qty == pytest.approx(expected_qty)
    assert sim._last_bid is None and sim._last_ask is None


def test_cancel_all_cancels_open_limits():
    sim = ExecutionSimulator(filters_path=None)
    limit_proto = ActionProto(
        action_type=ActionType.LIMIT,
        volume_frac=0.5,
        abs_price=100.0,
        ttl_steps=2,
    )
    limit_id = sim.submit(limit_proto)

    first_report = sim.pop_ready(ref_price=100.0)
    assert first_report.trades == []
    assert limit_id in first_report.new_order_ids
    assert (limit_id, 2) in sim._ttl_orders

    cancel_proto = ActionProto(action_type=ActionType.CANCEL_ALL, volume_frac=0.0)
    cancel_id = sim.submit(cancel_proto)
    cancel_report = sim.pop_ready(ref_price=100.0)

    assert cancel_report.trades == []
    assert limit_id in cancel_report.cancelled_ids
    assert cancel_id in cancel_report.cancelled_ids
    assert cancel_report.new_order_ids == []
    assert sim._ttl_orders == []
    assert cancel_report.cancelled_reasons[limit_id] == "CANCEL_ALL"


def test_market_quantity_rounded_up_passes_filters():
    sim = ExecutionSimulator(filters_path=None)
    sim.set_quantizer(Quantizer(filters, strict=True))

    qty_total, rejection = sim._apply_filters_market("BUY", 0.099, ref_price=100.0)

    assert qty_total == pytest.approx(0.1)
    assert rejection is None


def test_market_filters_enforced_without_quantizer(monkeypatch, tmp_path):
    filters_file = tmp_path / "filters.json"
    filters_file.write_text(json.dumps({"filters": filters}))

    monkeypatch.setattr(exec_mod, "Quantizer", None)
    monkeypatch.setattr(exec_mod, "load_filters", None, raising=False)
    monkeypatch.setattr(exec_mod, "QuantizerImpl", None)

    sim = ExecutionSimulator(filters_path=str(filters_file), strict_filters=True)

    assert sim.quantizer is None
    assert sim.quantizer_impl is None
    assert "BTCUSDT" in sim.filters
    assert sim.strict_filters is True

    qty_total, rejection = sim._apply_filters_market("BUY", 0.05, ref_price=100.0)

    assert qty_total == pytest.approx(0.0)
    assert rejection is not None
    assert rejection.code == "LOT_SIZE"


def test_market_clamp_notional_growth_rejected_by_qty_limits():
    ref_price = 50.0
    local_filters = {
        "TESTUSDT": {
            "PRICE_FILTER": {"minPrice": "0", "maxPrice": "100000", "tickSize": "0.1"},
            "LOT_SIZE": {"minQty": "0.1", "maxQty": "5", "stepSize": "0.1"},
            "MIN_NOTIONAL": {"minNotional": "400"},
        }
    }
    forced_qty = float(local_filters["TESTUSDT"]["MIN_NOTIONAL"]["minNotional"]) / ref_price

    sim = ExecutionSimulator(symbol="TESTUSDT", filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = False
    sim.filters = local_filters
    sim.quantizer = _ClampNotionalQuantizer(forced_qty)

    qty_total, rejection = sim._apply_filters_market("BUY", 1.0, ref_price=ref_price)

    assert qty_total == 0.0
    assert rejection is not None
    assert rejection.code == "LOT_SIZE"
    assert rejection.message == "Quantity above maximum"


def test_capacity_normalization_caps_to_requested_qty_without_quantizer():
    sim = ExecutionSimulator(symbol="BTCUSDT", filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = False
    sim.quantizer = None
    sim.filters = json.loads(json.dumps(filters))

    normalized, rejection = sim._normalize_capacity_quantity(
        0.15, remaining_base=0.3, ref_price=100.0
    )

    assert rejection is None
    assert normalized == pytest.approx(0.1)


def test_lowercase_filters_enforce_strict_checks():
    ref_price_high = 10000.0
    ref_price_low = 50.0
    lowercase_filters = {
        "testusdt": {
            "PRICE_FILTER": {"minPrice": "0", "maxPrice": "100000", "tickSize": "0.1"},
            "LOT_SIZE": {"minQty": "0.1", "maxQty": "5", "stepSize": "0.1"},
            "MIN_NOTIONAL": {"minNotional": "400"},
        }
    }

    sim = ExecutionSimulator(symbol="TESTUSDT", filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = False
    sim.set_quantizer(Quantizer(lowercase_filters, strict=True))

    assert "TESTUSDT" in sim.filters

    order_check = sim.quantizer_impl.validate_order(
        "TESTUSDT", "BUY", ref_price_high, 0.05, ref_price_high, enforce_ppbs=False
    )
    assert order_check.reason_code is None
    assert order_check.qty == pytest.approx(0.1)

    qty_total, rejection = sim._apply_filters_market("BUY", 0.05, ref_price=ref_price_high)
    assert qty_total == pytest.approx(0.1)
    assert rejection is None

    forced_qty = (
        float(lowercase_filters["testusdt"]["MIN_NOTIONAL"]["minNotional"]) / ref_price_low
    )
    qty_total, rejection = sim._apply_filters_market("BUY", forced_qty, ref_price=ref_price_low)
    assert qty_total == pytest.approx(0.0)
    assert rejection is not None
    assert rejection.code == "MIN_NOTIONAL"


def test_legacy_quantizer_without_quantize_order_rejects_min_notional():
    legacy_filters = {
        "LEGACYMN": {
            "PRICE_FILTER": {"minPrice": "0.01", "maxPrice": "100000", "tickSize": "0.01"},
            "LOT_SIZE": {"minQty": "0.1", "maxQty": "0.5", "stepSize": "0.1"},
            "MIN_NOTIONAL": {"minNotional": "10"},
        }
    }

    sim = ExecutionSimulator(symbol="LEGACYMN", filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = True
    sim.set_quantizer(_LegacyQuantizerWithoutQuantizeOrder(legacy_filters))

    result = sim.quantizer_impl.validate_order(
        "LEGACYMN", "BUY", 10.0, 0.2, 10.0, enforce_ppbs=True
    )

    assert result.reason_code == "MIN_NOTIONAL"
    assert result.qty == pytest.approx(0.0)
    details = getattr(result, "details", None)
    assert details is not None
    assert details.get("min_notional") == pytest.approx(10.0)


def test_legacy_quantizer_without_quantize_order_rejects_percent_price():
    legacy_filters = {
        "LEGACYPBS": {
            "PRICE_FILTER": {"minPrice": "0.01", "maxPrice": "100000", "tickSize": "0.01"},
            "LOT_SIZE": {"minQty": "0.001", "maxQty": "10", "stepSize": "0.001"},
            "MIN_NOTIONAL": {"minNotional": "0.5"},
            "PERCENT_PRICE_BY_SIDE": {"multiplierUp": "1.05", "multiplierDown": "0.95"},
        }
    }

    sim = ExecutionSimulator(symbol="LEGACYPBS", filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = True
    sim.set_quantizer(_LegacyQuantizerWithoutQuantizeOrder(legacy_filters))

    result = sim.quantizer_impl.validate_order(
        "LEGACYPBS", "BUY", 110.0, 0.01, 100.0, enforce_ppbs=True
    )

    assert result.reason_code == "PPBS"
    assert result.qty == pytest.approx(0.0)
    details = getattr(result, "details", None)
    assert details is not None
    assert details.get("multiplier_up") == pytest.approx(1.05)


def test_legacy_quantizer_percent_price_fallback_rejects_both_limits():
    legacy_filters = {
        "LEGACYFALLBACK": {
            "PRICE_FILTER": {"minPrice": "0.01", "maxPrice": "100000", "tickSize": "0.01"},
            "LOT_SIZE": {"minQty": "0.001", "maxQty": "10", "stepSize": "0.001"},
            "MIN_NOTIONAL": {"minNotional": "0.5"},
            "PERCENT_PRICE_BY_SIDE": {"multiplierUp": "1.05", "multiplierDown": "0.95"},
        }
    }

    sim = ExecutionSimulator(symbol="LEGACYFALLBACK", filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = True
    sim.set_quantizer(_LegacyQuantizerWithoutPercentPriceCheck(legacy_filters))

    buy_result = sim.quantizer_impl.validate_order(
        "LEGACYFALLBACK", "BUY", 90.0, 0.01, 100.0, enforce_ppbs=True
    )
    assert buy_result.reason_code == "PPBS"

    sell_result = sim.quantizer_impl.validate_order(
        "LEGACYFALLBACK", "SELL", 110.0, 0.01, 100.0, enforce_ppbs=True
    )
    assert sell_result.reason_code == "PPBS"

    buy_ok = sim.quantizer_impl.validate_order(
        "LEGACYFALLBACK", "BUY", 100.0, 0.01, 100.0, enforce_ppbs=True
    )
    assert buy_ok.reason_code is None

    sell_ok = sim.quantizer_impl.validate_order(
        "LEGACYFALLBACK", "SELL", 100.0, 0.01, 100.0, enforce_ppbs=True
    )
    assert sell_ok.reason_code is None


def test_next_open_risk_adjustment_revalidated_by_filters():
    sim = make_sim(strict=True)
    sim._last_ref_price = 100.0
    sim.risk = _FakeRisk(0.105)

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=0.1)
    cid = sim._submit_next_open(proto, now_ts=123456)

    state = sim._pending_next_open.get("BUY")
    assert state is not None
    assert state.qty == pytest.approx(0.1)
    assert cid not in sim._next_open_cancelled


def test_next_open_risk_adjustment_rejected_when_off_filters():
    sim = ExecutionSimulator(filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = False
    sim.quantizer = None
    sim.filters = json.loads(json.dumps(filters))
    sim._last_ref_price = 100.0
    sim.risk = _FakeRisk(0.05)

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=0.1)
    cid = sim._submit_next_open(proto, now_ts=123456)

    assert cid in sim._next_open_cancelled
    assert sim._next_open_cancelled_reasons[cid] == "LOT_SIZE"
    assert not sim._pending_next_open


def test_market_step_violation_rejected_without_quantizer():
    sim = ExecutionSimulator(filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = False
    sim.quantizer = None
    sim.filters = json.loads(json.dumps(filters))

    qty_total, rejection = sim._apply_filters_market("BUY", 0.105, ref_price=100.0)

    assert qty_total == pytest.approx(0.0)
    assert rejection is not None
    assert rejection.code == "LOT_SIZE"
    assert rejection.message == "Quantity not aligned to step"


def test_market_qty_max_enforced_with_sub_picosecond_step():
    max_qty = 1e-9
    qty_step = 2e-15
    local_filters = {
        "FEMTOQTY": {
            "PRICE_FILTER": {
                "minPrice": "0",
                "maxPrice": "1000000",
                "tickSize": "0.0001",
            },
            "LOT_SIZE": {
                "minQty": "0",
                "maxQty": f"{max_qty}",
                "stepSize": f"{qty_step}",
            },
            "MIN_NOTIONAL": {"minNotional": "0"},
        }
    }

    sim = ExecutionSimulator(symbol="FEMTOQTY", filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = False
    sim.quantizer = None
    sim.filters = local_filters

    qty_total, rejection = sim._apply_filters_market(
        "BUY", max_qty + 1e-13, ref_price=1.0
    )

    assert qty_total == pytest.approx(0.0)
    assert rejection is not None
    assert rejection.code == "LOT_SIZE"
    assert rejection.message == "Quantity above maximum"


def test_limit_ppbs_violation_without_quantizer():
    sim = ExecutionSimulator(filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = True
    sim.quantizer = None
    sim.filters = {
        "BTCUSDT": {
            "PRICE_FILTER": {
                "minPrice": "0",
                "maxPrice": "200",
                "tickSize": "0.1",
            },
            "LOT_SIZE": {
                "minQty": "0.1",
                "maxQty": "100",
                "stepSize": "0.1",
            },
            "MIN_NOTIONAL": {"minNotional": "5"},
            "PERCENT_PRICE_BY_SIDE": {
                "multiplierUp": "1.5",
                "multiplierDown": "0.5",
                "bidMultiplierUp": "1.01",
                "bidMultiplierDown": "0.99",
                "askMultiplierUp": "1.5",
                "askMultiplierDown": "0.5",
            },
        }
    }

    price_adj, qty_adj, rejection = sim._apply_filters_limit(
        "BUY", 101.2, 0.5, ref_price=100.0
    )

    assert price_adj == pytest.approx(101.2)
    assert qty_adj == pytest.approx(0.0)
    assert rejection is not None
    assert rejection.code == "PPBS"
    assert rejection.message == "PERCENT_PRICE_BY_SIDE violation"


def test_limit_ppbs_skipped_when_strict_filters_disabled():
    sim = ExecutionSimulator(filters_path=None)
    sim.strict_filters = False
    sim.enforce_ppbs = True
    sim.quantizer = None
    sim.filters = {
        "BTCUSDT": {
            "PRICE_FILTER": {
                "minPrice": "0",
                "maxPrice": "200",
                "tickSize": "0.1",
            },
            "LOT_SIZE": {
                "minQty": "0.1",
                "maxQty": "100",
                "stepSize": "0.1",
            },
            "MIN_NOTIONAL": {"minNotional": "5"},
            "PERCENT_PRICE_BY_SIDE": {
                "multiplierUp": "1.5",
                "multiplierDown": "0.5",
                "bidMultiplierUp": "1.01",
                "bidMultiplierDown": "0.99",
                "askMultiplierUp": "1.5",
                "askMultiplierDown": "0.5",
            },
        }
    }

    price_adj, qty_adj, rejection = sim._apply_filters_limit(
        "BUY", 170.0, 0.5, ref_price=100.0
    )

    assert rejection is None
    assert price_adj == pytest.approx(170.0)
    assert qty_adj == pytest.approx(0.5)


def test_market_pop_ready_requantizes_post_risk_quantity():
    sim = make_sim(strict=True)
    sim._last_ref_price = 100.0
    sim.risk = _FakeRisk(0.105)

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=0.1)
    oid = sim.submit(proto)
    report = sim.pop_ready(ref_price=100.0)

    assert report.cancelled_ids == []
    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.client_order_id == oid
    assert trade.qty == pytest.approx(0.1)


def test_market_pop_ready_rejects_post_risk_filter_violation():
    sim = ExecutionSimulator(filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = False
    sim.quantizer = None
    sim.filters = json.loads(json.dumps(filters))
    sim._last_ref_price = 100.0
    sim.risk = _FakeRisk(0.05)

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=0.1)
    oid = sim.submit(proto)
    report = sim.pop_ready(ref_price=100.0)

    assert report.trades == []
    assert report.cancelled_ids == [oid]
    assert report.cancelled_reasons[oid] == "LOT_SIZE"
    assert report.status == "REJECTED_BY_FILTER"
    assert report.reason and report.reason.get("primary") == "FILTER_REJECTION"


def test_limit_near_minimum_passes_after_quantization():
    local_filters = {
        "TESTUSDT": {
            "PRICE_FILTER": {
                "minPrice": "10",
                "maxPrice": "100000",
                "tickSize": "0.5",
            },
            "LOT_SIZE": {
                "minQty": "0.1",
                "maxQty": "1000",
                "stepSize": "0.1",
            },
            "MIN_NOTIONAL": {"minNotional": "1"},
        }
    }

    sim = ExecutionSimulator(symbol="TESTUSDT", filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = False
    sim.filters = local_filters
    sim.quantizer = Quantizer(local_filters, strict=True)

    price, qty, rejection = sim._apply_filters_limit_legacy(
        "BUY", price=9.999, qty=0.099, ref_price=10.0
    )

    assert rejection is None
    assert price == pytest.approx(10.0)
    assert qty == pytest.approx(0.1)


def test_limit_clamp_notional_growth_rejected_by_qty_limits():
    price = 50.0
    local_filters = {
        "TESTUSDT": {
            "PRICE_FILTER": {"minPrice": "10", "maxPrice": "100000", "tickSize": "0.1"},
            "LOT_SIZE": {"minQty": "0.1", "maxQty": "5", "stepSize": "0.1"},
            "MIN_NOTIONAL": {"minNotional": "400"},
        }
    }
    forced_qty = float(local_filters["TESTUSDT"]["MIN_NOTIONAL"]["minNotional"]) / price

    sim = ExecutionSimulator(symbol="TESTUSDT", filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = False
    sim.filters = local_filters
    sim.quantizer = _ClampNotionalQuantizer(forced_qty)

    price_adj, qty_adj, rejection = sim._apply_filters_limit(
        "BUY", price=price, qty=1.0, ref_price=price
    )

    assert price_adj == pytest.approx(price)
    assert qty_adj == 0.0
    assert rejection is not None
    assert rejection.code == "LOT_SIZE"
    assert rejection.message == "Quantity above maximum"


def test_limit_tick_alignment_with_offset_min_price():
    local_filters = {
        "OFFSETPAIR": {
            "PRICE_FILTER": {
                "minPrice": "0.35",
                "maxPrice": "1000",
                "tickSize": "0.2",
            },
            "LOT_SIZE": {"minQty": "1", "maxQty": "1000", "stepSize": "1"},
            "MIN_NOTIONAL": {"minNotional": "0"},
        }
    }

    sim = ExecutionSimulator(symbol="OFFSETPAIR", filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = False
    sim.quantizer = None
    sim.filters = local_filters

    price_ok, qty_ok, rejection_ok = sim._apply_filters_limit_legacy(
        "BUY", price=0.35, qty=10.0, ref_price=0.4
    )

    assert rejection_ok is None
    assert price_ok == pytest.approx(0.35)
    assert qty_ok == pytest.approx(10.0)

    price_bad, qty_bad, rejection_bad = sim._apply_filters_limit_legacy(
        "BUY", price=0.45, qty=10.0, ref_price=0.4
    )

    assert rejection_bad is not None
    assert rejection_bad.code == "PRICE_FILTER"
    assert rejection_bad.message == "Price not aligned to tick"
    assert rejection_bad.constraint is not None
    assert rejection_bad.constraint.get("min_price") == pytest.approx(0.35)
    snapped = rejection_bad.constraint.get("snapped_price")
    assert snapped == pytest.approx(0.35) or snapped == pytest.approx(0.55)


def test_limit_large_tick_aligned_price_survives_price_filter():
    price_tick = 0.01
    large_price = 123_456_789.23
    local_filters = {
        "BIGNUM": {
            "PRICE_FILTER": {
                "minPrice": "0",
                "maxPrice": "1000000000",
                "tickSize": f"{price_tick}",
            },
            "LOT_SIZE": {"minQty": "0.01", "maxQty": "1000", "stepSize": "0.01"},
            "MIN_NOTIONAL": {"minNotional": "0"},
        }
    }

    sim = ExecutionSimulator(symbol="BIGNUM", filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = False
    sim.quantizer = None
    sim.filters = local_filters

    price_ok, qty_ok, rejection_ok = sim._apply_filters_limit_legacy(
        "BUY", price=large_price, qty=1.0, ref_price=large_price
    )

    assert rejection_ok is None
    assert price_ok == pytest.approx(large_price)
    assert qty_ok == pytest.approx(1.0)

    misaligned_price = large_price + price_tick * 0.4
    price_bad, qty_bad, rejection_bad = sim._apply_filters_limit_legacy(
        "BUY", price=misaligned_price, qty=1.0, ref_price=large_price
    )

    assert rejection_bad is not None
    assert rejection_bad.code == "PRICE_FILTER"
    assert price_bad == pytest.approx(0.0)
    assert qty_bad == pytest.approx(0.0)


def test_limit_price_max_enforced_with_sub_picosecond_tick():
    price_max = 1e-9
    price_tick = 2e-15
    local_filters = {
        "FEMTOPRICE": {
            "PRICE_FILTER": {
                "minPrice": "0",
                "maxPrice": f"{price_max}",
                "tickSize": f"{price_tick}",
            },
            "LOT_SIZE": {"minQty": "0", "maxQty": "10", "stepSize": "1"},
            "MIN_NOTIONAL": {"minNotional": "0"},
        }
    }

    sim = ExecutionSimulator(symbol="FEMTOPRICE", filters_path=None)
    sim.strict_filters = True
    sim.enforce_ppbs = False
    sim.quantizer = None
    sim.filters = local_filters

    price_adj, qty_adj, rejection = sim._apply_filters_limit(
        "BUY", price=price_max + 1e-13, qty=1.0, ref_price=price_max
    )

    assert price_adj == pytest.approx(0.0)
    assert qty_adj == pytest.approx(0.0)
    assert rejection is not None
    assert rejection.code == "PRICE_FILTER"
    assert rejection.message == "Price above maximum"


def test_attach_quantizer_sets_metadata(tmp_path: pathlib.Path):
    filters_path = tmp_path / "filters.json"
    filters_path.write_text(json.dumps({"filters": filters}), encoding="utf-8")

    sim = ExecutionSimulator(filters_path=None)
    impl = QuantizerImpl.from_dict(
        {
            "path": str(filters_path),
            "filters_path": str(filters_path),
            "strict_filters": True,
            "enforce_percent_price_by_side": True,
        }
    )

    sim.attach_quantizer(impl=impl)

    assert sim.quantizer is impl.quantizer
    assert getattr(sim, "quantizer_impl", None) is impl
    assert isinstance(sim.filters, dict) and sim.filters
    metadata = getattr(sim, "quantizer_metadata", {})
    assert isinstance(metadata, dict)
    assert metadata.get("symbol_count") == len(filters)
    assert sim.quantizer_filters_sha256


def test_ttl_two_steps_sim():
    sim = make_sim(strict=False)
    sim.set_market_snapshot(bid=100.0, ask=101.0)
    proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=0.2, abs_price=99.0, ttl_steps=2)
    oid = sim.submit(proto)
    rep1 = sim.pop_ready(ref_price=100.0)
    assert rep1.new_order_ids == []
    assert rep1.cancelled_ids == [oid]
    assert rep1.trades == []
    rep2 = sim.pop_ready(ref_price=100.0)
    assert rep2.cancelled_ids == []
    rep3 = sim.pop_ready(ref_price=100.0)
    assert rep3.cancelled_ids == []


def test_limit_maker_price_enqueues_without_trade():
    sim = ExecutionSimulator(filters_path=None)
    sim.set_market_snapshot(bid=100.0, ask=101.0)

    proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=0.1, abs_price=100.5)
    oid = sim.submit(proto)

    report = sim.pop_ready(ref_price=100.0)

    assert report.trades == []
    assert report.cancelled_ids == []
    assert report.new_order_ids == [oid]
    assert report.new_order_pos == [0]


def test_latency_sample_slightly_above_step_waits_full_delay():
    sim = ExecutionSimulator(filters_path=None)
    sim.step_ms = 100
    sim.set_market_snapshot(bid=100.0, ask=101.0)

    class _LatencyModel:
        def sample(self, ts=None):
            return {"total_ms": sim.step_ms + 1}

    sim.latency = _LatencyModel()

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=0.5)
    oid = sim.submit(proto, now_ts=0)

    assert len(sim._q._q) == 1
    assert sim._q._q[0].client_order_id == oid
    assert sim._q._q[0].remaining_lat == 2

    report_first = sim.pop_ready(now_ts=sim.step_ms, ref_price=100.5)
    assert report_first.trades == []
    assert len(sim._q._q) == 1
    assert sim._q._q[0].remaining_lat == 1

    report_second = sim.pop_ready(now_ts=2 * sim.step_ms, ref_price=100.5)
    assert report_second.trades == []
    assert len(sim._q._q) == 1
    assert sim._q._q[0].remaining_lat == 0

    report_third = sim.pop_ready(now_ts=3 * sim.step_ms, ref_price=100.5)
    assert [t.client_order_id for t in report_third.trades] == [oid]


# --- C++ LOB tests (using stub) ---

def test_unquantized_limit_rejected_lob():
    lob = CythonLOB()
    q = Quantizer(filters, strict=True)
    oid, _ = add_limit_with_filters(lob, True, 100.3, 0.25, q)
    assert oid == 0


def test_quantized_limit_crosses_lob():
    lob = CythonLOB()
    q = Quantizer(filters, strict=True)
    ask_ticks = int(round(101.0 * PRICE_SCALE))
    ask_id, _ = lob.add_limit_order(False, ask_ticks, 0.2, 0, True)
    bid_id, _ = add_limit_with_filters(lob, True, 101.0, 0.2, q)
    assert bid_id == ask_id
    assert not lob.contains_order(ask_id)


def test_ttl_two_steps_lob():
    lob = CythonLOB()
    bid_ticks = int(round(100.0 * PRICE_SCALE))
    oid, _ = lob.add_limit_order(True, bid_ticks, 1.0, 0, True)
    assert lob.set_order_ttl(oid, 2)
    assert lob.contains_order(oid)
    assert lob.decay_ttl_and_cancel() == []
    assert lob.contains_order(oid)
    assert lob.decay_ttl_and_cancel() == [oid]
    assert not lob.contains_order(oid)
