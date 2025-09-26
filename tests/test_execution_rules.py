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
        "PERCENT_PRICE_BY_SIDE": {"multiplierUp": "1000", "multiplierDown": "0"},
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


def add_limit_with_filters(lob: CythonLOB, is_buy: bool, price: float, qty: float, q: Quantizer):
    p_abs = q.quantize_price("BTCUSDT", price)
    q_qty = q.quantize_qty("BTCUSDT", qty)
    q_qty = q.clamp_notional("BTCUSDT", p_abs if p_abs > 0 else price, q_qty)
    p_ticks = int(round(p_abs * PRICE_SCALE))
    if p_ticks != int(round(price * PRICE_SCALE)) or abs(q_qty - qty) > 1e-12:
        return 0, 0
    return lob.add_limit_order(is_buy, p_ticks, q_qty, 0, True)


# --- Python ExecutionSimulator tests ---

def test_unquantized_limit_executes_permissive():
    sim = make_sim(strict=False)
    proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=0.05, abs_price=100.3)
    oid = sim.submit(proto)
    report = sim.pop_ready(ref_price=100.0)

    assert report.cancelled_ids == []
    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.client_order_id == oid
    assert trade.price == pytest.approx(100.0)
    assert sim.strict_filters is False


def test_unquantized_limit_rejected_strict():
    sim = make_sim(strict=True)
    assert sim.strict_filters is True
    proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=0.02, abs_price=101.0)
    oid = sim.submit(proto)
    report = sim.pop_ready(ref_price=100.0)

    assert report.cancelled_ids == []
    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.client_order_id == oid
    assert trade.price == pytest.approx(100.0)


def test_market_quantity_rounded_up_passes_filters():
    sim = ExecutionSimulator(filters_path=None)
    sim.set_quantizer(Quantizer(filters, strict=True))

    qty_total, rejection = sim._apply_filters_market("BUY", 0.099, ref_price=100.0)

    assert qty_total == pytest.approx(0.1)
    assert rejection is None


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
