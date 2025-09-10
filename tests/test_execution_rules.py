import importlib.util
import pathlib
import sys

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
    sim = ExecutionSimulator()
    q = Quantizer(filters, strict=strict)
    sim.set_quantizer(q)
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

def test_unquantized_limit_rejected_sim():
    sim = make_sim(strict=False)
    proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=0.05, abs_price=100.3)
    oid = sim.submit(proto)
    report = sim.pop_ready(ref_price=100.0)
    assert report.cancelled_ids == [oid]
    assert report.trades == []


def test_quantized_order_fills_sim():
    sim = make_sim(strict=True)
    sim.set_market_snapshot(bid=99.0, ask=101.0)
    proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=0.2, abs_price=101.0)
    oid = sim.submit(proto)
    report = sim.pop_ready(ref_price=100.0)
    assert report.cancelled_ids == []
    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.client_order_id == oid and trade.price == 101.0 and trade.qty == 0.2


def test_ttl_two_steps_sim():
    sim = make_sim(strict=True)
    sim.set_market_snapshot(bid=100.0, ask=101.0)
    proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=0.2, abs_price=99.0, ttl_steps=2)
    oid = sim.submit(proto)
    rep1 = sim.pop_ready(ref_price=100.0)
    assert rep1.new_order_ids == [oid]
    rep2 = sim.pop_ready(ref_price=100.0)
    assert rep2.cancelled_ids == []
    rep3 = sim.pop_ready(ref_price=100.0)
    assert rep3.cancelled_ids == [oid]
    assert rep3.cancelled_reasons == {oid: "TTL"}


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
