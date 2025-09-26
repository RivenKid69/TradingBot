import importlib.util
import pathlib
import sys
from types import SimpleNamespace

base = pathlib.Path(__file__).resolve().parents[1]
spec_exec = importlib.util.spec_from_file_location("execution_sim", base / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec_exec)
sys.modules["execution_sim"] = exec_mod
spec_exec.loader.exec_module(exec_mod)

ActionProto = exec_mod.ActionProto
ActionType = exec_mod.ActionType
ExecutionSimulator = exec_mod.ExecutionSimulator

def test_limit_order_ttl_expires():
    sim = ExecutionSimulator()
    proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=1.0, abs_price=100.0, ttl_steps=1)
    oid = sim.submit(proto)
    report1 = sim.pop_ready(ref_price=100.0)
    assert report1.new_order_ids == [oid]
    report2 = sim.pop_ready(ref_price=100.0)
    assert report2.cancelled_ids == [oid]
    assert report2.cancelled_reasons == {oid: "TTL"}
    assert report2.trades == []


def test_limit_order_ttl_survives():
    sim = ExecutionSimulator()
    proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=1.0, abs_price=100.0, ttl_steps=3)
    oid = sim.submit(proto)
    report1 = sim.pop_ready(ref_price=100.0)
    assert report1.new_order_ids == [oid]
    report2 = sim.pop_ready(ref_price=100.0)
    assert report2.cancelled_ids == []
    report3 = sim.pop_ready(ref_price=100.0)
    assert report3.cancelled_ids == []
    report4 = sim.pop_ready(ref_price=100.0)
    assert report4.cancelled_ids == [oid]
    assert report4.cancelled_reasons == {oid: "TTL"}


def test_limit_order_ttl_ms_rounds_up():
    sim = ExecutionSimulator()
    sim.step_ms = 200
    proto = SimpleNamespace(
        action_type=ActionType.LIMIT,
        volume_frac=1.0,
        abs_price=100.0,
        ttl_ms=350,
    )
    oid = sim.submit(proto)
    report1 = sim.pop_ready(ref_price=100.0)
    assert report1.new_order_ids == [oid]
    report2 = sim.pop_ready(ref_price=100.0)
    assert report2.cancelled_ids == []
    report3 = sim.pop_ready(ref_price=100.0)
    assert report3.cancelled_ids == [oid]
    assert report3.cancelled_reasons == {oid: "TTL"}


from fast_lob import CythonLOB

def test_cpp_lob_ttl_expires():
    lob = CythonLOB()
    oid, _ = lob.add_limit_order(True, 1000, 1.0, 0, True)
    assert lob.set_order_ttl(oid, 2)
    assert lob.contains_order(oid)
    assert lob.decay_ttl_and_cancel() == []
    assert lob.contains_order(oid)
    assert lob.decay_ttl_and_cancel() == [oid]
    assert not lob.contains_order(oid)


def test_limit_order_ioc_partial_cancel():
    sim = ExecutionSimulator()
    sim.set_market_snapshot(bid=99.0, ask=101.0, liquidity=5.0)
    proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=10.0, abs_price=101.0, tif="IOC")
    oid = sim.submit(proto)
    report = sim.pop_ready(ref_price=100.0)
    assert [t.qty for t in report.trades] == [5.0]
    assert report.cancelled_ids == [oid]
    assert report.cancelled_reasons == {oid: "IOC"}
    assert report.new_order_ids == []


def test_limit_order_fok_partial_cancel():
    sim = ExecutionSimulator()
    sim.set_market_snapshot(bid=99.0, ask=101.0, liquidity=5.0)
    proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=10.0, abs_price=101.0, tif="FOK")
    oid = sim.submit(proto)
    report = sim.pop_ready(ref_price=100.0)
    assert report.trades == []
    assert report.cancelled_ids == [oid]
    assert report.cancelled_reasons == {oid: "FOK"}
    assert report.new_order_ids == []
