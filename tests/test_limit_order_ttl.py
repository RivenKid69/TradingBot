import importlib.util
import pathlib
import sys

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
    assert report2.trades == []
