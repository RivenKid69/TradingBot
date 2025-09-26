import importlib.util
import pathlib
import sys

base = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(base))
spec = importlib.util.spec_from_file_location("execution_sim", base / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec)
sys.modules["execution_sim"] = exec_mod
spec.loader.exec_module(exec_mod)

ActionProto = exec_mod.ActionProto
ActionType = exec_mod.ActionType
ExecutionSimulator = exec_mod.ExecutionSimulator


def test_zero_timestamp_preserved_on_submit_and_execution():
    sim = ExecutionSimulator(filters_path=None, latency_steps=0)
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)

    sim.submit(proto, now_ts=0)

    # The order should remain queued with the provided timestamp.
    pending_entries = getattr(sim._q, "_q", [])
    assert pending_entries, "expected pending order in latency queue"
    assert pending_entries[0].timestamp == 0

    report = sim.pop_ready(now_ts=0, ref_price=10.0)
    assert report.trades, "expected trade execution"
    assert all(trade.ts == 0 for trade in report.trades)
