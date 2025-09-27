import importlib.util
import pathlib
import sys


BASE_DIR = pathlib.Path(__file__).resolve().parents[1]


def _load_execution_simulator():
    existing = sys.modules.get("execution_sim")
    if existing is not None:
        return existing.ExecutionSimulator

    spec = importlib.util.spec_from_file_location(
        "execution_sim", BASE_DIR / "execution_sim.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["execution_sim"] = module
    spec.loader.exec_module(module)
    return module.ExecutionSimulator


ExecutionSimulator = _load_execution_simulator()


def test_apply_trade_inventory_resets_position_with_tolerance():
    sim = ExecutionSimulator(filters_path=None)

    sim._apply_trade_inventory("BUY", price=100.0, qty=0.1)
    sim._apply_trade_inventory("BUY", price=110.0, qty=0.2)
    sim._apply_trade_inventory("SELL", price=105.0, qty=0.3)

    assert sim.position_qty == 0.0
    assert sim._avg_entry_price is None
