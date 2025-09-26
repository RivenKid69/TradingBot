import builtins
import importlib
import json
import sys


def test_execution_sim_numpy_shim(monkeypatch, tmp_path):
    original_numpy = sys.modules.pop("numpy", None)

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy" and globals and globals.get("__name__") == "execution_sim":
            raise ImportError("numpy is intentionally unavailable for this test")
        return original_import(name, globals, locals, fromlist, level)

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fake_import)

    sys.modules.pop("execution_sim", None)

    try:
        execution_sim = importlib.import_module("execution_sim")
        data = {
            "liquidity": [1.0] * 168,
            "spread": [1.0] * 168,
        }
        path = tmp_path / "seasonality.json"
        path.write_text(json.dumps(data))

        sim = execution_sim.ExecutionSimulator(
            liquidity_seasonality_path=str(path),
            use_seasonality=True,
            seasonality_auto_reload=False,
        )

        sim.load_seasonality_multipliers(
            {
                "liquidity": [1.0] * 168,
                "spread": [1.0] * 168,
            }
        )

        dumped = sim.dump_seasonality_multipliers()
        assert dumped["liquidity"] == [1.0] * 168
        assert dumped["spread"] == [1.0] * 168
    finally:
        sys.modules.pop("execution_sim", None)
        sys.modules.pop("numpy", None)
        if original_numpy is not None:
            sys.modules["numpy"] = original_numpy

