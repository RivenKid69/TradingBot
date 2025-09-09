import math
import importlib.util
import pathlib

base = pathlib.Path(__file__).resolve().parents[1]
spec_fees = importlib.util.spec_from_file_location("fees", base / "fees.py")
fees_mod = importlib.util.module_from_spec(spec_fees)
import sys
sys.modules["fees"] = fees_mod
spec_fees.loader.exec_module(fees_mod)
FeesModel = fees_mod.FeesModel

spec_exec = importlib.util.spec_from_file_location("execution_sim", base / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec_exec)
sys.modules["execution_sim"] = exec_mod
spec_exec.loader.exec_module(exec_mod)
ExecTrade = exec_mod.ExecTrade

def test_limit_order_maker_fee():
    fees = FeesModel.from_dict({"maker_bps": 1.0, "taker_bps": 5.0})
    price = 101.0
    qty = 1.0
    fee = fees.compute(side="SELL", price=price, qty=qty, liquidity="maker")
    trade = ExecTrade(ts=0, side="SELL", price=price, qty=qty,
                      notional=price * qty, liquidity="maker",
                      proto_type=2, client_order_id=1, fee=fee)
    expected = price * qty * (1.0 / 1e4)
    assert math.isclose(trade.fee, expected, rel_tol=1e-9)
