import json
import importlib.util
import pathlib
import sys
import types

import pytest

base = pathlib.Path(__file__).resolve().parents[1]
if str(base) not in sys.path:
    sys.path.append(str(base))

spec_med = importlib.util.spec_from_file_location("mediator", base / "mediator.py")
med_mod = importlib.util.module_from_spec(spec_med)
sys.modules["mediator"] = med_mod
spec_med.loader.exec_module(med_mod)
Mediator = med_mod.Mediator

spec_ap = importlib.util.spec_from_file_location("action_proto", base / "action_proto.py")
ap_mod = importlib.util.module_from_spec(spec_ap)
sys.modules["action_proto"] = ap_mod
spec_ap.loader.exec_module(ap_mod)
ActionProto = ap_mod.ActionProto
ActionType = ap_mod.ActionType

spec_quant = importlib.util.spec_from_file_location("quantizer", base / "quantizer.py")
quant_mod = importlib.util.module_from_spec(spec_quant)
sys.modules["quantizer"] = quant_mod
spec_quant.loader.exec_module(quant_mod)

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


class DummyLOB:
    def __init__(self):
        self.next_id = 1

    def add_limit_order(
        self,
        is_buy_side,
        price_ticks,
        volume,
        timestamp,
        taker_is_agent=True,
    ):
        oid = self.next_id
        self.next_id += 1
        return oid, 0

    def remove_order(self, is_buy_side, price_ticks, order_id):
        return True

    def match_market_order(
        self,
        is_buy_side,
        volume,
        timestamp,
        taker_is_agent,
        out_prices=None,
        out_volumes=None,
        out_is_buy=None,
        out_is_self=None,
        out_ids=None,
        max_len: int = 0,
    ):
        return 0, 0.0


class DummyState:
    def __init__(self):
        self.units = 0.0
        self.cash = 0.0
        self.max_position = 10.0


class DummyEnv:
    def __init__(
        self,
        lob,
        symbol: str = "BTCUSDT",
        quantizer_path: pathlib.Path | None = None,
    ):
        self.state = DummyState()
        self.lob = lob
        self.symbol = symbol
        self.last_mid = 100.0
        self.last_mtm_price = 100.0
        quantizer_cfg: dict[str, object] = {}
        if quantizer_path is not None:
            path_str = str(quantizer_path)
            quantizer_cfg = {
                "path": path_str,
                "filters_path": path_str,
                "strict_filters": True,
                "enforce_percent_price_by_side": True,
            }
        self.run_config = types.SimpleNamespace(quantizer=quantizer_cfg)


filters = {
    "BTCUSDT": {
        "PRICE_FILTER": {"minPrice": "0", "maxPrice": "1000000", "tickSize": "0.5"},
        "LOT_SIZE": {"minQty": "0.1", "maxQty": "1000", "stepSize": "0.1"},
        "MIN_NOTIONAL": {"minNotional": "5"},
        "PERCENT_PRICE_BY_SIDE": {"multiplierUp": "1000", "multiplierDown": "0"},
    }
}


@pytest.fixture
def filters_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "filters.json"
    path.write_text(json.dumps({"filters": filters}), encoding="utf-8")
    return path


def make_mediator(filters_path: pathlib.Path, *, use_exec: bool = False):
    env = DummyEnv(DummyLOB(), quantizer_path=filters_path)
    med = Mediator(env, use_exec_sim=use_exec)
    assert med.quantizer is not None
    assert med.quantizer_impl is not None
    return med


def test_unquantized_order_rejected(filters_file: pathlib.Path):
    med = make_mediator(filters_file)
    price = int(100.3 * PRICE_SCALE)
    oid, qpos = med.add_limit_order(
        is_buy_side=True, price_ticks=price, volume=0.25, timestamp=0
    )
    assert (oid, qpos) == (0, 0)


def test_quantized_order_accepted(filters_file: pathlib.Path):
    med = make_mediator(filters_file)
    price = int(100.5 * PRICE_SCALE)
    oid, qpos = med.add_limit_order(
        is_buy_side=True, price_ticks=price, volume=0.2, timestamp=0
    )
    assert oid != 0


def test_exec_simulator_receives_quantizer(filters_file: pathlib.Path):
    med = make_mediator(filters_file, use_exec=True)
    assert med.exec is not None
    assert getattr(med.exec, "quantizer", None) is med.quantizer
    assert getattr(med.exec, "quantizer_impl", None) is med.quantizer_impl
