from decimal import Decimal
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core_models import Bar
from core_contracts import FeaturePipe, SignalPolicy
from pipeline import policy_decide, apply_risk


class DummyFP(FeaturePipe):
    def warmup(self):
        pass

    def update(self, bar):
        return {"x": 1}


class DummyPolicy(SignalPolicy):
    def decide(self, feats, ctx):
        return ["o1", "o2"]


class DummyGuards:
    def apply(self, ts_ms, symbol, decisions):
        return decisions[:1], None


def test_decision_flow():
    bar = Bar(ts=0, symbol="BTC", open=Decimal("0"), high=Decimal("0"), low=Decimal("0"), close=Decimal("0"))
    fp = DummyFP()
    policy = DummyPolicy()
    pol_res = policy_decide(fp, policy, bar)
    assert pol_res.decision == ["o1", "o2"]
    guards = DummyGuards()
    risk_res = apply_risk(0, "BTC", guards, pol_res.decision)
    assert risk_res.decision == ["o1"]
