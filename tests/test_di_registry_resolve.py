from types import SimpleNamespace

from core_contracts import PolicyCtx, SignalPolicy
from di_registry import build_component, resolve
from strategies.base import BaseSignalPolicy


class DummyPolicy(BaseSignalPolicy):
    def decide(self, features, ctx: PolicyCtx):
        return []


class PolicyAdapter(BaseSignalPolicy):
    def __init__(self, policy: SignalPolicy) -> None:
        super().__init__()
        self._policy = policy

    def decide(self, features, ctx: PolicyCtx):
        return list(self._policy.decide(features, ctx) or [])


def _spec(target: str, params: dict | None = None):
    return SimpleNamespace(target=target, params=params or {})


def test_resolve_direct_policy():
    container = {}
    spec = _spec("tests.test_di_registry_resolve:DummyPolicy")
    build_component("policy", spec, container)
    assert resolve(SignalPolicy, container) is container["policy"]


def test_resolve_policy_adapter():
    container: dict = {}
    base_spec = _spec("tests.test_di_registry_resolve:DummyPolicy")
    build_component("base_policy", base_spec, container)
    adapter_spec = _spec("tests.test_di_registry_resolve:PolicyAdapter")
    build_component("policy", adapter_spec, container)
    assert resolve(SignalPolicy, container) is container["policy"]
