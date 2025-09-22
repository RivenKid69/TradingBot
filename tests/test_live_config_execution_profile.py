"""Ensure live configuration exposes execution profile details."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core_config import ExecutionParams, ExecutionProfile, load_config
from impl_sim_executor import SimExecutor


class _DummyExecutionSimulator:
    def __init__(self) -> None:
        self.latency_config_payload = {}
        self.close_lag_ms = 0
        self.received_profile: str | None = None
        self.received_params: dict[str, object] | None = None

    def attach_quantizer(self, impl, metadata=None):  # noqa: D401 - simple stub
        self.quantizer_attachment = (impl, metadata)

    def register_market_regime_listener(self, callback):
        self.market_regime_listener = callback

    def set_market_regime_hint(self, regime):  # noqa: D401 - simple stub
        self.market_regime_hint = regime

    def set_execution_profile(self, profile, params):
        self.received_profile = profile
        self.received_params = params


class _DummyQuantizer:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(
            strict=True,
            enforce_percent_price_by_side=True,
        )
        self.filters_metadata = {}
        self.attach_calls: list[tuple[object, dict[str, object]]] = []

    def attach_to(self, sim, **kwargs):  # noqa: D401 - simple stub
        self.attach_calls.append((sim, kwargs))


class _DummyAttachable:
    def __init__(self) -> None:
        self.attach_calls: list[tuple[object, dict[str, object]]] = []

    def attach_to(self, sim, **kwargs):  # noqa: D401 - simple stub
        self.attach_calls.append((sim, kwargs))


class _DummySlippage(_DummyAttachable):
    def __init__(self) -> None:
        super().__init__()
        self.dynamic_profile = None
        self.last_regime = None

    def set_market_regime(self, regime):  # noqa: D401 - simple stub
        self.last_regime = regime


def test_live_config_execution_profile_reaches_executor():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "config_live.yaml"
    cfg = load_config(str(cfg_path))

    assert cfg.execution_profile == ExecutionProfile.MKT_OPEN_NEXT_H1
    assert isinstance(cfg.execution_params, ExecutionParams)

    expected_params = cfg.execution_params.model_dump(exclude_unset=False)
    assert expected_params == {
        "slippage_bps": 0.0,
        "limit_offset_bps": 0.0,
        "ttl_steps": 0,
        "tif": "GTC",
    }

    sim = _DummyExecutionSimulator()
    quantizer = _DummyQuantizer()
    risk = _DummyAttachable()
    latency = _DummyAttachable()
    slippage = _DummySlippage()
    fees = _DummyAttachable()

    executor = SimExecutor(
        sim,
        symbol="BTCUSDT",
        quantizer=quantizer,
        risk=risk,
        latency=latency,
        slippage=slippage,
        fees=fees,
        data_degradation=SimpleNamespace(),
        run_config=cfg,
    )

    expected_profile_text = str(ExecutionProfile.MKT_OPEN_NEXT_H1)

    assert sim.received_profile == expected_profile_text
    assert sim.received_params == expected_params
    assert sim.execution_profile == expected_profile_text
    assert sim.execution_params == expected_params
    assert executor._exec_profile == ExecutionProfile.MKT_OPEN_NEXT_H1
    assert executor._exec_params.model_dump(exclude_unset=False) == expected_params
