import importlib
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest


@contextmanager
def block_import(name: str):
    class _Blocker(importlib.abc.MetaPathFinder):  # type: ignore[attr-defined]
        def find_spec(self, fullname, path, target=None):  # pragma: no cover - hook
            if fullname == name or fullname.startswith(f"{name}."):
                raise ImportError(f"Blocked import: {fullname}")
            return None

    blocker = _Blocker()
    sys.modules.pop(name, None)
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


@contextmanager
def temp_module(name: str, module: ModuleType):
    original = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture(autouse=True)
def clear_service_modules():
    to_clear = [
        "service_signal_runner",
        "service_backtest",
        "sandbox.sim_adapter",
        "impl_sim_executor",
    ]
    for mod in to_clear:
        sys.modules.pop(mod, None)
    yield
    for mod in to_clear:
        sys.modules.pop(mod, None)


def test_service_signal_runner_bar_mode_no_execution_sim(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with block_import("execution_sim"):
        module = importlib.import_module("service_signal_runner")

        class DummyAdapter:
            sim = None
            source = None
            market_data = None
            ws = None

        class DummyFeaturePipe:
            def process(self, *args, **kwargs):  # pragma: no cover - smoke helper
                return []

        class DummyPolicy:
            def __call__(self, *args, **kwargs):  # pragma: no cover - smoke helper
                return []

        cfg = module.SignalRunnerConfig(logs_dir=str(tmp_path / "logs"), run_id="test")
        run_cfg = SimpleNamespace(
            execution=SimpleNamespace(mode="bar"),
            slippage_regime_updates=False,
        )
        runner = module.ServiceSignalRunner(
            DummyAdapter(),
            DummyFeaturePipe(),
            DummyPolicy(),
            cfg=cfg,
            run_config=run_cfg,
        )

        assert runner._execution_mode == "bar"
        assert "execution_sim" not in sys.modules


def test_service_signal_runner_order_mode_instantiation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    module = importlib.import_module("service_signal_runner")

    class DummyAdapter:
        sim = None
        source = None
        market_data = None
        ws = None

    cfg = module.SignalRunnerConfig(logs_dir=str(tmp_path / "logs"), run_id="test-order")
    run_cfg = SimpleNamespace(execution=SimpleNamespace(mode="order"), slippage_regime_updates=False)

    runner = module.ServiceSignalRunner(
        DummyAdapter(),
        object(),
        lambda *args, **kwargs: [],
        cfg=cfg,
        run_config=run_cfg,
    )

    assert runner._execution_mode == "order"


def test_service_signal_runner_execution_mode_normalization(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    module = importlib.import_module("service_signal_runner")

    class DummyAdapter:
        sim = None
        source = None
        market_data = None
        ws = None

    cfg = module.SignalRunnerConfig(logs_dir=str(tmp_path / "logs"), run_id="test-normalized")
    run_cfg = SimpleNamespace(execution=SimpleNamespace(mode=" bar "), slippage_regime_updates=False)

    runner = module.ServiceSignalRunner(
        DummyAdapter(),
        object(),
        lambda *args, **kwargs: [],
        cfg=cfg,
        run_config=run_cfg,
    )

    assert runner._execution_mode == "bar"


def test_service_signal_runner_bar_mode_inline_execution(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    module = importlib.import_module("service_signal_runner")
    monkeypatch.setattr(module.signal_bus, "ENABLED", False)

    class DummyAdapter:
        sim = None
        source = None
        market_data = None
        ws = None

    class DummyFeaturePipe:
        def warmup(self):  # pragma: no cover - smoke helper
            return None

        def process(self, *args, **kwargs):  # pragma: no cover - smoke helper
            return []

    class DummyPolicy:
        def __call__(self, *args, **kwargs):  # pragma: no cover - smoke helper
            return []

    class InlineExecutorStub:
        def __init__(self):
            self.calls = 0
            self._snapshot = {"execution_mode": "bar"}

        def execute(self, order):
            self.calls += 1
            meta = getattr(order, "meta", None)
            if not isinstance(meta, dict):
                meta = dict(meta or {})
                setattr(order, "meta", meta)
            execution_meta = {
                "filled": True,
                "turnover_usd": 42.0,
                "target_weight": 0.5,
                "delta_weight": 0.5,
            }
            meta["_bar_execution"] = dict(execution_meta)
            self._snapshot = {
                "execution_mode": "bar",
                "bar_execution": {
                    "decisions": 1,
                    "act_now": 1,
                    "turnover_usd": 42.0,
                    "bar_ts": 1,
                },
            }
            return SimpleNamespace(meta={"execution": dict(execution_meta)})

        def monitoring_snapshot(self):
            return self._snapshot

    cfg = module.SignalRunnerConfig(logs_dir=str(tmp_path / "logs"), run_id="test-inline")
    run_cfg = SimpleNamespace(execution=SimpleNamespace(mode="bar"), slippage_regime_updates=False)

    feature_pipe = DummyFeaturePipe()
    policy = DummyPolicy()
    runner = module.ServiceSignalRunner(
        DummyAdapter(),
        feature_pipe,
        policy,
        cfg=cfg,
        run_config=run_cfg,
    )

    executor_stub = InlineExecutorStub()
    worker = module._Worker(
        feature_pipe,
        policy,
        runner.logger,
        executor_stub,
        runner.risk_guards,
        lambda: False,
        enforce_closed_bars=runner.enforce_closed_bars,
        close_lag_ms=runner.close_lag_ms,
        ws_dedup_enabled=runner.ws_dedup_enabled,
        ws_dedup_log_skips=runner.ws_dedup_log_skips,
        ws_dedup_timeframe_ms=runner.ws_dedup_timeframe_ms,
        bar_timeframe_ms=None,
        throttle_cfg=runner.throttle_cfg,
        no_trade_cfg=runner.no_trade_cfg,
        pipeline_cfg=runner.pipeline_cfg,
        signal_quality_cfg=runner.signal_quality_cfg,
        zero_signal_alert=getattr(runner.monitoring_cfg.thresholds, "zero_signals", 0),
        state_enabled=runner.cfg.state.enabled,
        rest_candidates=None,
        monitoring=None,
        monitoring_agg=None,
        worker_id="test-worker",
        status_callback=None,
        execution_mode=runner._execution_mode,
        portfolio_equity=None,
        max_total_weight=None,
        idempotency_cache_size=1024,
        cooldown_settings=None,
        signal_dispatcher=None,
    )

    bar_open_ms = 1_700_000_000_000
    order = SimpleNamespace(
        symbol="BTCUSDT",
        meta={"payload": {"target_weight": 0.5}, "signal_leg": "entry"},
        created_ts_ms=bar_open_ms,
    )

    result = worker.publish_decision(
        order,
        "BTCUSDT",
        bar_open_ms,
        bar_close_ms=bar_open_ms + 60_000,
    )

    assert result.action == "pass"
    assert executor_stub.calls == 1
    exec_meta = order.meta.get("_bar_execution")
    assert exec_meta is not None
    assert exec_meta.get("target_weight") == pytest.approx(0.5)
    assert worker._weights.get("BTCUSDT") == pytest.approx(0.5)


def _make_execution_sim_stub() -> ModuleType:
    mod = ModuleType("execution_sim")

    class ExecutionSimulator:
        def __init__(self, *_, **__):
            self._adv_store = None
            self.run_config = None

        def set_adv_store(self, *args, **kwargs):  # pragma: no cover - smoke helper
            self._adv_store = kwargs.get("store") if "store" in kwargs else args[0] if args else None

        def has_adv_store(self):  # pragma: no cover - smoke helper
            return self._adv_store is not None

        def set_bar_capacity_base_config(self, **kwargs):  # pragma: no cover - smoke helper
            self._bar_capacity_base = kwargs

    mod.ExecutionSimulator = ExecutionSimulator
    return mod


def _make_sim_executor_stub() -> ModuleType:
    mod = ModuleType("impl_sim_executor")

    class SimExecutor:
        @staticmethod
        def configure_simulator_execution(sim, cfg, default_profile):  # pragma: no cover - smoke helper
            return ("entry", default_profile, False, False)

        @staticmethod
        def apply_execution_profile(sim, profile, params):  # pragma: no cover - smoke helper
            sim._profile = profile

        @staticmethod
        def _bool_or_none(value):  # pragma: no cover - smoke helper
            if value is None:
                return None
            return bool(value)

    mod.SimExecutor = SimExecutor
    return mod


def _make_sim_adapter_stub() -> ModuleType:
    mod = ModuleType("sandbox.sim_adapter")

    class SimAdapter:
        def __init__(self, sim, **kwargs):  # pragma: no cover - smoke helper
            self.sim = sim

    mod.SimAdapter = SimAdapter
    return mod


def _make_exchange_stubs():
    package = ModuleType("exchange")
    specs = ModuleType("exchange.specs")

    def load_specs(*args, **kwargs):  # pragma: no cover - smoke helper
        return {}, {}

    def round_price_to_tick(price, tick, *_args, **_kwargs):  # pragma: no cover - smoke helper
        return price

    specs.load_specs = load_specs
    specs.round_price_to_tick = round_price_to_tick
    package.specs = specs
    return package, specs


def test_service_backtest_bar_mode_no_execution_sim(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    exchange_pkg_stub, exchange_specs_stub = _make_exchange_stubs()

    with block_import("execution_sim"), temp_module("exchange", exchange_pkg_stub), temp_module(
        "exchange.specs", exchange_specs_stub
    ):
        module = importlib.import_module("service_backtest")
        from impl_bar_executor import BarExecutor

        executor = BarExecutor(run_id="bar", default_equity_usd=1000.0)
        bridge = module.BarBacktestSimBridge(
            executor,
            symbol="BTCUSDT",
            timeframe_ms=60_000,
            initial_equity=1000.0,
        )

        assert bridge.executor is executor
        assert "execution_sim" not in sys.modules


def test_service_backtest_order_mode_instantiation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    exec_stub = _make_execution_sim_stub()
    sim_executor_stub = _make_sim_executor_stub()
    sim_adapter_stub = _make_sim_adapter_stub()
    exchange_pkg_stub, exchange_specs_stub = _make_exchange_stubs()

    with temp_module("execution_sim", exec_stub), temp_module("impl_sim_executor", sim_executor_stub), temp_module(
        "sandbox.sim_adapter", sim_adapter_stub
    ), temp_module("exchange", exchange_pkg_stub), temp_module(
        "exchange.specs", exchange_specs_stub
    ):
        module = importlib.import_module("service_backtest")
        sim = exec_stub.ExecutionSimulator()
        cfg = module.BacktestConfig(symbol="BTCUSDT", timeframe="1m")
        policy = object()
        run_cfg = SimpleNamespace(execution=SimpleNamespace(mode="order"))

        service = module.ServiceBacktest(policy, sim, cfg, run_config=run_cfg)

        assert isinstance(service.sim, exec_stub.ExecutionSimulator)
        assert "execution_sim" in sys.modules

        assert hasattr(service, "sim_bridge")
        assert service.sim_bridge.sim is sim

