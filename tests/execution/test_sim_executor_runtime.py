from types import SimpleNamespace

from core_config import ExecutionEntryMode, ExecutionProfile
from impl_sim_executor import SimExecutor


class DummyClipConfig:
    def __init__(self, enabled=None, strict_open_fill=None):
        self.enabled = enabled
        self.strict_open_fill = strict_open_fill


class DummyExecutionConfig:
    def __init__(self, entry_mode=None, clip_to_bar=None):
        self.entry_mode = entry_mode
        self.clip_to_bar = clip_to_bar


def test_resolve_runtime_settings_alias_next_bar_open_defaults():
    cfg = {"entry_mode": "next_bar_open"}

    entry_mode, profile, clip_enabled, strict_fill = SimExecutor.resolve_execution_runtime_settings(cfg)

    assert entry_mode is ExecutionEntryMode.DEFAULT
    assert profile is ExecutionProfile.MKT_OPEN_NEXT_H1
    assert clip_enabled is True
    assert strict_fill is False


def test_resolve_runtime_settings_alias_limit_from_object_and_clip_namespace():
    clip_cfg = SimpleNamespace(enabled="1", strict_open_fill="no")
    cfg = DummyExecutionConfig(entry_mode="limit", clip_to_bar=clip_cfg)

    entry_mode, profile, clip_enabled, strict_fill = SimExecutor.resolve_execution_runtime_settings(
        cfg, default_profile=ExecutionProfile.MKT_OPEN_NEXT_H1
    )

    assert entry_mode is ExecutionEntryMode.STRICT
    assert profile is ExecutionProfile.LIMIT_MID_BPS
    assert clip_enabled is True
    assert strict_fill is False


def test_resolve_runtime_settings_explicit_profile_and_clip_mapping():
    clip_cfg = {"enabled": False, "strict_open_fill": True}
    cfg = {"entry_mode": ExecutionProfile.VWAP_CURRENT_H1, "clip_to_bar": clip_cfg}

    entry_mode, profile, clip_enabled, strict_fill = SimExecutor.resolve_execution_runtime_settings(
        cfg, default_profile=ExecutionProfile.MKT_OPEN_NEXT_H1
    )

    assert entry_mode is ExecutionEntryMode.DEFAULT
    assert profile is ExecutionProfile.VWAP_CURRENT_H1
    assert clip_enabled is False
    assert strict_fill is True


def test_resolve_runtime_settings_clip_config_object_variants():
    clip_cfg = DummyClipConfig(enabled=0, strict_open_fill=1)
    cfg = {"clip_to_bar": clip_cfg}

    entry_mode, profile, clip_enabled, strict_fill = SimExecutor.resolve_execution_runtime_settings(cfg)

    assert entry_mode is ExecutionEntryMode.DEFAULT
    assert profile is ExecutionProfile.MKT_OPEN_NEXT_H1
    assert clip_enabled is False
    assert strict_fill is True


def test_configure_simulator_execution_writes_expected_simulator_state():
    class SimulatorDouble:
        pass

    sim = SimulatorDouble()
    cfg = {"entry_mode": "limit", "clip_to_bar": {"enabled": False, "strict_open_fill": True}}

    entry_mode, profile, clip_enabled, strict_fill = SimExecutor.configure_simulator_execution(sim, cfg)

    assert entry_mode is ExecutionEntryMode.STRICT
    assert profile is ExecutionProfile.LIMIT_MID_BPS
    assert clip_enabled is False
    assert strict_fill is True

    assert getattr(sim, "execution_entry_mode") == ExecutionEntryMode.STRICT.value
    assert getattr(sim, "_clip_to_bar_enabled") is False
    assert getattr(sim, "_clip_to_bar_strict_open_fill") is True


def test_resolve_runtime_settings_with_malformed_clip_config_uses_defaults():
    cfg = {"clip_to_bar": {"enabled": "maybe", "strict_open_fill": "perhaps"}}

    entry_mode, profile, clip_enabled, strict_fill = SimExecutor.resolve_execution_runtime_settings(cfg)

    assert entry_mode is ExecutionEntryMode.DEFAULT
    assert profile is ExecutionProfile.MKT_OPEN_NEXT_H1
    assert clip_enabled is True
    assert strict_fill is False
