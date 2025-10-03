import sys
import types

import pytest


def _install_sb3_stub():
    if "sb3_contrib" in sys.modules:
        return

    sb3_contrib = types.ModuleType("sb3_contrib")
    sb3_contrib.__path__ = []  # mark as package
    sys.modules["sb3_contrib"] = sb3_contrib

    common = types.ModuleType("sb3_contrib.common")
    common.__path__ = []
    sys.modules["sb3_contrib.common"] = common
    sb3_contrib.common = common  # type: ignore[attr-defined]

    recurrent = types.ModuleType("sb3_contrib.common.recurrent")
    recurrent.__path__ = []
    sys.modules["sb3_contrib.common.recurrent"] = recurrent
    common.recurrent = recurrent  # type: ignore[attr-defined]

    policies = types.ModuleType("sb3_contrib.common.recurrent.policies")
    class _DummyPolicy:  # pragma: no cover - simple placeholder
        pass

    policies.RecurrentActorCriticPolicy = _DummyPolicy
    sys.modules["sb3_contrib.common.recurrent.policies"] = policies
    recurrent.policies = policies  # type: ignore[attr-defined]


_install_sb3_stub()

import train_model_multi_patch as train_script  # noqa: E402  (import after stub)


def test_dataset_split_none_skips_offline_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path):
    seasonality = tmp_path / "seasonality.json"
    seasonality.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mode: train\n", encoding="utf-8")

    called = False

    def fake_resolve(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("resolve_split_bundle should not be called when dataset-split is none")

    def stop_after_config(_path):
        raise RuntimeError("stop after config load")

    monkeypatch.setattr(train_script, "resolve_split_bundle", fake_resolve)
    monkeypatch.setattr(train_script, "load_config", stop_after_config)

    argv = [
        "train_model_multi_patch.py",
        "--config",
        str(config_path),
        "--dataset-split",
        "none",
        "--liquidity-seasonality",
        str(seasonality),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(RuntimeError, match="stop after config load"):
        train_script.main()

    assert called is False
