from __future__ import annotations

import sys
import types
from typing import Any

import pandas as pd
import pytest


def _install_sb3_stub() -> None:
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

    class _DummyPolicy:  # pragma: no cover - placeholder for imports
        pass

    policies.RecurrentActorCriticPolicy = _DummyPolicy
    sys.modules["sb3_contrib.common.recurrent.policies"] = policies
    recurrent.policies = policies  # type: ignore[attr-defined]


_install_sb3_stub()

import train_model_multi_patch as train_script  # noqa: E402  (import after stub)
from core_config import TrainDataConfig  # noqa: E402


def _build_payload(split_name: str = "demo") -> dict[str, Any]:
    return {
        "datasets": {
            split_name: {
                "name": split_name,
                "version": "v1",
                "splits": {
                    "time": {
                        "train": [
                            {
                                "start": "2024-01-01T00:00:00Z",
                                "end": "2024-03-31T23:59:59Z",
                            }
                        ],
                        "val": [
                            {
                                "start": "2024-04-01T00:00:00Z",
                                "end": "2024-04-30T23:59:59Z",
                            }
                        ],
                    }
                },
            }
        }
    }


def _make_data_frame(timestamps: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": timestamps, "wf_role": ["none"] * len(timestamps)})


def test_inline_split_overrides_populates_validation_window() -> None:
    payload = _build_payload()
    overrides = train_script._extract_offline_split_overrides(payload, "demo")
    assert "val" in overrides and overrides["val"], "Expected validation override from payload"

    data_cfg = TrainDataConfig(
        symbols=[],
        timeframe="1h",
        split_overrides=overrides,
        split_version="v1",
    )
    version, time_splits = train_script._load_time_splits(data_cfg)
    assert version == "v1"

    train_bounds = overrides["train"][0]
    val_bounds = overrides["val"][0]

    timestamps = [
        train_bounds["start_ts"],
        train_bounds["start_ts"] + 1,
        val_bounds["start_ts"],
        val_bounds["start_ts"] + 1,
    ]
    df = _make_data_frame(timestamps)
    annotated, _ = train_script._apply_role_column(
        df,
        time_splits,
        timestamp_column="timestamp",
        role_column="wf_role",
    )

    dfs_with_roles = {"demo": annotated}
    # Should not raise: validation rows are present thanks to overrides.
    train_script._ensure_validation_split_present(
        dfs_with_roles,
        time_splits,
        timestamp_column="timestamp",
        role_column="wf_role",
    )

    assert (annotated["wf_role"] == "val").any(), "Validation selection should not be empty"


def test_validation_split_diagnostic_on_empty_selection() -> None:
    payload = _build_payload()
    overrides = train_script._extract_offline_split_overrides(payload, "demo")
    data_cfg = TrainDataConfig(
        symbols=[],
        timeframe="1h",
        split_overrides=overrides,
        split_version="v1",
    )
    _, time_splits = train_script._load_time_splits(data_cfg)

    train_bounds = overrides["train"][0]
    # Place all timestamps before the validation window to force zero overlap.
    timestamps = [
        train_bounds["start_ts"],
        train_bounds["start_ts"] + 1,
        train_bounds["end_ts"] - 1,
    ]
    df = _make_data_frame(timestamps)
    annotated, _ = train_script._apply_role_column(
        df,
        time_splits,
        timestamp_column="timestamp",
        role_column="wf_role",
    )
    dfs_with_roles = {"demo": annotated}

    with pytest.raises(SystemExit) as excinfo:
        train_script._ensure_validation_split_present(
            dfs_with_roles,
            time_splits,
            timestamp_column="timestamp",
            role_column="wf_role",
        )

    message = str(excinfo.value)
    assert "Validation split is empty" in message
    assert "Configured validation intervals" in message
    assert "Observed data coverage" in message
    assert "does not overlap" in message
