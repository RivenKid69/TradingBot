from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("requests")

from services.rest_budget import RestBudgetSession


def _session(tmp_path: Path, *, resume: bool = True, enabled: bool = True) -> RestBudgetSession:
    cfg = {
        "checkpoint": {
            "path": str(tmp_path / "ckpt.json"),
            "enabled": enabled,
            "resume_from_checkpoint": resume,
        }
    }
    return RestBudgetSession(cfg)


def test_save_and_load_checkpoint(tmp_path: Path) -> None:
    session = _session(tmp_path)
    payload = {"position": 5, "symbols": ["BTCUSDT", "ETHUSDT"]}
    session.save_checkpoint(payload)

    ckpt_path = tmp_path / "ckpt.json"
    assert ckpt_path.exists()
    on_disk = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert on_disk["position"] == 5
    assert on_disk["symbols"] == ["BTCUSDT", "ETHUSDT"]

    loaded = session.load_checkpoint()
    assert loaded == on_disk


def test_load_checkpoint_disabled(tmp_path: Path) -> None:
    session = _session(tmp_path, resume=False)
    session.save_checkpoint({"position": 1})
    assert session.load_checkpoint() is None


def test_save_checkpoint_non_serialisable(tmp_path: Path) -> None:
    session = _session(tmp_path)
    class Dummy:
        pass

    session.save_checkpoint({"obj": Dummy()})
    assert not (tmp_path / "ckpt.json").exists()
