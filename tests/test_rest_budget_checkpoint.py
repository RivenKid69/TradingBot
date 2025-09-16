from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")

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


def test_stats_plan_and_checkpoint(tmp_path: Path) -> None:
    session = RestBudgetSession(
        {
            "checkpoint": {
                "path": str(tmp_path / "ckpt.json"),
                "enabled": True,
                "resume_from_checkpoint": True,
            }
        }
    )
    session.plan_request("GET /api/test", count=3, tokens=2.5)
    session.save_checkpoint({"position": 1})
    assert session.load_checkpoint() == {"position": 1}

    stats = session.stats()
    assert stats["planned_requests"] == {"GET /api/test": 3}
    assert stats["planned_tokens"]["GET /api/test"] == pytest.approx(7.5)
    assert stats["checkpoint"] == {"loads": 1, "saves": 1}
    json.dumps(stats)  # should be serialisable


class _DummyResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.url = ""

    def json(self) -> object:
        return self._payload

    @property
    def text(self) -> str:
        return json.dumps(self._payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


class _DummySession(requests.Session):
    def __init__(self, responses: list[_DummyResponse]) -> None:
        super().__init__()
        self._responses: deque[_DummyResponse] = deque(responses)

    def get(self, url: str, params=None, headers=None, timeout=None):  # type: ignore[override]
        if not self._responses:
            raise AssertionError("no more responses")
        resp = self._responses.popleft()
        resp.url = url
        return resp


def test_stats_requests_and_cache(tmp_path: Path) -> None:
    dummy_session = _DummySession([_DummyResponse({"value": 42})])
    cfg = {
        "cache": {
            "dir": str(tmp_path / "cache"),
            "ttl_days": 1,
            "mode": "read_write",
        }
    }
    session = RestBudgetSession(cfg, session=dummy_session)
    try:
        payload = session.get("https://example.com/api", endpoint="GET /api", tokens=1.5)
        assert payload == {"value": 42}
        payload_cached = session.get("https://example.com/api", endpoint="GET /api", tokens=1.5)
        assert payload_cached == {"value": 42}
    finally:
        session.close()

    stats = session.stats()
    assert stats["requests"] == {"GET /api": 1}
    assert stats["planned_requests"] == {}
    assert stats["cache_stores"] == {"GET /api": 1}
    assert stats["cache_hits"]["GET /api"] == 1
    assert stats["cache_misses"]["GET /api"] >= 1
    assert stats["checkpoint"] == {"loads": 0, "saves": 0}
    assert pytest.approx(1.5) == stats["request_tokens"]["GET /api"]
    json.dumps(stats)
