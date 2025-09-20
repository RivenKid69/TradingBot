from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Mapping

import pytest

requests = pytest.importorskip("requests")
if not hasattr(requests, "Session"):
    pytest.skip("requests.Session not available", allow_module_level=True)

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
    session.save_checkpoint(
        payload,
        last_symbol="ethusdt",
        last_range=(1_690_000_000_000, 1_690_008_600_000),
        progress_pct=12.5,
    )

    ckpt_path = tmp_path / "ckpt.json"
    assert ckpt_path.exists()
    on_disk = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert on_disk["data"]["position"] == 5
    assert on_disk["data"]["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert on_disk["last_symbol"] == "ETHUSDT"
    assert on_disk["last_range"] == [1_690_000_000_000, 1_690_008_600_000]
    assert on_disk["progress_pct"] == pytest.approx(12.5)
    assert isinstance(on_disk["saved_at"], str)

    loaded = session.load_checkpoint()
    assert isinstance(loaded, Mapping)
    assert loaded["position"] == 5
    assert loaded["data"]["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert loaded["last_symbol"] == "ETHUSDT"
    assert loaded["last_range"] == [1_690_000_000_000, 1_690_008_600_000]
    assert loaded["progress_pct"] == pytest.approx(12.5)
    assert loaded["_checkpoint"]["last_symbol"] == "ETHUSDT"


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
    checkpoint = session.load_checkpoint()
    assert isinstance(checkpoint, Mapping)
    assert checkpoint["position"] == 1
    assert checkpoint["progress_pct"] is None

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
        assert session.is_cached("https://example.com/api", endpoint="GET /api")
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
    assert stats["cache_totals"]["hits"] == 1
    assert stats["cache_totals"]["stores"] == 1
    assert stats["cache_totals"]["misses"] >= 1
    assert stats["checkpoint"] == {"loads": 0, "saves": 0}
    assert pytest.approx(1.5) == stats["request_tokens"]["GET /api"]
    json.dumps(stats)


def test_last_response_metadata_tracks_weights(tmp_path: Path) -> None:
    response = _DummyResponse({"value": 1})
    response.headers = {
        "X-MBX-USED-WEIGHT-1M": "123",
        "X-MBX-ORDER-COUNT-1M": "5",
        "X-OTHER": "ignored",
    }
    dummy_session = _DummySession([response])
    session = RestBudgetSession({}, session=dummy_session)
    try:
        payload = session.get(
            "https://example.com/api",
            endpoint="GET /api",
            budget="test-budget",
            tokens=2.0,
        )
        assert payload == {"value": 1}
        metadata = session.get_last_response_metadata()
    finally:
        session.close()

    assert metadata is not None
    assert metadata["budget"] == "test-budget"
    assert metadata["cache_hit"] is False
    assert metadata["tokens"] == pytest.approx(2.0)
    weights = metadata.get("binance_weights")
    assert weights == {"x-mbx-used-weight-1m": 123.0, "x-mbx-order-count-1m": 5.0}


def test_qps_bucket_support() -> None:
    session = RestBudgetSession({"global": {"qps": 2.5, "burst": 5}})
    try:
        bucket = session._global_bucket
        assert bucket is not None
        assert bucket.rps == pytest.approx(2.5)
        assert bucket.burst == pytest.approx(5.0)
    finally:
        session.close()


def test_rest_budget_disabled_flag() -> None:
    session = RestBudgetSession(
        {
            "enabled": False,
            "global": {"qps": 2, "burst": 3},
            "endpoints": {"klines": {"qps": 1, "burst": 2}},
        }
    )
    try:
        assert session._global_bucket is None
        assert session._endpoint_buckets == {}
    finally:
        session.close()
