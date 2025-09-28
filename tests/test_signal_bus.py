import sys
import json
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.spot_signals import (
    SpotSignalEconomics,
    SpotSignalTargetWeightPayload,
    build_envelope,
)
import services.signal_bus as sb
from services import ops_kill_switch


@pytest.fixture(autouse=True)
def _reset_ops(tmp_path):
    ops_kill_switch.init(
        {"state_path": str(tmp_path / "ops_state.json"), "flag_path": str(tmp_path / "ops_flag")}
    )
    ops_kill_switch.manual_reset()


def test_publish_signal_dedup(tmp_path):
    # redirect state path to temporary location to avoid polluting repo
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb.dropped_by_reason.clear()
    sb._loaded = False
    sb.load_state()

    sent = []
    now = 1000

    def send_fn(payload):
        sent.append(payload)

    sid = sb.signal_id("BTCUSDT", 1)

    first_payload = _make_payload(0.1)
    assert sb.publish_signal(
        "BTCUSDT", 1, first_payload, send_fn, expires_at_ms=now + 100, now_ms=now
    )
    assert sent[0]["payload"]["target_weight"] == pytest.approx(0.1)
    assert sent[0]["payload"]["economics"]["edge_bps"] == pytest.approx(10.0)
    assert sb._SEEN[sid] == now + 100

    # duplicate before expiry should be skipped
    assert not sb.publish_signal(
        "BTCUSDT", 1, _make_payload(0.2), send_fn, expires_at_ms=now + 150, now_ms=now + 50
    )
    assert len(sent) == 1
    assert sb._SEEN[sid] == now + 100

    # after expiration it should send again
    assert sb.publish_signal(
        "BTCUSDT",
        1,
        _make_payload(0.3),
        send_fn,
        expires_at_ms=now + 200 + 100,
        now_ms=now + 200,
    )
    assert len(sent) == 2
    assert sent[-1]["payload"]["target_weight"] == pytest.approx(0.3)
    assert sb._SEEN[sid] == now + 200 + 100


def test_duplicate_counter_resets(tmp_path):
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb.dropped_by_reason.clear()
    sb._loaded = False
    sb.load_state()

    sent: list[dict[str, Any]] = []
    now = 1000

    def send_fn(payload):
        sent.append(payload)

    assert sb.publish_signal(
        "BTCUSDT", 1, _make_payload(0.1), send_fn, expires_at_ms=now + 100, now_ms=now
    )
    assert ops_kill_switch._counters["duplicates"] == 0

    assert not sb.publish_signal(
        "BTCUSDT", 1, _make_payload(0.2), send_fn, expires_at_ms=now + 150, now_ms=now
    )
    assert ops_kill_switch._counters["duplicates"] == 1

    assert sb.publish_signal(
        "ETHUSDT", 2, _make_payload(0.3), send_fn, expires_at_ms=now + 200, now_ms=now
    )
    assert ops_kill_switch._counters["duplicates"] == 0


def test_publish_signal_custom_dedup_key(tmp_path):
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb.dropped_by_reason.clear()
    sb._loaded = False
    sb.load_state()

    sent: list[dict[str, Any]] = []
    now = 1000

    def send_fn(payload):
        sent.append(payload)

    # first call with custom dedup key should send
    assert sb.publish_signal(
        "BTCUSDT",
        1,
        _make_payload(0.2),
        send_fn,
        expires_at_ms=now + 100,
        now_ms=now,
        dedup_key="custom1",
    )
    # duplicate with same key should be skipped
    assert not sb.publish_signal(
        "BTCUSDT",
        1,
        _make_payload(0.3),
        send_fn,
        expires_at_ms=now + 200,
        now_ms=now + 50,
        dedup_key="custom1",
    )
    # different key should send
    assert sb.publish_signal(
        "BTCUSDT",
        1,
        _make_payload(0.4),
        send_fn,
        expires_at_ms=now + 300,
        now_ms=now + 60,
        dedup_key="custom2",
    )

    assert [row["payload"]["target_weight"] for row in sent] == [pytest.approx(0.2), pytest.approx(0.4)]


def test_publish_signal_payload_fields(tmp_path):
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb.dropped_by_reason.clear()
    sb._loaded = False
    sb.load_state()

    captured = []

    def send_fn(payload):
        captured.append(payload)

    payload = _make_payload(0.5, edge=20.0)
    assert sb.publish_signal("ETHUSDT", 2, payload, send_fn, expires_at_ms=1100, now_ms=1000)
    assert len(captured) == 1
    envelope = captured[0]
    assert envelope["symbol"] == "ETHUSDT"
    assert envelope["bar_close_ms"] == 2
    assert envelope["expires_at_ms"] == 1100
    assert envelope["payload"]["target_weight"] == pytest.approx(0.5)
    economics = envelope["payload"]["economics"]
    assert economics["edge_bps"] == pytest.approx(20.0)
    assert economics["net_bps"] == pytest.approx(15.0)
    assert "signature" not in envelope


def test_publish_signal_rejects_expired_valid_until(tmp_path):
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb.dropped_by_reason.clear()
    sb._loaded = False
    sb.load_state()

    sent: list[dict[str, Any]] = []

    def send_fn(payload):
        sent.append(payload)

    now = 10_000
    stale_payload = _make_payload(0.1).model_dump()
    stale_payload["valid_until_ms"] = now - 1

    assert not sb.publish_signal(
        "BTCUSDT",
        1,
        stale_payload,
        send_fn,
        expires_at_ms=now + 5_000,
        now_ms=now,
    )
    assert sent == []
    assert sb.dropped_by_reason["valid_until_expired"] == 1

    fresh_payload = _make_payload(0.2).model_dump()
    fresh_payload["valid_until_ms"] = now + 5_000

    assert sb.publish_signal(
        "BTCUSDT",
        1,
        fresh_payload,
        send_fn,
        expires_at_ms=now + 10_000,
        now_ms=now,
    )
    assert len(sent) == 1


def test_publish_signal_disabled(tmp_path):
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb.dropped_by_reason.clear()
    sb._loaded = False
    sb.load_state()

    sent = []

    def send_fn(payload):
        sent.append(payload)

    sb.config.enabled = False
    try:
        assert not sb.publish_signal(
            "BTCUSDT", 1, _make_payload(0.1), send_fn, expires_at_ms=100, now_ms=0
        )
        assert sent == []
        assert sb._SEEN == {}
    finally:
        sb.config.enabled = True


def test_load_and_flush_state(tmp_path):
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb.dropped_by_reason.clear()
    sb._loaded = False
    sb.load_state()
    assert sb._SEEN == {}

    now = int(time.time() * 1000)
    sid = sb.signal_id("BTCUSDT", 1)
    sb.mark_emitted(sid, expires_at_ms=now + 100, now_ms=now)
    assert json.loads(sb._STATE_PATH.read_text()) == {sid: now + 100}

    # Add expired entry and ensure mark_emitted purges it
    expired_sid = sb.signal_id("ETHUSDT", 2)
    sb._SEEN[expired_sid] = now - 1
    sb.flush_state()
    sb.mark_emitted(sid, expires_at_ms=now + 200, now_ms=now)
    data = json.loads(sb._STATE_PATH.read_text())
    assert expired_sid not in data
    assert data[sid] == now + 200

    # Prepare file with expired and valid entries to test load_state purge
    valid_sid = sid
    future_exp = now + 5000
    past_exp = now - 5000
    sb._STATE_PATH.write_text(
        json.dumps({valid_sid: future_exp, expired_sid: past_exp})
    )
    sb._SEEN.clear()
    sb._loaded = False
    sb.load_state()
    assert sb._SEEN == {valid_sid: future_exp}
    assert json.loads(sb._STATE_PATH.read_text()) == {valid_sid: future_exp}


def test_publish_signal_loads_once_and_flushes(tmp_path):
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb.dropped_by_reason.clear()
    sb._loaded = False

    calls: list[int] = []
    orig_load = sb.load_state

    def _load(*a, **k):
        calls.append(1)
        return orig_load(*a, **k)

    sb.load_state = _load  # type: ignore
    try:
        sent: list[dict[str, Any]] = []

        def send_fn(payload):
            sent.append(payload)

        now = 1000
        sid = sb.signal_id("BTCUSDT", 1)
        ok = sb.publish_signal(
            "BTCUSDT", 1, _make_payload(0.2), send_fn, expires_at_ms=now + 100, now_ms=now
        )
        assert ok
        assert calls == [1]
        assert json.loads(sb._STATE_PATH.read_text()) == {sid: now + 100}
    finally:
        sb.load_state = orig_load  # type: ignore


def test_load_state_reinit_on_corruption(tmp_path):
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb.dropped_by_reason.clear()
    sb._loaded = False
    sb._STATE_PATH.write_text("not-json")
    sb.load_state()
    assert sb._SEEN == {}
    assert json.loads(sb._STATE_PATH.read_text()) == {}


def test_publish_signal_csv_logging(tmp_path):
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb.dropped_by_reason.clear()
    sb._loaded = False
    sb.load_state()

    sb.OUT_CSV = str(tmp_path / "out.csv")
    sb.DROPS_CSV = str(tmp_path / "drop.csv")

    sent = []

    def send_fn(payload):
        sent.append(payload)

    now = 1000
    ok = sb.publish_signal(
        "BTCUSDT", 1, _make_payload(0.1), send_fn, expires_at_ms=now + 100, now_ms=now
    )
    assert ok
    assert sent[0]["payload"]["target_weight"] == pytest.approx(0.1)
    out_path = Path(sb.OUT_CSV)
    assert out_path.exists()
    assert len(out_path.read_text().strip().splitlines()) == 2

    # expired signal should be logged to drops CSV and not sent
    ok = sb.publish_signal(
        "BTCUSDT", 2, _make_payload(0.2), send_fn, expires_at_ms=now - 1, now_ms=now
    )
    assert not ok
    assert len(sent) == 1
    drop_path = Path(sb.DROPS_CSV)
    assert drop_path.exists()
    assert len(drop_path.read_text().strip().splitlines()) == 2

    sb.OUT_CSV = None
    sb.DROPS_CSV = None


def test_log_drop_counts():
    sb.dropped_by_reason.clear()
    envelope = build_envelope(
        symbol="BTC",
        bar_close_ms=1,
        expires_at_ms=2,
        payload=_make_payload(),
    )
    sb.log_drop(envelope, "RISK_TEST")
    assert sb.dropped_by_reason["RISK_TEST"] == 1
def _make_payload(weight: float = 0.1, *, edge: float = 10.0) -> SpotSignalTargetWeightPayload:
    economics = SpotSignalEconomics(
        edge_bps=edge,
        cost_bps=5.0,
        net_bps=edge - 5.0,
        turnover_usd=100.0,
        act_now=True,
    )
    return SpotSignalTargetWeightPayload(target_weight=weight, economics=economics)

