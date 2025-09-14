import sys
import json
import time
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

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

    # first call should send and mark emitted
    assert sb.publish_signal("BTCUSDT", 1, {"p": 1}, send_fn, expires_at_ms=now + 100, now_ms=now)
    assert sent == [{"p": 1}]
    assert sb._SEEN[sid] == now + 100

    # duplicate before expiry should be skipped
    assert not sb.publish_signal("BTCUSDT", 1, {"p": 2}, send_fn, expires_at_ms=now + 150, now_ms=now + 50)
    assert sent == [{"p": 1}]
    assert sb._SEEN[sid] == now + 100

    # after expiration it should send again
    assert sb.publish_signal("BTCUSDT", 1, {"p": 3}, send_fn, expires_at_ms=now + 200 + 100, now_ms=now + 200)
    assert sent == [{"p": 1}, {"p": 3}]
    assert sb._SEEN[sid] == now + 200 + 100


def test_duplicate_counter_resets(tmp_path):
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb.dropped_by_reason.clear()
    sb._loaded = False
    sb.load_state()

    sent: list[dict[str, int]] = []
    now = 1000

    def send_fn(payload):
        sent.append(payload)

    assert sb.publish_signal("BTCUSDT", 1, {"p": 1}, send_fn, expires_at_ms=now + 100, now_ms=now)
    assert ops_kill_switch._counters["duplicates"] == 0

    assert not sb.publish_signal("BTCUSDT", 1, {"p": 2}, send_fn, expires_at_ms=now + 150, now_ms=now)
    assert ops_kill_switch._counters["duplicates"] == 1

    assert sb.publish_signal("ETHUSDT", 2, {"p": 3}, send_fn, expires_at_ms=now + 200, now_ms=now)
    assert ops_kill_switch._counters["duplicates"] == 0


def test_publish_signal_custom_dedup_key(tmp_path):
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb.dropped_by_reason.clear()
    sb._loaded = False
    sb.load_state()

    sent: list[dict[str, int]] = []
    now = 1000

    def send_fn(payload):
        sent.append(payload)

    # first call with custom dedup key should send
    assert sb.publish_signal(
        "BTCUSDT",
        1,
        {"p": 1},
        send_fn,
        expires_at_ms=now + 100,
        now_ms=now,
        dedup_key="custom1",
    )
    # duplicate with same key should be skipped
    assert not sb.publish_signal(
        "BTCUSDT",
        1,
        {"p": 2},
        send_fn,
        expires_at_ms=now + 200,
        now_ms=now + 50,
        dedup_key="custom1",
    )
    # different key should send
    assert sb.publish_signal(
        "BTCUSDT",
        1,
        {"p": 3},
        send_fn,
        expires_at_ms=now + 300,
        now_ms=now + 60,
        dedup_key="custom2",
    )

    assert sent == [{"p": 1}, {"p": 3}]


def test_publish_signal_payload_fields(tmp_path):
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb.dropped_by_reason.clear()
    sb._loaded = False
    sb.load_state()

    captured = []

    def send_fn(payload):
        captured.append(payload)

    payload = {"score": 1.23, "features_hash": "abc"}
    assert sb.publish_signal("ETHUSDT", 2, payload, send_fn, expires_at_ms=1100, now_ms=1000)
    assert captured == [payload]


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
        assert not sb.publish_signal("BTCUSDT", 1, {"p": 1}, send_fn, expires_at_ms=100, now_ms=0)
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
        sent: list[dict[str, int]] = []

        def send_fn(payload):
            sent.append(payload)

        now = 1000
        sid = sb.signal_id("BTCUSDT", 1)
        ok = sb.publish_signal("BTCUSDT", 1, {"p": 1}, send_fn, expires_at_ms=now + 100, now_ms=now)
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
    ok = sb.publish_signal("BTCUSDT", 1, {"p": 1}, send_fn, expires_at_ms=now + 100, now_ms=now)
    assert ok
    assert sent == [{"p": 1}]
    out_path = Path(sb.OUT_CSV)
    assert out_path.exists()
    assert len(out_path.read_text().strip().splitlines()) == 2

    # expired signal should be logged to drops CSV and not sent
    ok = sb.publish_signal("BTCUSDT", 2, {"p": 2}, send_fn, expires_at_ms=now - 1, now_ms=now)
    assert not ok
    assert sent == [{"p": 1}]
    drop_path = Path(sb.DROPS_CSV)
    assert drop_path.exists()
    assert len(drop_path.read_text().strip().splitlines()) == 2

    sb.OUT_CSV = None
    sb.DROPS_CSV = None


def test_log_drop_counts():
    sb.dropped_by_reason.clear()
    sb.log_drop("BTC", 1, {}, "RISK_TEST")
    assert sb.dropped_by_reason["RISK_TEST"] == 1
