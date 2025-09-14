import sys
import json
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import services.signal_bus as sb


def test_publish_signal_dedup(tmp_path):
    # redirect state path to temporary location to avoid polluting repo
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb._loaded = False
    sb.load_state()

    sent = []
    now = 1000

    def send_fn(payload):
        sent.append(payload)

    sid = sb.signal_id("BTCUSDT", 1)

    # first call should send and mark emitted
    assert sb.publish_signal("BTCUSDT", 1, {"p": 1}, send_fn, ttl_ms=100, now_ms=now)
    assert sent == [{"p": 1}]
    assert sb._SEEN[sid] == now + 100

    # duplicate before expiry should be skipped
    assert not sb.publish_signal("BTCUSDT", 1, {"p": 2}, send_fn, ttl_ms=100, now_ms=now + 50)
    assert sent == [{"p": 1}]
    assert sb._SEEN[sid] == now + 100

    # after expiration it should send again
    assert sb.publish_signal("BTCUSDT", 1, {"p": 3}, send_fn, ttl_ms=100, now_ms=now + 200)
    assert sent == [{"p": 1}, {"p": 3}]
    assert sb._SEEN[sid] == now + 200 + 100


def test_publish_signal_payload_fields(tmp_path):
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb._loaded = False
    sb.load_state()

    captured = []

    def send_fn(payload):
        captured.append(payload)

    payload = {"score": 1.23, "features_hash": "abc"}
    assert sb.publish_signal("ETHUSDT", 2, payload, send_fn, ttl_ms=100, now_ms=1000)
    assert captured == [payload]


def test_publish_signal_disabled(tmp_path):
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
    sb._loaded = False
    sb.load_state()

    sent = []

    def send_fn(payload):
        sent.append(payload)

    sb.config.enabled = False
    try:
        assert not sb.publish_signal("BTCUSDT", 1, {"p": 1}, send_fn, ttl_ms=100, now_ms=0)
        assert sent == []
        assert sb._SEEN == {}
    finally:
        sb.config.enabled = True


def test_load_and_flush_state(tmp_path):
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()
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
