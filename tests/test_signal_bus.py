import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import services.signal_bus as sb


def test_publish_signal_dedup(tmp_path):
    # redirect state path to temporary location to avoid polluting repo
    sb._STATE_PATH = tmp_path / "seen.json"
    sb._SEEN.clear()

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
