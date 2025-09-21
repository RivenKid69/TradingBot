import os
from datetime import datetime, timedelta, timezone

from services.signal_csv_writer import SignalCSVWriter


def _ts_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def test_header_and_append(tmp_path):
    path = tmp_path / "signals.csv"
    w = SignalCSVWriter(str(path))
    ts = _ts_ms(datetime.utcnow())
    w.write({"ts_ms": ts, "symbol": "BTC", "side": "BUY", "volume_frac": 1, "score": 0.1, "features_hash": "x"})
    w.flush_fsync()
    w.close()

    w2 = SignalCSVWriter(str(path))
    w2.write({"ts_ms": ts, "symbol": "ETH", "side": "SELL", "volume_frac": 2, "score": 0.2, "features_hash": "y"})
    w2.close()

    lines = path.read_text().strip().splitlines()
    assert lines[0].startswith("ts_ms")
    assert len(lines) == 3  # header + 2 rows


def test_rotation_on_init(tmp_path):
    path = tmp_path / "signals.csv"
    old_day = datetime.utcnow().date() - timedelta(days=1)
    rotated = tmp_path / f"signals-{old_day.isoformat()}.csv"
    path.write_text("ts_ms,symbol,side,volume_frac,score,features_hash\n1,BTC,BUY,0.1,0.2,fh\n")
    ts_old = datetime.combine(old_day, datetime.min.time(), tzinfo=timezone.utc).timestamp()
    os.utime(path, (ts_old, ts_old))

    w = SignalCSVWriter(str(path))
    w.write({"ts_ms": _ts_ms(datetime.utcnow()), "symbol": "BTC", "side": "BUY", "volume_frac": 1, "score": 0.1, "features_hash": "x"})
    w.close()

    assert rotated.exists()
    assert path.exists()
    assert rotated.read_text().strip().splitlines()[0].startswith("ts_ms")


def test_rotation_on_write(tmp_path):
    path = tmp_path / "signals.csv"
    w = SignalCSVWriter(str(path))
    day1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    w.write({"ts_ms": _ts_ms(day1), "symbol": "BTC", "side": "BUY", "volume_frac": 1, "score": 0.1, "features_hash": "x"})
    w.write({"ts_ms": _ts_ms(day2), "symbol": "BTC", "side": "SELL", "volume_frac": 2, "score": 0.2, "features_hash": "y"})
    w.close()

    rotated = tmp_path / "signals-2024-01-01.csv"
    assert rotated.exists()
    assert path.exists()
    assert len(rotated.read_text().strip().splitlines()) == 2
    assert len(path.read_text().strip().splitlines()) == 2


def test_stats_and_reopen(tmp_path):
    path = tmp_path / "signals.csv"
    w = SignalCSVWriter(str(path), fsync_mode="off")
    now = datetime.utcnow()
    w.write(
        {
            "ts_ms": _ts_ms(now),
            "symbol": "BTC",
            "side": "BUY",
            "volume_frac": 1,
            "score": 0.1,
            "features_hash": "x",
        }
    )
    stats = w.stats()
    assert stats["written"] == 1
    assert stats["errors"] == 0
    w.reopen()
    w.write(
        {
            "ts_ms": _ts_ms(now),
            "symbol": "ETH",
            "side": "SELL",
            "volume_frac": 2,
            "score": 0.2,
            "features_hash": "y",
        }
    )
    stats = w.stats()
    assert stats["written"] == 2
    assert stats["retries"] >= 0
    w.close()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3


def test_rotate_disabled(tmp_path):
    path = tmp_path / "signals.csv"
    w = SignalCSVWriter(str(path), rotate_daily=False)
    day1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    day2 = day1 + timedelta(days=1)
    w.write(
        {
            "ts_ms": _ts_ms(day1),
            "symbol": "BTC",
            "side": "BUY",
            "volume_frac": 1,
            "score": 0.1,
            "features_hash": "x",
        }
    )
    w.write(
        {
            "ts_ms": _ts_ms(day2),
            "symbol": "BTC",
            "side": "SELL",
            "volume_frac": 2,
            "score": 0.2,
            "features_hash": "y",
        }
    )
    w.close()
    assert not (tmp_path / "signals-2024-01-01.csv").exists()
    assert path.exists()
    assert len(path.read_text().strip().splitlines()) == 3
