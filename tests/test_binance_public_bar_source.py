import json
from decimal import Decimal

import impl_binance_public


def test_handle_message_populates_volume_quote(monkeypatch):
    monkeypatch.setattr(impl_binance_public, "websockets", object())

    source = impl_binance_public.BinancePublicBarSource(timeframe="1m")

    message = json.dumps(
        {
            "data": {
                "k": {
                    "t": 1,
                    "s": "btcusdt",
                    "o": "1",
                    "h": "2",
                    "l": "0.5",
                    "c": "1.5",
                    "v": "10",
                    "q": "100",
                    "n": 3,
                    "x": True,
                }
            }
        }
    )

    source._handle_message(message)

    bar = source._q.get_nowait()
    assert bar.volume_base == Decimal("10")
    assert bar.volume_quote == Decimal("100")
