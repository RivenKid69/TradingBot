# Data degradation

`DataDegradationConfig` injects missing or delayed bars to emulate
real‑world data issues. All fields are optional; by default the stream is
unchanged.

## Configuration fields

| Field | Type | Description |
|------|------|-------------|
| `stale_prob` | float | Chance to repeat the previous bar instead of the new one. |
| `drop_prob` | float | Probability to drop the bar entirely. |
| `dropout_prob` | float | Probability to delay delivery by up to `max_delay_ms`. |
| `max_delay_ms` | int | Upper bound on the random delay in milliseconds. |
| `seed` | int | RNG seed for reproducibility. |

## Examples

### Offline CSV

```yaml
# config_sim.yaml
data_degradation:
  stale_prob: 0.1
  drop_prob: 0.05
  dropout_prob: 0.2
  max_delay_ms: 50
  seed: 42
```

By default, runners load symbols from ``data/universe/symbols.json``.

Run:

```bash
python script_backtest.py --config config_sim.yaml
```

### Live WebSocket

```python
from binance_ws import BinanceWS
from services.event_bus import EventBus
from config import DataDegradationConfig
from services.universe import get_symbols

cfg = DataDegradationConfig(stale_prob=0.05, drop_prob=0.02,
                            dropout_prob=0.1, max_delay_ms=50, seed=7)
bus = EventBus(queue_size=1000, drop_policy="newest")
ws = BinanceWS(symbols=get_symbols(), bus=bus, data_degradation=cfg)
ws.run()
```

## Monitoring

Components log degradation statistics on shutdown. Look for messages
`OfflineCSVBarSource degradation`, `BinanceWS degradation` or
`LatencyQueue degradation` in the INFO log level to inspect drop and
delay ratios.
