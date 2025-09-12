# Large Order Execution

Parent orders above a configurable notional threshold are sliced into deterministic child orders.
This mechanism helps control market impact and align simulated execution with real markets.

## Configuration fields

| Field | Type | Description |
|------|------|-------------|
| `notional_threshold` | float | Orders above this notional value trigger a slicing algorithm. |
| `large_order_algo` | str | Slicing method: `"TWAP"` for time-slicing or `"POV"` for participation-of-volume. |
| `pov.participation` | float | Fraction of observed market volume used by each POV child order. |
| `pov.child_interval_s` | int | Seconds between POV child orders. |
| `pov.min_child_notional` | float | Minimum notional per POV child order. |

## Examples

### TWAP

```yaml
notional_threshold: 10000.0
large_order_algo: TWAP
twap:
  child_interval_s: 1
```

A buy order for 50 000 notional generates five children of 10 000 each at one-second intervals:

| Time (s) | Child notional |
|---------|----------------|
| 0       | 10000 |
| 1       | 10000 |
| 2       | 10000 |
| 3       | 10000 |
| 4       | 10000 |

### POV

```yaml
notional_threshold: 10000.0
large_order_algo: POV
pov:
  participation: 0.2
  child_interval_s: 1
  min_child_notional: 1000.0
```

If the market trades 10 000 notional per second, a 50 000 order submits 20% of that volume each second until filled:

| Time (s) | Child notional |
|---------|----------------|
| 0       | 2000 |
| 1       | 2000 |
| 2       | 2000 |
| 3       | 2000 |
| 4       | 2000 |
| …    | …   |

## Determinism and calibration

Child-order schedules are deterministic for a given market stream and configuration. The unit tests
`tests/test_execution_determinism.py` and `tests/test_executor_threshold.py` lock in this behaviour.
Parameters like `participation` should be calibrated on historical data (e.g. impact vs participation
curves) to produce realistic trajectories.

