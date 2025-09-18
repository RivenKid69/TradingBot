# Seasonality Walkthrough

This example demonstrates how to derive hour-of-week multipliers,
configure the simulator with them and validate the result.

All timestamp data **must** be in UTC to avoid daylight-saving time ambiguity. Feeding local-time values into multiplier computations will misalign the hour-of-week indices, so convert any inputs to UTC before processing.

## Generate multipliers from sample data

```python
import pandas as pd
from scripts.build_hourly_seasonality import compute_multipliers

base = pd.Timestamp("2024-01-01", tz="UTC").timestamp()*1000
MS = 3_600_000  # one hour in ms

# toy dataset covering the first four hours of the week
frame = pd.DataFrame({
    "ts_ms": [base, base+MS, base+2*MS, base+3*MS],
    "quantity": [100, 200, 100, 50],
    "latency_ms": [100, 120, 110, 90],
    "spread_bps": [5, 6, 5, 4],
})

multipliers, _ = compute_multipliers(frame, min_samples=1)
print(multipliers["liquidity"][:4])
print(multipliers["latency"][:4])
print(multipliers["spread"][:4])
```

Output:

```
[0.88888889 1.77777778 0.88888889 0.44444444]
[0.95238095 1.14285714 1.04761905 0.85714286]
[1.   1.2  1.   0.8 ]
```

## Configure `ExecutionSimulator` and `LatencyImpl`

```python
from execution_sim import ExecutionSimulator
from impl_latency import LatencyImpl

sim = ExecutionSimulator(
    liquidity_seasonality=multipliers["liquidity"],
    spread_seasonality=multipliers["spread"],
    seasonality_interpolate=True,  # enable minute-level interpolation
)

lat_cfg = {
    "base_ms": 100,
    "jitter_ms": 20,
    "seasonality_path": "data/latency/liquidity_latency_seasonality.json",
    "seasonality_override": multipliers["latency"],
    "seasonality_interpolate": True,  # smooth latency multipliers
}
lat = LatencyImpl.from_dict(lat_cfg)
lat.attach_to(sim)

ts = int(base)  # Monday 00:00 UTC
sim.set_market_snapshot(bid=100.0, ask=101.0, liquidity=1.0, spread_bps=1.0, ts_ms=ts)
print(sim._last_liquidity, sim._last_spread_bps)
print(sim.latency.sample(ts))
```

Output:

```
0.8888888888888888 1.0
{'total_ms': 107, 'spike': False, 'timeout': False, 'attempts': 1}
```

## Run validation

Save the toy dataset and multipliers to disk and run the validator:

```bash
python scripts/validate_seasonality.py \
  --historical sample.csv \
  --multipliers multipliers.json \
  --threshold 0.5
```

Example output:

```
Historical data checksum written to sample.csv.sha256
Metric: liquidity
  max_rel_diff: 0.3333
  mean_rel_diff: 0.1111
Metric: spread_bps
  max_rel_diff: 0.1000
  mean_rel_diff: 0.0012
Metric: latency_ms
  max_rel_diff: 0.1667
  mean_rel_diff: 0.0023
✅ Seasonality validation passed
```
