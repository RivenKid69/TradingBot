# Hour-of-Week Seasonality

Certain hours of the week exhibit systematic patterns in market depth, bid-ask spreads and order-processing delays. To capture this behaviour, the simulator supports **hour-of-week multipliers** that scale baseline liquidity, spread and latency parameters. Multipliers are indexed from `0` (Monday 00:00 UTC) to `167` (Sunday 23:00 UTC). All hour-of-week calculations use `datetime.utcfromtimestamp`, so timestamps must be interpreted as UTC.

All timestamp inputs **must** be expressed in UTC to avoid daylight-saving time (DST) ambiguity. Feeding local-time data into multiplier computations will misalign hour-of-week indices, so always convert source data to UTC before generating or applying multipliers.

## `liquidity_latency_seasonality.json` format

The JSON file contains up to three arrays with 168 floating-point numbers each.
A template with all multipliers set to `1.0` is available at `configs/liquidity_latency_seasonality.sample.json`.
It can either be a flat mapping or nested under instrument symbols:

```json
{
  "liquidity": [1.0, 1.1, ...],
  "latency":   [1.0, 0.9, ...],
  "spread":    [1.0, 0.8, ...]
}
```

When storing multipliers for multiple instruments in a single file the arrays
are grouped by symbol:

```json
{
  "BTCUSDT": {
    "liquidity": [1.0, 1.1, ...],
    "latency":   [1.0, 0.9, ...],
    "spread":    [1.0, 0.8, ...]
  },
  "hour_of_week_definition": "0=Monday 00:00 UTC"
}
```

`liquidity` multiplies available volume, `latency` scales simulated execution delays while `spread` adjusts the baseline bid-ask spread (in bps). Missing arrays or indices default to `1.0`.

During simulation the baseline parameters are adjusted with the hour-of-week multiplier for hour `h`:

```text
liquidity_adj = base_liq * m_liq[h]
spread_adj    = base_spread_bps * m_spread[h]
latency_adj   = base_latency_ms * m_lat[h]
```

Each array therefore captures the relative deviation from the baseline for the given hour of week.

## Quick inspection

Visualise the multipliers to ensure they look reasonable:

```bash
python scripts/plot_seasonality.py --multipliers data/latency/liquidity_latency_seasonality.json
```

The script writes line charts and heatmaps for liquidity and latency multipliers to `reports/seasonality/plots`.

## Regenerating multipliers from historical data

1. Prepare a CSV or Parquet file with columns such as `ts_ms`, `quantity`, `latency_ms` or `spread_bps`.
2. Run the helper script to compute averages for each hour of week and normalise them:

   ```bash
   python scripts/build_hourly_seasonality.py --data path/to/trades.parquet --out data/latency/liquidity_latency_seasonality.json

   # To wrap the arrays under a specific symbol:
   python scripts/build_hourly_seasonality.py --data path/to/trades.parquet --out data/latency/liquidity_latency_seasonality.json --symbol BTCUSDT
   ```
3. Optionally smooth the multipliers by applying a circular rolling mean
   and/or shrinking values towards 1.0:

   ```bash
   python scripts/build_hourly_seasonality.py \
     --data path/to/trades.parquet \
     --out data/latency/liquidity_latency_seasonality.json \
     --smooth-window 3 \
     --smooth-alpha 0.1
   ```

4. Optionally verify the multipliers against the original dataset:

   ```bash
   python scripts/validate_seasonality.py --historical path/to/trades.parquet --multipliers data/latency/liquidity_latency_seasonality.json
   ```
   See [seasonality_QA.md](seasonality_QA.md) for QA steps and acceptance thresholds.

5. Optionally iterate on the multipliers by feeding previous validation metrics
   back into the generator. Save per-hour relative errors from a validation run
   to a JSON file and pass it via `--prior-metrics`:

   ```bash
   python scripts/build_hourly_seasonality.py \
     --data path/to/trades.parquet \
     --out data/latency/liquidity_latency_seasonality.json \
     --prior-metrics reports/seasonality/validation_metrics.json
   ```
   The script converts each error ``e`` into a weight ``1/(1+e)`` to down-weight
    hours that previously deviated from historical data, then renormalises the
    multipliers so their average remains close to ``1.0``. Repeat the
    generate→validate cycle until the validation metrics stabilise.

## Multiplier computation and normalisation rationale

For a given metric ``x`` (liquidity, spread or latency) the helper script first
computes the average value for each hour of week ``h`` and for the entire
dataset:

```text
avg_h[h] = mean(x | hour_of_week == h)
avg_all  = mean(x)
m[h]     = avg_h[h] / avg_all
```

During simulation the baseline parameter is multiplied by ``m[h]`` as shown
above.  Dividing by the global average normalises the multipliers so their mean
is near ``1.0``. This preserves the overall weekly level while highlighting
relative intraday patterns.  Using ratios rather than absolute values also
makes the multipliers scale-free, allowing them to be reused across symbols or
periods with different volume regimes.

## Enabling seasonality in configs

Seasonality can be activated either via CLI flags or directly in YAML configs:

```bash
python script_backtest.py --config configs/config_sim.yaml --liquidity-seasonality data/latency/liquidity_latency_seasonality.json
```yaml
liquidity_seasonality_path: "data/latency/liquidity_latency_seasonality.json"
seasonality_interpolate: true  # default false; enable minute-level interpolation

latency:
  seasonality_path: "data/latency/liquidity_latency_seasonality.json"
  seasonality_interpolate: true
```

## Manual overrides

In some cases you may want to tweak the computed multipliers. Both
`ExecutionSimulator` and `LatencyImpl` accept **override arrays** of 168
values that are multiplied element-wise with the base multipliers. The
override file shares the same structure as
`liquidity_latency_seasonality.json`:

```json
{
  "liquidity": [1.0, 0.8, ...],
  "latency":   [1.0, 1.2, ...],
  "spread":    [1.0, 1.1, ...]
}
```

Overrides can be supplied via constructor arguments
(`liquidity_seasonality_override`, `spread_seasonality_override` or
`seasonality_override`) or by specifying paths in config files
(`liquidity_seasonality_override_path`, `latency.seasonality_override_path`).
Precedence is as follows:

1. Arrays passed directly to constructors.
2. Arrays embedded in config objects.
3. Arrays loaded from `*_override_path` files.
4. Base multipliers from `liquidity_latency_seasonality.json`.
5. Default multipliers of `1.0`.

Missing entries default to `1.0`, so partial overrides are permitted.

## Runtime reloading

`ExecutionSimulator` and `LatencyImpl` can refresh multipliers without a
restart. Set `seasonality_auto_reload: true` in the simulator or latency
config to enable a background watcher. The file specified by
`*_seasonality_path` is polled once per minute and, if modified, the new
arrays replace the old ones atomically. Reload events are emitted via the
`seasonality` logger.

To avoid inconsistent reads, update the JSON file using an atomic rename:

```bash
python scripts/build_hourly_seasonality.py --out tmp.json
mv tmp.json data/latency/liquidity_latency_seasonality.json
```

Operators can therefore tweak multipliers and have the running system pick
up changes automatically.

## Disabling seasonality

Hourly multipliers are enabled by default. To ignore them, set the
`use_seasonality` flag to `false` either globally or within the latency
section of your config:

```yaml
use_seasonality: false          # disable liquidity and spread multipliers

latency:
  use_seasonality: false        # disable latency multipliers
```

The same flag can be passed as a constructor argument to
`ExecutionSimulator` or `LatencyImpl`.

At runtime, the environment variable ``ENABLE_SEASONALITY`` can be set to
``0``/``false`` to disable all seasonality features regardless of the
configuration files. Leaving it unset or truthy keeps seasonality
enabled.

## Seeds and determinism

The latency model's random draws are controlled by the `seed` field in
the `latency` section of simulation configs. Hourly multipliers only
scale deterministic parameters and do not reseed or otherwise affect the
random number generator. Runs executed with the same `seed` and
multiplier set will therefore produce identical latency samples.

## Data storage and retention

Raw historical snapshots used to derive the multipliers are stored under
`data/seasonality_source/`. Each snapshot should be accompanied by a
`<filename>.sha256` file generated by the helper scripts to allow
auditing of the inputs.

Retain at least 12 months of snapshots so that previous seasonality runs
can be reproduced. Snapshot sizes depend on the exchange feed but
typically require several hundred megabytes per month; plan storage
accordingly.

## Automated updates

The helper script `scripts/cron_update_seasonality.sh` can be executed
periodically (e.g. via `cron`) to rebuild the multipliers from the latest
snapshot and commit the updated
`data/latency/liquidity_latency_seasonality.json`.

The script performs two comparisons against the previously committed
version:

* if the maximum absolute difference across all multipliers is below
  `SEASONALITY_THRESHOLD` (default `0.01`), the update is discarded;
* if the difference exceeds `SEASONALITY_MAX_DELTA` (default `0.5`), the
  run aborts for manual inspection.

Only changes that pass these checks are committed and pushed. The cron
job requires write access to the repository and credentials capable of
pushing to the remote. The executing user must also have read access to
the raw snapshot under `data/seasonality_source/`.

Example crontab entry (UTC):

```
5 3 * * 1 /path/to/repo/scripts/cron_update_seasonality.sh >> /var/log/seasonality.log 2>&1
```


## Operational checklist

When deploying new multipliers:

1. Compute and record the SHA256 hash of `liquidity_latency_seasonality.json`.
2. Store the hash in configuration fields such as `liquidity_seasonality_hash`
   and `latency.seasonality_hash`.
3. At runtime, the loader logs the hash and warns if it differs from the
   expected value.


## Performance

Microbenchmarks repeatedly invoking `ExecutionSimulator.set_market_snapshot`
for 100k synthetic ticks show that enabling hourly multipliers adds only a
small overhead:

```
$ python benchmarks/simulator_seasonality_bench.py
use_seasonality=False interpolate=False: 0.333s
use_seasonality=True interpolate=False: 0.849s
use_seasonality=True interpolate=True: 1.046s
```

Lookup now avoids `time.gmtime` and uses precomputed NumPy arrays for the
168 multipliers, reducing datetime conversions and keeping the cost of
seasonality modest. Enabling optional linear interpolation between adjacent
hours (``seasonality_interpolate=True``) performs two lookups and a
blend, which increases runtime but smooths transitions around hour
boundaries. Leave interpolation disabled if maximum performance is
required.

## Helper functions

``utils_time.py`` provides convenience helpers to fetch the appropriate
hour-of-week multiplier for a given UTC timestamp. Modules can import
``get_liquidity_multiplier`` or ``get_latency_multiplier`` instead of
manually indexing arrays:

```python
from utils_time import get_liquidity_multiplier, get_latency_multiplier
m_liq = get_liquidity_multiplier(ts_ms, liquidity_array, interpolate=True)
m_lat = get_latency_multiplier(ts_ms, latency_array, interpolate=True)
```

These helpers ensure consistent hour-of-week calculations and will be
reused across future modules. Pass ``interpolate=True`` to blend adjacent
hours using the minute offset.
