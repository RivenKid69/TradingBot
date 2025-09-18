# Seasonality QA Guide

This guide describes how the QA team should validate hourly seasonality multipliers against historical data.

A reference template with all multipliers equal to `1.0` is available at `configs/liquidity_latency_seasonality.sample.json`.

## Running the validation script

1. Prepare a CSV or Parquet file with historical trades. The file must include a timestamp column (`ts_ms` or `ts`) expressed in UTC. Feeding local-time timestamps into the validator can misalign hour-of-week multipliers due to DST, so convert any source data to UTC first.
2. Run the validation script with paths to the historical dataset and the multipliers JSON:

```bash
python scripts/validate_seasonality.py \
  --historical data/seasonality_source/latest.parquet \
  --multipliers data/latency/liquidity_latency_seasonality.json \
  --threshold 0.1
```

- Use `--symbol` if the multipliers file contains multiple instruments.
- The script writes a `<file>.sha256` checksum next to the historical dataset for audit purposes.

## Interpreting metrics

For each available metric (`liquidity`, `spread_bps`, `latency_ms`) the script prints two values:

- `max_rel_diff` – maximum relative difference across all 168 hours of the week.
- `mean_rel_diff` – average relative difference across the same period.

A final line indicates overall status:

- `✅ Seasonality validation passed` – all metrics are within the allowed threshold.
- `❌ Seasonality validation failed` – at least one metric exceeded the threshold.

## Acceptance criteria

- `max_rel_diff` for every reported metric must stay at or below the threshold provided via `--threshold` (default `0.1`, i.e. 10%).
- `mean_rel_diff` is informational but should remain well below the threshold (recommended `< 0.05` when using the default).

Record the command output and checksum file in your validation report.
