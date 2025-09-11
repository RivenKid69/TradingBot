# Seasonality Data Policy

This repository uses aggregated exchange data to compute seasonality multipliers.
The following rules apply to any data stored under `data/seasonality_source`:

## Data Handling
- Only market data derived from public sources may be stored.
- Files must never contain personal identifying information (PII).
- Retain at least 12 months of history for auditability.
- Each file requires a matching `.sha256` checksum.

## Usage Limitations
- Seasonality data is for internal analytics and model training only.
- Redistribution outside the organization is prohibited.
- When sharing derived metrics, ensure that no raw snapshots are exposed.
- Run `scripts/check_pii.py` before committing new snapshots to verify that no
  PII patterns exist.

## Licensing and Attribution
- Historical snapshots are typically sourced from public exchange APIs (for
  example, Binance) and remain subject to their respective terms of use. See the
  [Binance API Terms of Use](https://www.binance.com/en/legal/api-terms-of-use)
  for a reference.
- The data may be used internally for analytics and model training but must not
  be redistributed or sold.
- When sharing derived metrics, include attribution such as "Based on data from
  Binance. © Binance. All rights reserved." and ensure any additional provider
  requirements are met.

Adhering to this policy helps maintain compliance and data privacy.
