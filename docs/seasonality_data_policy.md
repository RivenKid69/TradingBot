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

Adhering to this policy helps maintain compliance and data privacy.
