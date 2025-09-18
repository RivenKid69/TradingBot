# Latency & Liquidity Seasonality Artifacts

This directory stores generated seasonality multipliers and related backups.

* `liquidity_latency_seasonality.json` – default location for the hourly
  liquidity/latency multipliers built by `scripts/build_hourly_seasonality.py`.
* `backups/` – timestamped copies of the previous JSON created whenever the
  builder overwrites the file. The directory is created on demand by the
  script and is not tracked in Git.

Do not commit large generated files unless explicitly required. Use the
builder script to refresh the JSON and rely on automated backups to retain
previous versions.
