# Changelog

## [Unreleased]

### Added
- **Seasonality Support**: Introduced hour-of-week seasonality multipliers to improve simulation fidelity.
  - **Required actions**:
    - Regenerate multipliers with the quick-start script.
    - Validate and update configurations before training or running simulations.
  - **Resources**:
    - [Seasonality overview](docs/seasonality.md)
    - [Quick start guide](docs/seasonality_quickstart.md)
    - [Process checklist](docs/seasonality_checklist.md)
    - [Example notebook](docs/seasonality_example.md)
    - [Migration guide](docs/seasonality_migration.md)
- **Dynamic spread builder**: Added `scripts/build_spread_seasonality.py` for generating
  hour-of-week spread profiles consumed by `slippage.dynamic_spread`. The script
  supports custom output paths, rolling windows and warns when the source
  snapshot exceeds the configured `refresh_warn_days` threshold.

### Deprecated
- `LatencyImpl.dump_latency_multipliers` and
  `LatencyImpl.load_latency_multipliers` have been replaced by
  `dump_multipliers` and `load_multipliers`. The old names continue to work but
  emit `DeprecationWarning`. See the migration guide for details.
