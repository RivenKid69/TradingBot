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

### Deprecated
- `LatencyImpl.dump_latency_multipliers` and
  `LatencyImpl.load_latency_multipliers` have been replaced by
  `dump_multipliers` and `load_multipliers`. The old names continue to work but
  emit `DeprecationWarning`. See the migration guide for details.
