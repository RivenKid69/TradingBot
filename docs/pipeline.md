# Pipeline Overview

This document summarizes the stages of the decision-making pipeline, the
reasons a stage may drop an item, and the configuration flags that control
each part of the process.

## Order of stages

1. **CLOSED_BAR** – Ensure incoming bars are fully closed before processing.
2. **WINDOWS** – Skip timestamps falling into configured no-trade windows.
3. **ANOMALY** – Optional anomaly screens for returns or spread.
4. **EXTREME** – Optional guards for extreme volatility or spread.
5. **POLICY** – Generate candidate orders from policies and feature pipes.
6. **RISK** – Enforce account and position limits on candidate orders.
7. **PUBLISH** – Throttle, queue or emit orders to downstream services.

## Drop reasons

Stages can return a ``PipelineResult`` with one of the following reasons:

- ``INCOMPLETE_BAR``
- ``MAINTENANCE``
- ``WINDOW``
- ``ANOMALY_RET``
- ``ANOMALY_SPREAD``
- ``EXTREME_VOL``
- ``EXTREME_SPREAD``
- ``RISK_POSITION``
- ``OTHER``

## Configuration

The pipeline is configured with :class:`PipelineConfig`, which holds a global
``enabled`` flag and per-stage settings.  Each stage entry is a
:class:`PipelineStageConfig` containing its own ``enabled`` flag and optional
parameters passed to the stage function.

Example:

```python
PipelineConfig(
    enabled=True,
    stages={
        "closed_bar": PipelineStageConfig(enabled=True),
        "windows": PipelineStageConfig(enabled=True),
        "policy": PipelineStageConfig(enabled=True),
        "risk": PipelineStageConfig(enabled=True),
        "publish": PipelineStageConfig(enabled=True),
    },
)
```

By default, symbol lists come from ``data/universe/symbols.json`` and can be
overridden via CLI or ``data.symbols`` in the configuration.

Disabling a stage removes it from the processing chain.  Individual stage
parameters allow finer control, such as no-trade window definitions or risk
limits.
