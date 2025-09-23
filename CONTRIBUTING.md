# Contributing

Thank you for considering contributing to TradingBot.

## Seasonality review requirements

Before submitting changes, evaluate how seasonality could impact the affected strategies and describe your findings in the pull request.
Contributions lacking a seasonality review may be delayed until this analysis is provided.

## Issue tracking and testing

Open an issue using the `Task` template for every piece of work. The template
requires explicit completion criteria along with links to code reviews and test
evidence. Provide this information before requesting a merge.

## Dataset artefact windows

Offline dataset splits are versioned contracts. When updating `configs/offline*.yml`
definitions or regenerating ADV, seasonality, or fee artefacts, ensure the
recorded data window does not extend beyond each split's end timestamp.

1. Regenerate the artefact metadata by running the relevant builder (for
   example `scripts/build_adv.py`) with the `--split` flag so the metadata block
   contains the actual window used during the refresh.
2. Execute `pytest tests/test_offline_split_windows.py` to confirm the
   regenerated metadata and offline configuration stay within the declared
   split boundaries.
3. If the check fails, adjust the artefact input window or split definition so
   the `data_window.actual.end` value never exceeds the split's `end`
   timestamp before submitting your changes.
