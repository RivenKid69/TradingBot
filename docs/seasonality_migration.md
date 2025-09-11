# Seasonality Multipliers Migration

Older versions of the codebase stored hourly seasonality multipliers as a
single array under a top-level `multipliers` key or even as a bare JSON list.
Current loaders expect mappings such as `{"liquidity": [...], "latency": [...]}`.

## Converting legacy files

Use `scripts/convert_multipliers.py` to rewrite old files into the new format:

```bash
python scripts/convert_multipliers.py old.json new.json --key liquidity
```

The script reads a legacy file and writes a new JSON mapping with the provided
key (default: `liquidity`). Adjust `--key` if the multipliers should be stored
under a different name, for example `--key latency`.

## Loader requirements

The helper functions in `utils_time.py` (`load_hourly_seasonality` and
`load_seasonality`) no longer support legacy structures. Convert old files with
the script above before loading them to avoid errors.
