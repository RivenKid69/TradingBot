# Symbol universe refresh

The `services.universe` module maintains a cached list of Binance spot
symbols trading against USDT.  When imported, :func:`get_symbols` checks
``data/universe/symbols.json`` and refreshes it if the file is missing or
older than 24 hours.  The refresh can also be triggered explicitly by
running the module as a CLI.

## CLI usage

```bash
python -m services.universe \
    --output data/universe/symbols.json \
    --liquidity-threshold 1e6
```

* ``--output`` – destination JSON file.  The directory is created if needed.
* ``--liquidity-threshold`` – minimum 24‑hour quote volume in USDT.  Set to
  ``0`` to bypass the liquidity filter and include all trading pairs.
* ``--force`` – refresh even if the cache is still fresh.

## Refresh schedule

The default time‑to‑live for the cache is 24 hours.  The module refreshes
on the first import if the file is stale.  For deterministic updates,
install a daily cron job:

```
0 3 * * * cd /path/to/repo && /usr/bin/python -m services.universe
```

This example refreshes the symbol list every day at 03:00 UTC.

## Custom symbol lists

To use a custom universe, generate a file with the CLI and point runners to
it via the ``--symbols`` flag or the ``data.symbols`` field in YAML config.
You can also maintain the JSON manually and run the CLI with
``--liquidity-threshold 0`` to store the latest exchange symbols without
filtering by volume.
