# Fee schedule data

This directory contains JSON descriptions of maker/taker fee schedules per trading symbol.
Runtime components can now rebuild these tables automatically when the bundled
snapshot is missing or outdated – see *Runtime auto-refresh* below for details.

## Update process

1. Run the automated refresher to fetch the latest fee schedule:

   ```bash
   python scripts/refresh_fees.py --dry-run
   ```

   The script relies on :mod:`binance_fee_refresh` to fetch the active spot
   universe from `exchangeInfo`, retrieve maker/taker basis points via Binance
   public APIs (or an authenticated CSV/`tradeFee` call when available), and
   reconstruct `fees_by_symbol.json`.
   When the existing file is newer than 30 days the script prints a reminder,
   otherwise it highlights that a refresh is overdue.

2. After reviewing the dry-run output, write the refreshed table:

   ```bash
   python scripts/refresh_fees.py --vip-tier "VIP 0" --out data/fees/fees_by_symbol.json
   ```

   - Provide `--csv path/to/export.csv` if Binance publishes an up-to-date CSV
     snapshot locally.
   - Set `BINANCE_API_KEY`/`BINANCE_API_SECRET` (or pass `--api-key` /
     `--api-secret`) to use the private `tradeFee` endpoint instead of the
     public snapshot.
   - Adjust `--bnb-discount-rate` when Binance changes the BNB fee discount
     factor (defaults to 25%).

## Runtime auto-refresh

`FeesImpl` will attempt to refresh `data/fees/fees_by_symbol.json` on the fly
whenever the configured table is missing, invalid or older than the public
snapshot threshold (30 days by default). The helper only relies on public
endpoints (or an optional cached CSV specified via the
`BINANCE_FEE_SNAPSHOT_CSV` environment variable) so it works in the
signal-only deployment path.

During an auto-refresh the helper records the inferred baseline maker/taker
fees, BNB discount multipliers and VIP tier. These values are exposed via the
`FeesConfig.metadata.table.public_refresh` block and are used as defaults for
the simulator when explicit overrides are absent. To disable the network call,
set `BINANCE_PUBLIC_FEES_DISABLE_AUTO=1` or provide bespoke overrides in the
configuration (`maker_bps`, `taker_bps`, `use_bnb_discount`, etc.).

Additional knobs recognised during auto-refresh:

- `BINANCE_PUBLIC_FEE_URL` – override the public fee endpoint.
- `BINANCE_FEE_TIMEOUT` – timeout (seconds) for public requests (defaults to 30).
- `BINANCE_BNB_DISCOUNT_RATE` – fallback BNB discount rate used to populate
  discount multipliers when Binance omits them.
- `BINANCE_API_KEY`/`BINANCE_API_SECRET` – optional keys for the private
  `tradeFee` endpoint if available.

3. Validate the JSON formatting before committing:

   ```bash
   python -m json.tool data/fees/fees_by_symbol.json > /dev/null
   ```

4. Document any non-obvious assumptions in the commit message or within this README to aid future maintenance.

