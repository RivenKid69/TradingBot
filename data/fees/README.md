# Fee schedule data

This directory contains JSON descriptions of maker/taker fee schedules per trading symbol.

## Update process

1. Run the automated refresher to fetch the latest fee schedule:

   ```bash
   python scripts/refresh_fees.py --dry-run
   ```

   The script fetches the active spot universe from `exchangeInfo`, retrieves
   maker/taker basis points via Binance public APIs (or an authenticated CSV or
   `tradeFee` call when available), and reconstructs `fees_by_symbol.json`.
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

3. Validate the JSON formatting before committing:

   ```bash
   python -m json.tool data/fees/fees_by_symbol.json > /dev/null
   ```

4. Document any non-obvious assumptions in the commit message or within this README to aid future maintenance.

