# Fee schedule data

This directory contains JSON descriptions of maker/taker fee schedules per trading symbol.

## Update process

1. Pull the latest fee schedule from the exchange's official documentation or API.
2. Update `fees_by_symbol.json`:
   - Adjust the entries under `fees` to match the new maker/taker basis points (bps) for every affected symbol.
   - Add or remove symbols as they appear/disappear in the public schedule.
   - Include `fee_rounding_step_bps` when the exchange specifies a minimum rounding increment for fee calculations.
   - Set `bnb_discount_bps` (or a similar discount field) only when the exchange confirms the discount level for the symbol.
3. Refresh the metadata block:
   - `built_at`: UTC timestamp in ISO-8601 format reflecting when the data was generated.
   - `source`: Plain-text description of the data source (URL or document title) and retrieval date.
   - `vip_tier`: The VIP tier the data corresponds to (e.g., "VIP 0").
   - `schema_version`: Increment if the JSON shape changes.
4. Validate the JSON formatting before committing:

   ```bash
   python -m json.tool data/fees/fees_by_symbol.json > /dev/null
   ```

5. Document any non-obvious assumptions in the commit message or within this README to aid future maintenance.

