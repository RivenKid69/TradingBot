# Seasonality Checklist

This checklist documents responsibilities when implementing or updating seasonality features.

## Data Engineering
- [ ] Verify raw market data sources are complete and use UTC timestamps.
- [ ] Recompute seasonality metrics and upload the results to shared storage.
- [ ] Publish checksums for data artifacts and notify stakeholders of updates.

## Development
- [ ] Update configuration files to reference the latest seasonality datasets.
- [ ] Integrate seasonality multipliers into feature or model pipelines.
- [ ] Add or adjust unit tests covering seasonality edge cases.

## QA
- [ ] Execute the seasonality validation script against the latest dataset.
- [ ] Review metrics and confirm they meet acceptance thresholds.
- [ ] Record validation output and archive artifacts for audit.

---

**Data Engineering Sign-off:** _____________________

**Developer Sign-off:** ___________________________

**QA Sign-off:** _________________________________

