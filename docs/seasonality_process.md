# Seasonality Multiplier Review Process

This document describes the review and approval workflow for introducing new seasonality multipliers.

## Review steps
1. **Data preparation** – Recompute multipliers from the latest public exchange data and upload the artifacts with matching `.sha256` checksums.
2. **Validation run** – Execute the seasonality validation script against the proposed multipliers and source dataset (see `seasonality_QA.md`). Archive the command output and checksum files.
3. **Peer review** – Submit multipliers, validation metrics, and logs for developer review. Address any discrepancies before seeking approval.
4. **Approval** – Obtain data validation and QA sign-offs prior to merging into `main`.

## Data validation checklist
- [ ] Source dataset covers at least 12 months and uses UTC timestamps.
- [ ] Checksums exist for every dataset and multiplier file.
- [ ] Validation script executed with the agreed `--threshold`.
- [ ] All `max_rel_diff` values are at or below the threshold.
- [ ] Validation output and checksums archived with the PR.

**Data Validation Sign-off:** _____________________

## QA sign-off checklist
- [ ] QA reviewed validation metrics and confirmed acceptance criteria.
- [ ] QA reproduced validation run on an independent machine if required.
- [ ] Archived logs and checksum files stored for audit.
- [ ] QA approved multipliers and documented findings.

**QA Sign-off:** _________________________________
