# Task: Add issue template requiring completion criteria and evidence

## Completion Criteria
- Issue template created with mandatory fields for completion criteria, code review evidence, and test evidence.
- Contributing guide instructs developers to use the new template.

## Code Review Evidence
- Self-review of documentation changes.

## Test Evidence
- `pytest tests/test_pii_detection.py -q` (fails: ModuleNotFoundError: utils_time)
