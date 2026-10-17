# Data policy

`raw/` holds the unchanged user-supplied source archives and extracted copies. `interim/` holds the aggregate inspection report and a row-level exclusion log. `processed/` holds documented cleaned data, deterministic splits, and a manifest.

All contents are ignored by Git except directory markers. The original source must be hashed before processing and must never be overwritten.

Phase 4 local layout:

```text
raw/_archives/      immutable downloaded archives/files
raw/sources/        extracted working copies
interim/            inspection.json and exclusions.csv
processed/<name>/   cleaned.csv and split files
processed/manifest.json
```

Reproduce and validate with `scripts/prepare_datasets.py` and `scripts/validate_prepared_data.py`. Binary label fields are intentionally blank when source documentation does not establish their meaning.
