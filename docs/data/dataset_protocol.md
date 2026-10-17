# Dataset inspection and preparation protocol

Status: executed on the eight user-supplied sources received on 2026-10-17. No substitute dataset was selected. See `dataset_preparation_report.md` for actual results.

## 1. Preserve and identify

1. Copy the source into `data/raw/` without changing its contents.
2. Record filename, byte size, SHA-256, source/provenance, license, and receipt date.
3. Keep the raw file read-only in practice and never overwrite it.
4. Confirm format, encoding, compression, and whether the file contains predefined splits.

## 2. Profile the schema

- Count records and columns.
- Identify candidate text, title, label, URL, source, author, date, event, language, and media fields.
- Measure missingness, unique counts, class distribution, text-length distribution, and invalid values.
- Confirm label semantics with the supplied documentation. Never infer that `0` or `1` means fake without evidence.

## 3. Analyze quality and leakage

- Find byte-identical and row-identical records.
- Detect normalized exact text duplicates and conflicting labels.
- Assess near duplicates using a documented threshold after inspecting the language and text length.
- Check title/body copies, syndicated stories, template boilerplate, source shortcuts, and label artifacts.
- If predefined splits exist, test duplicate, event, author, source, and timestamp overlap across them.
- Check whether fact-check verdicts or post-publication evidence leak into model inputs.
- Distinguish missing text from intentionally short claims.

## 4. Clean conservatively

- Preserve original text and add cleaned fields rather than replacing it.
- Normalize encoding and whitespace only when justified.
- Remove markup or extraction artifacts with a reversible rule.
- Do not blindly remove URLs, punctuation, stop words, numbers, named entities, or casing. They may contain useful or leakage-prone information and require model-specific experiments.
- Quarantine corrupted records and ambiguous labels; document exclusions with reasons.
- Resolve label mapping only from authoritative dataset documentation.

## 5. Split reproducibly

- Define a fixed random seed in the generated manifest.
- Prefer group-aware splits for duplicates and events.
- Use temporal evaluation when timestamps support the deployment scenario.
- Consider holding out sources to measure cross-source generalization.
- Stratify only after grouping and only when it does not create leakage.
- Fit vocabulary, normalization statistics, calibration, and resampling on training data only.

## 6. Produce outputs

Expected Phase 4 outputs after receipt:

- `data/interim/inspection.json`
- `data/interim/exclusions.csv` or equivalent decision log
- `data/processed/cleaned.*`
- `data/processed/train.*`, `validation.*`, and `test.*` when appropriate
- `data/processed/manifest.json` with hashes, schema, label map, seed, and transformation versions
- a dataset-specific report under `docs/data/`

## 7. Verification

- Re-hash the raw source and confirm it is unchanged.
- Reload every produced file.
- Assert expected columns and allowed labels.
- Assert split disjointness under exact and selected near-duplicate keys.
- Report class and source distributions by split.
- Re-run preparation and compare hashes or deterministic record identifiers.

Synthetic fixtures may test the tooling but never replace or represent the real dataset.
