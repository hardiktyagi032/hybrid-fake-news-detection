# ADR 0004: Integrate the Veritas React frontend

## Status

Accepted.

## Decision

Integrate the collaborator frontend from
`https://github.com/AarushVerma07/frontend-fake-news` under `frontend/` using a
Git subtree merge. Preserve the source repository's commit history and author
attribution instead of copying only its current files.

Add a Model audit page backed by a versioned, static metrics snapshot. The page
shows source-held validation, held-out dataset results, probability thresholds,
calibration metrics, the contemporary diagnostic, error distributions, and the
Random Forest tuning comparison. It must prominently state that the models are
content classifiers rather than evidence-based fact verifiers.

## Boundaries

- Existing fact-checking screens continue to use mock data because production
  backend routes have not been implemented.
- Model files, datasets, and private local artifacts are not bundled into the
  browser application.
- Training metrics are static and reproducible from the documented audit. A
  future backend may expose versioned experiment JSON after its API contract is
  designed.
- No license file was present in the source repository at integration time.
  The integration was performed at the repository owner's direction, with the
  original Git attribution retained. Licensing should be formalized before
  wider redistribution.

## Consequences

The project has a usable frontend and can demonstrate the classical-model audit
without requiring a running ML service. Static results cannot update when a new
training run completes, so new audited metrics must be reviewed and committed
until a safe experiment-reporting API is introduced.
