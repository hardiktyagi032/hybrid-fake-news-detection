# ADR 0002: Start with an auditable TF-IDF classical baseline

## Status

Accepted after the user explicitly authorized post-Phase-4 classical ML work.

## Decision

Use the verified ISOT binary labels and existing leakage-aware splits for the first
benchmark. Compare only logistic regression and a calibrated linear SVM using
TF-IDF word features. Select on validation macro-F1 and evaluate the selected model
once on the held-out test split.

Do not use datasets with unresolved binary-label semantics. Do not treat the
classifier as a fact verifier or its probability as factual certainty.

## Consequences

The stage supplies a reproducible `P(fake)` content signal and a transparent
baseline. It does not address cross-source generalization, temporal drift, factual
evidence, claim verification, or the remaining fusion features. Those limitations
must be tested in later explicitly authorized stages.
