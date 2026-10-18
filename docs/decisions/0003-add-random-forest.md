# ADR 0003: Add Random Forest without replacing the linear baseline

## Status

Accepted.

## Decision

Add Random Forest as the third approved classical candidate. Train and evaluate
it under the same leakage-aware, dataset-balanced, source-held-out protocol as
Logistic Regression and calibrated Linear SVM. Select its threshold on validation
folds only and preserve it for model-wise comparison and potential future fusion.

Keep Logistic Regression as the preferred standalone classical model because it
wins the declared source-held mean and worst-dataset macro-F1 criteria. Do not
promote Random Forest from its stronger fake recall or selected in-domain scores.

## Consequences

The project gains a nonlinear comparison model and a useful error-diversity
signal. Training takes longer on CPU, artifacts are larger, and feature
importance is less directional than linear coefficients. The audit also shows a
large Random Forest false-positive burden and continued sensitivity to editorial
framing. None of the three models is a fact verifier.
