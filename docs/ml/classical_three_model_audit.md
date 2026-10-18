# Three-model classical ML audit

## Scope and decision

The approved classical candidates are now Logistic Regression, calibrated Linear
SVM, and Random Forest. All three use the same source-masked word TF-IDF input,
dataset/class-balanced sample weights, LIAR clear endpoints at 0.25 dataset
weight, and the same source-held-out validation protocol. Label `1` means fake
and label `0` means real.

Logistic Regression remains the preferred classical model. Random Forest is
retained as a comparison and possible diversity signal, not promoted as the
primary classifier. None of these content models verifies facts.

## Leakage and source-shortcut audit

The prepared 107,292-record corpus has zero exact cross-split fingerprints and
zero configured near-duplicate cross-split clusters. There are 261 within-split
near-duplicate clusters containing 525 records; none leaks a test record into
development under the configured detector.

Publisher metadata remains a major shortcut. A metadata-only lookup reaches
macro-F1 0.9834 on IFND and 1.0000 on ISOT. The model text masks known source
names and URLs, but this cannot remove every collection-specific writing and
editorial cue. This is why source-held validation, rather than only random
within-source evaluation, controls model and threshold selection.

## Source-held validation and threshold selection

Thresholds were selected from dataset-held-out validation folds only. Neither
test splits nor the contemporary diagnostic influenced selection.

| Model | Threshold | Mean dataset macro-F1 | Worst dataset macro-F1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.49 | **0.5805** | **0.4283** |
| Calibrated Linear SVM | 0.71 | 0.5338 | 0.2295 |
| Random Forest | 0.37 | 0.5195 | 0.2523 |

The low values relative to ordinary held-out test scores demonstrate that unseen
collection transfer is still the central weakness. Logistic Regression wins on
both the primary mean and worst-dataset criteria.

## Random Forest validation-only tuning

The initial forest uses 200 trees, Gini splits, square-root feature sampling,
minimum leaf size 2, no depth cap, and all available CPU cores. Four alternatives
were audited without consulting test data.

| Candidate | Threshold | Mean macro-F1 | Worst macro-F1 | Decision |
| --- | ---: | ---: | ---: | --- |
| 200 trees, sqrt, leaf 2, Gini | 0.37 | **0.5195** | 0.2523 | Selected |
| 120 trees, sqrt, leaf 1, Gini | 0.37 | 0.5082 | 0.2466 | Reject |
| 120 trees, sqrt, leaf 2, log-loss | 0.36 | 0.5114 | **0.2582** | Reject: lower primary mean |
| 120 trees, 1% features, leaf 2, log-loss | 0.33 | 0.4878 | 0.2449 | Reject |
| 120 trees, sqrt, leaf 2, Gini, framing masked | 0.38 | 0.4826 | 0.2507 | Reject |

The log-loss candidate slightly improves the worst fold but does not exceed the
primary mean score. The original forest is therefore retained rather than
claiming an unsupported improvement.

## Held-out test performance by dataset

These splits were untouched during v2 model and threshold selection, although
they had already been inspected during earlier baseline work and must not be
treated as pristine for future iterations.

| Dataset | Model | Macro-F1 | ROC-AUC | Brier score |
| --- | --- | ---: | ---: | ---: |
| BharatGather | Logistic Regression | 0.9049 | 0.9716 | 0.0684 |
| BharatGather | Linear SVM | 0.9022 | 0.9429 | 0.0956 |
| BharatGather | Random Forest | 0.7569 | 0.9648 | 0.1171 |
| IFND | Logistic Regression | 0.9440 | 0.9825 | 0.0448 |
| IFND | Linear SVM | 0.9411 | 0.9853 | 0.0427 |
| IFND | Random Forest | 0.9281 | 0.9866 | 0.0530 |
| Indian Fake News | Logistic Regression | 0.9514 | 0.9976 | 0.0177 |
| Indian Fake News | Linear SVM | 0.9534 | 0.9972 | 0.0255 |
| Indian Fake News | Random Forest | 0.8905 | 0.9986 | 0.0390 |
| ISOT | Logistic Regression | 0.9842 | 0.9986 | 0.0158 |
| ISOT | Linear SVM | 0.9630 | 0.9985 | 0.0234 |
| ISOT | Random Forest | 0.9819 | 0.9989 | 0.0459 |

Mean per-dataset test macro-F1 is 0.9461 for Logistic Regression, 0.9399 for
Linear SVM, and 0.8894 for Random Forest. The corresponding worst-dataset values
are 0.9049, 0.9022, and 0.7569.

Random Forest's threshold emphasizes fake recall. Its fake recall is 0.9920 on
BharatGather, 0.9645 on IFND, 1.0000 on Indian Fake News, and 0.9939 on ISOT.
The tradeoff is especially severe on BharatGather, where real recall falls to
0.5944.

## Misclassification audit

| Model | Total errors | False positives | False negatives | Errors with margin >= 0.4 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | **592** | 325 | 267 | 50 |
| Linear SVM | 679 | 125 | 554 | 122 |
| Random Forest | 1,105 | 1,022 | **83** | **19** |

The forest misses relatively few fake-labelled records but incorrectly flags
many real-labelled records. Its most important features include fact-check and
editorial framing terms such as `fact check`, `fake`, `viral`, `video`, and
`image`. These are learned dataset associations, not evidence that an article
is false. Masking the framing phrases did not improve the validation objective.

## Contemporary 2026 diagnostic

The frozen diagnostic contains 20 short items (10 official-source authentic
items and 10 independently debunked items). It is small and hand-curated and is
not a benchmark.

| Model | Accuracy | Balanced accuracy | Macro-F1 | Confusion matrix `[[TN,FP],[FN,TP]]` |
| --- | ---: | ---: | ---: | --- |
| Logistic Regression | **0.75** | **0.75** | **0.7333** | `[[10,0],[5,5]]` |
| Linear SVM | 0.70 | 0.70 | 0.6703 | `[[10,0],[6,4]]` |
| Random Forest | 0.70 | 0.70 | 0.7000 | `[[7,3],[3,7]]` |

Random Forest catches more debunked items than either linear model but creates
three false alarms. That diversity may later be useful to a validated fusion
layer, but it is not evidence that a majority vote or ensemble is better; such
a decision requires a separately authorized and leakage-safe experiment.

## Reproducible outputs

Generated artifacts remain ignored by Git under
`models/classical/multidataset_v2_robust/`:

- `benchmark.json` — configurations, thresholds, validation, test metrics, and hashes;
- `random_forest_tuning.json` — validation-only configuration audit;
- `errors/*_test_errors.csv` and `errors/summary.json` — model-wise error exports;
- one `pipeline.joblib` per model;
- `models/classical/contemporary_2026_evaluation.json` — frozen diagnostic results.

The Colab notebook reproduces training, the forest tuning audit, all three model
comparisons, contemporary evaluation, and error export on a CPU runtime.
