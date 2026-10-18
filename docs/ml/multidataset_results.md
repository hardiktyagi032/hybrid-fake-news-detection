# Dataset-balanced classical ML results

## Purpose

The ISOT-only models achieved near-perfect in-dataset scores but transferred
poorly to IFND and BharatGather. This experiment keeps the approved algorithms—
Logistic Regression and calibrated Linear SVM—while changing the training protocol
to reduce dataset-specific shortcuts.

## Protocol

- Datasets: ISOT, IFND, Indian Fake News, and BharatGather.
- Only Phase 4 datasets with verified binary-label semantics are used.
- Training records: 84,479.
- Validation records: 10,456.
- Final development refit records: 94,935.
- Held-out test records: 12,357.
- Every dataset/class pair receives equal total sample weight.
- Selection metric: mean per-dataset validation macro-F1, followed by the
  worst-dataset validation macro-F1.
- Features: TF-IDF word unigrams and bigrams, maximum 100,000 features.
- Random seed: `20261017`.

## Validation selection

| Model | Mean dataset macro-F1 | Worst dataset macro-F1 | Pooled accuracy | Pooled ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | **0.9479** | **0.8975** | **0.9633** | **0.9907** |
| Linear SVM | 0.9423 | 0.8881 | 0.9610 | 0.9890 |

Logistic Regression is selected under the declared validation rule. Both models
are retained for model-wise analysis.

## Held-out test results

| Dataset | Model | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC | Brier score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BharatGather | Logistic Regression | 0.9101 | 0.9124 | 0.9081 | 0.9722 | 0.0677 |
| BharatGather | Linear SVM | 0.9123 | 0.9168 | 0.9106 | 0.9414 | 0.0958 |
| IFND | Logistic Regression | 0.9575 | 0.9474 | 0.9510 | 0.9848 | 0.0406 |
| IFND | Linear SVM | 0.9660 | 0.9578 | 0.9609 | 0.9885 | 0.0349 |
| Indian Fake News | Logistic Regression | 0.9776 | 0.9551 | 0.9599 | 0.9963 | 0.0186 |
| Indian Fake News | Linear SVM | 0.9686 | 0.9288 | 0.9427 | 0.9962 | 0.0254 |
| ISOT | Logistic Regression | 0.9840 | 0.9832 | 0.9839 | 0.9985 | 0.0159 |
| ISOT | Linear SVM | 0.9817 | 0.9811 | 0.9816 | 0.9988 | 0.0209 |

Across held-out datasets, mean macro-F1 is 0.9507 for Logistic Regression and
0.9489 for Linear SVM. The worst dataset macro-F1 is 0.9081 and 0.9106,
respectively.

## Improvement over the ISOT-only models

| Target | Logistic Regression macro-F1 | Improved | Linear SVM macro-F1 | Improved |
| --- | ---: | ---: | ---: | ---: |
| BharatGather | 0.4048 | **0.9081** | 0.3778 | **0.9106** |
| IFND | 0.4168 | **0.9510** | 0.3538 | **0.9609** |
| Indian Fake News | 0.7615 | **0.9599** | 0.7472 | **0.9427** |

ISOT macro-F1 decreases modestly from 0.9909 to 0.9839 for Logistic Regression
and from 0.9956 to 0.9816 for Linear SVM. This tradeoff is desirable: a small
reduction on one source collection accompanies a large increase in cross-dataset
robustness.

The generic audited-report example that both ISOT-only models labeled fake is now
labeled real by both improved models:

- Logistic Regression: `P(fake) = 0.1505`.
- Linear SVM: `P(fake) = 0.1893`.

## Interpretation and remaining limits

The improved results demonstrate that dataset-balanced multi-source training is
substantially more robust than ISOT-only training. They do not prove factual
verification ability. All four collections can still contain source, time, topic,
translation, and annotation artifacts. The next evaluation should use a truly
unseen source or later time period and should include systematic error analysis.

Generated pipelines and the full JSON report remain local and ignored by Git under
`models/classical/multidataset_v1/`.
