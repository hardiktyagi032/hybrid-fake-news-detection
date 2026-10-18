# Classical ML experiment results

## Run record

- Run date: 2026-10-18
- Training dataset: ISOT
- Development records: 34,946 (train plus validation for final refit)
- Held-out ISOT test records: 3,883
- Models: logistic regression and calibrated linear SVM only
- Features: word TF-IDF, 1-2 grams, maximum 75,000 features
- Random seed: `20261017`
- Environment: Python 3.14.5, pandas 3.0.6, scikit-learn 1.9.1

## Validation comparison

| Model | Accuracy | Macro-F1 | Fake precision | Fake recall | ROC-AUC | Brier score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.9910 | 0.9909 | 0.9955 | 0.9849 | 0.9989 | 0.0107 |
| Linear SVM (calibrated) | **0.9936** | **0.9935** | **0.9966** | **0.9894** | **0.9993** | **0.0051** |

Linear SVM is marked as the preferred in-domain baseline because selection was
predeclared as validation macro-F1, followed by ROC-AUC as a tie-breaker.

## Held-out ISOT test

| Model | Accuracy | Macro-F1 | Fake precision | Fake recall | ROC-AUC | Brier score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.9910 | 0.9909 | 0.9955 | 0.9849 | 0.9991 | 0.0097 |
| Linear SVM (calibrated) | **0.9956** | **0.9956** | **0.9983** | **0.9922** | **0.9993** | **0.0036** |

Confusion matrices use rows as actual `[real, fake]` and columns as predicted
`[real, fake]`:

- Logistic regression: `[[2085, 8], [27, 1763]]`
- Linear SVM: `[[2090, 3], [14, 1776]]`

## Frozen-model cross-dataset evaluation

The same ISOT-trained pipelines were evaluated without retraining on test splits
from three other datasets whose binary label mapping was verified in Phase 4.

| Target dataset | Model | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| IFND | Logistic regression | 0.4371 | 0.5756 | 0.4168 | 0.8473 |
| IFND | Linear SVM | 0.3904 | 0.5430 | 0.3538 | 0.8097 |
| Indian Fake News | Logistic regression | 0.8969 | 0.7078 | 0.7615 | 0.8855 |
| Indian Fake News | Linear SVM | 0.8924 | 0.6947 | 0.7472 | 0.8713 |
| BharatGather | Logistic regression | 0.4478 | 0.5122 | 0.4048 | 0.4568 |
| BharatGather | Linear SVM | 0.4402 | 0.5131 | 0.3778 | 0.4796 |

## Interpretation

The near-perfect ISOT result does not generalize across collections. Both models
strongly over-predict the fake class on IFND and BharatGather, and BharatGather
ROC-AUC is near or below random ranking. This is evidence that the ISOT score is
substantially influenced by dataset-specific source, topic, genre, and writing
style cues. It must not be presented as real-world fake-news detection accuracy.

Linear SVM remains the preferred ISOT baseline under the declared selection rule,
but no model is selected yet for production or fusion. Logistic regression showed
better transfer on IFND and Indian Fake News. The next classical-model research
step should define a multi-dataset training and external-validation protocol before
any final selection or hyperparameter search.

## Local artifacts

Artifacts are reproducible but intentionally ignored by Git:

- Logistic regression pipeline SHA-256:
  `2a787c56c1d35a3e0da4aec905dc255b1644468dcd603bfab54f7fa32468c132`
- Linear SVM pipeline SHA-256:
  `aa505ec9f89690d3042d365bcdcbacd4378654760f1cfdd8730f132bbad06a7c`
- Full local reports: `models/classical/isot/benchmark.json` and
  `models/classical/isot/external_evaluation.json`
