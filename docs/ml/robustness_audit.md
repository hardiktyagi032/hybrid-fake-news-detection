# Classical ML robustness audit and v2 results

## Scope

This audit examines the approved Logistic Regression and calibrated Linear SVM
classifiers. It does not add a transformer, neural model, or evidence verifier.
The audit covers exact and near-duplicate leakage, publisher/source shortcuts,
per-dataset and per-class behavior, confident errors, threshold sensitivity,
dataset-held-out transfer, and a small contemporary diagnostic set.

## Leakage results

The global audit covered 107,292 prepared records from ISOT, IFND, Indian Fake
News, and BharatGather.

- Cross-split normalized exact duplicate groups: **0**.
- Test records with an exact duplicate in development: **0**.
- Cross-split near-duplicate clusters: **0**.
- Test records near-duplicated in development: **0**.
- Global near-duplicate clusters found and safely contained within a split: 261
  clusters containing 525 records.

The preparation pipeline is therefore doing its intended job for exact and
SimHash-based near duplicates. This result does not exclude semantic paraphrases
that fall outside the configured SimHash distance.

## Publisher and collection shortcuts

A deliberately simple metadata-only majority lookup was evaluated on the test
splits. Publisher/source metadata is not supplied to the text model, but this
experiment reveals how strongly labels align with collection provenance.

| Dataset | Metadata-only macro-F1 | Test source values seen in train |
| --- | ---: | ---: |
| BharatGather | 0.5477 | 100% |
| IFND | 0.9834 | 100% |
| ISOT | 1.0000 | 100% |
| Indian Fake News | Not evaluable | No usable source grouping |

ISOT source categories perfectly encode its labels. IFND is nearly as severe.
The selected v1 Logistic Regression also assigned its largest-magnitude real
coefficient to `reuters`. The high random-within-dataset scores therefore include
collection recognition and must not be interpreted as general fact detection.

## Errors and threshold sensitivity

The v1 train-only threshold audit selected `0.53` for Logistic Regression and
`0.40` for Linear SVM. At those thresholds, the candidates made 561 and 541
errors, respectively, on the combined test splits; 48 Logistic Regression and 33
Linear SVM errors had confidence of at least 0.90.

Common confident errors included:

- authentic fact-check articles classified as fake because they repeatedly
  describe a false viral claim;
- fabricated claims written in ordinary news style classified as real;
- short augmented IFND headlines whose labels are not recoverable from wording;
- Reuters-like reporting style treated as a strong real-news signal.

These examples show a task mismatch: the four binary labels describe a mixture
of article authenticity, claim truthfulness, fact-versus-claim pairs, and
synthetic variants.

## Dataset-held-out baseline

For each fold, a model was trained on three datasets and evaluated on the fourth
dataset's test split. This is harsher and more realistic than random splitting.

| Model | Mean held-out-dataset macro-F1 | Worst held-out macro-F1 |
| --- | ---: | ---: |
| Logistic Regression v1 | 0.5374 | 0.4301 |
| Linear SVM v1 | 0.5391 | 0.4314 |

The gap from approximately 0.95 in-domain macro-F1 is evidence of weak source
transfer, not duplicate leakage.

## Representation experiments

Candidates were selected using only dataset-held-out validation folds. The test
splits were not used for feature selection.

| Logistic Regression representation | Mean macro-F1 | Worst macro-F1 |
| --- | ---: | ---: |
| Word TF-IDF, unmasked | 0.5402 | 0.4280 |
| Word TF-IDF, source masked | 0.5439 | **0.4335** |
| Source and fact-check framing masked | 0.5089 | 0.4257 |
| Word + character TF-IDF, source masked | 0.5277 | 0.3766 |
| Source masked + full LIAR endpoint weight | 0.5638 | 0.4109 |
| Source masked + 50% LIAR endpoint weight | 0.5764 | 0.4261 |
| Source masked + 25% LIAR endpoint weight | **0.5794** | 0.4267 |

Blanket removal of fact-check language and the character representation were
rejected. LIAR is used only as a model-training view: `true` maps to real and
`false`/`pants-fire` map to fake; `mostly-true`, `half-true`, and `barely-true`
remain excluded. Prepared LIAR files and labels are not changed. LIAR receives
only one quarter of a normal dataset's total training influence.

## v2 model selection and tests

The final v2 experiment uses source masking, word unigram/bigram TF-IDF, and the
down-weighted LIAR endpoint view. Thresholds are selected from dataset-held-out
validation predictions.

| Model | Selected threshold | Source-disjoint validation mean macro-F1 | Worst macro-F1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | **0.49** | **0.5805** | **0.4283** |
| Linear SVM | 0.71 | 0.5338 | 0.2295 |

Logistic Regression remains the selected classical model. Its final in-domain
test macro-F1 values are:

| Dataset | v1 Logistic Regression | v2 Logistic Regression |
| --- | ---: | ---: |
| BharatGather | 0.9081 | 0.9049 |
| IFND | 0.9510 | 0.9440 |
| Indian Fake News | 0.9599 | 0.9514 |
| ISOT | 0.9839 | 0.9842 |

The small in-domain decrease is accepted because the model-selection target is
source-disjoint robustness, not random-split optimization.

The v2 Logistic Regression makes 592 errors across 12,357 test records: 325
false positives and 267 false negatives. Fifty errors are at least 0.40 away
from its `0.49` decision threshold. The largest-margin examples repeat the same
failure modes seen in v1: genuine fact-check explanations are mistaken for fake
content, while fabricated articles written in conventional news style are
mistaken for real. The v2 Linear SVM makes 679 errors, including 554 false
negatives, which is consistent with its overly conservative `0.71` threshold.

## Contemporary 2026 diagnostic

`evaluation/contemporary_2026.json` contains ten short authentic official-source
claims and ten short claims debunked by AP or Full Fact. It was created after
all feature and threshold decisions and is evaluation-only.

| Model | Accuracy | Balanced accuracy | Macro-F1 | False claims detected |
| --- | ---: | ---: | ---: | ---: |
| v1 Logistic Regression | 0.65 | 0.65 | 0.6011 | 3/10 |
| v1 Linear SVM | 0.70 | 0.70 | 0.6703 | 4/10 |
| v2 Logistic Regression | **0.75** | **0.75** | **0.7333** | **5/10** |
| v2 Linear SVM | 0.70 | 0.70 | 0.6703 | 4/10 |

All models accepted all ten authentic-source claims. The selected v2 Logistic
Regression still missed half of the debunked claims, especially factual-sounding
claims about ballots, press credentials, oil reserves, and manipulated images.
This confirms that content classification alone cannot reliably verify facts.

## Reproduction

```powershell
python -m scripts.audit_classical
python -m scripts.evaluate_classical_lodo
python -m scripts.select_robust_representation
python -m scripts.train_classical_robust
python -m scripts.evaluate_contemporary
python -m scripts.export_classical_errors
```

Generated model files, detailed error CSV files, predictions, and JSON reports
remain ignored under `models/`. The aggregate methodology and results in this
document are tracked.
