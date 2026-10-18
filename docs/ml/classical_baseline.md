# Classical ML baseline protocol

## Scope

This stage implements article-level content classification only. It produces the
first fusion-vector feature, `classical_fake_probability`, using the convention
`0 = real` and `1 = fake`. It does not retrieve evidence or verify claims.

This document records the original two-model baseline. Random Forest was added
later under ADR 0003; see `classical_three_model_audit.md` for current scope and
results.

## Initial benchmark dataset

ISOT is the first benchmark because its binary label mapping was verified during
Phase 4 and leakage-aware train, validation, and test files already exist. The
pipeline refuses datasets whose binary mapping remains unresolved.

The ISOT benchmark is not evidence of general real-world performance. Its real
and fake articles were collected from different source pools, so models may learn
publisher, topic, formatting, and writing-style cues. External, temporal, and
cross-dataset tests are required before model selection for the final system.

## Experiment design

- Input: cleaned title and article body; a repeated title prefix is included once.
- Features: word TF-IDF, unigrams and bigrams, Unicode accent normalization,
  sublinear term frequency, maximum 75,000 features.
- Candidates: class-weighted logistic regression and a class-weighted linear SVM
  with sigmoid probability calibration.
- Selection: highest validation macro-F1, with validation ROC-AUC as tie-breaker.
- Final fit: both approved candidates retrained on train plus validation.
- Final evaluation: both models evaluated on the held-out test after selection;
  the validation winner is marked as the preferred baseline.
- Seed: `20261017`.
- Metrics: accuracy, balanced accuracy, per-class precision/recall/F1,
  macro/weighted F1, ROC-AUC, average precision, Brier score, and confusion matrix.

Artifacts and generated metrics are written below `models/`, which is intentionally
ignored by Git because trained files can be large and are reproducible from local
prepared data.

## Reproduction

```powershell
python -m pip install -r requirements-ml.txt
python -m scripts.train_classical --dataset isot
python -m scripts.evaluate_classical --training-dataset isot
```

The script validates the Phase 4 manifest and label semantics before training.
Both pipelines and their JSON evaluation reports are saved locally under
`models/classical/isot/<model>/`; the overall comparison is `benchmark.json`.
The second command tests the same frozen pipelines against the test splits of
other datasets with verified binary labels; it does not retrain on those targets.

## Interpretation boundary

The output is probability assigned by a dataset-trained classifier. It measures
similarity to learned content patterns; it is not a factuality guarantee. The
future hybrid system must display it as one signal alongside retrieved evidence,
NLI verification, source context, and an explicit insufficient-evidence state.
