# Hybrid Fake News Detection and Fact Verification

Academic research project for an evidence-grounded system that combines content classification, claim-level verification, source context, and an auditable decision layer.

## Project objective

The proposed system will eventually accept article text, a URL, a claim, or a document and produce an article-level assessment with calibrated confidence, supporting and contradicting evidence, an explicit insufficient-evidence state, and a grounded explanation.

The central research hypothesis is that the following signals can complement one another:

- classical text classification;
- transformer classification;
- claim extraction and hybrid evidence retrieval;
- NLI support, contradiction, and unknown decisions;
- source credibility and contextual linguistic signals;
- transparent or learned feature fusion;
- an LLM explanation generated after the decision from cited evidence.

This hypothesis has not yet been experimentally validated for this project. Reported results in the reviewed literature are dataset-specific and must not be treated as expected project performance.

## Current status

Phases 1-4 are complete:

- supplied research materials reviewed and indexed;
- proposed architecture, feature vector, assumptions, and risks documented;
- public GitHub repository and local clone established;
- reproducible Python environment instructions and initial checks added;
- the eight user-supplied dataset sources downloaded and preserved with SHA-256 hashes;
- source-specific schema, label, missingness, duplication, conflict, and leakage checks completed;
- conservative cleaned copies and deterministic leakage-aware splits generated locally;
- preparation tooling tested with synthetic fixtures and verified against the real outputs.

Classical ML now includes an auditable, dataset-balanced TF-IDF benchmark:

- verified labels and leakage-aware splits from ISOT, IFND, Indian Fake News, and BharatGather are used;
- logistic regression, calibrated linear SVM, and Random Forest are compared;
- dataset/class-balanced sample weights prevent large collections from dominating;
- selection uses mean per-dataset validation macro-F1;
- both final models receive one held-out evaluation on every dataset;
- generated model files and metrics remain local and ignored by Git.

A robustness audit and source-aware v2 experiment are also complete:

- no exact or configured near-duplicate leakage was found across splits;
- severe publisher/collection label shortcuts were measured in IFND and ISOT;
- leave-one-dataset-out evaluation exposed weak unseen-source transfer;
- source boilerplate masking and down-weighted LIAR endpoint examples improved
  source-disjoint validation without introducing another classifier family;
- a validation-only Random Forest tuning audit retained the 200-tree forest,
  while Logistic Regression remained preferred over both alternative models;
- a frozen 20-claim contemporary diagnostic improved from 0.6011 to 0.7333
  macro-F1, but only half of the debunked claims were detected.

The raw, interim, and processed dataset files remain local and are ignored by Git. Aggregate results and limitations are recorded in `docs/data/dataset_preparation_report.md`.

Not implemented: production APIs, data extraction services, transformer classification, RAG, NLI, credibility scoring, fusion training, LLM calls, database services, frontend, and deployment. Classical ML is an experimental baseline and is not yet integrated into an API.

## Proposed architecture

```text
Input -> preprocessing -> parallel analysis
                           |-- classical ML
                           |-- transformer ML
                           |-- claim extraction -> retrieval -> reranking -> NLI
                           |-- source credibility
                           `-- auxiliary context
                        -> feature construction
                        -> transparent baseline / learned fusion
                        -> classification + calibrated confidence
                        -> evidence-grounded explanation
```

Retrieval relevance and factual verification remain separate. A relevant passage does not necessarily support a claim. The system must also distinguish “no evidence found” from “source/API unavailable.”

## Repository layout

```text
backend/       setup-safe Python package and planned module boundaries
data/          ignored raw/interim/processed data locations
docs/          research synthesis, architecture, decisions, and data protocol
frontend/      frontend scope only; application not implemented
notebooks/     future experiments; outputs and checkpoints are ignored
scripts/       setup verification, dataset inspection, preparation, and validation
tests/         setup and synthetic data-preparation checks
```

## Local setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-data.txt
python -m pytest
python scripts\verify_setup.py
python scripts\validate_prepared_data.py --root .
```

Application runtime dependencies intentionally remain empty until post-Phase-4 implementation choices justify them. Development dependencies are isolated in `requirements-dev.txt`; the Phase 4 preparation stack is isolated in `requirements-data.txt`.

For the classical ML environment and benchmark:

```powershell
python -m pip install -r requirements-ml.txt
python -m scripts.train_classical_multidataset
python -m scripts.audit_classical
python -m scripts.evaluate_classical_lodo
python -m scripts.select_robust_representation
python -m scripts.train_classical_robust
python -m scripts.evaluate_contemporary
python -m scripts.export_classical_errors
```

For Google Colab, use
[Open in Colab](https://colab.research.google.com/github/hardiktyagi032/hybrid-fake-news-detection/blob/main/notebooks/classical_ml_colab.ipynb).
The public notebook requires no GitHub token. It uses Google Drive for persistent
data and model artifacts, trains Logistic Regression, Linear SVM, and Random Forest, and runs
the same validation and cross-dataset evaluation workflow.

See `docs/ml/classical_baseline.md` for the protocol and
`docs/ml/classical_results.md` for the original ISOT-only baseline. The improved
dataset-balanced results are in `docs/ml/multidataset_results.md`.
The leakage, source-shortcut, error, threshold, source-disjoint, and contemporary
results are in `docs/ml/robustness_audit.md`.
The completed three-model comparison and Random Forest tuning decisions are in
`docs/ml/classical_three_model_audit.md`.

## Dataset preparation

Files under `data/` are ignored by Git by default. The received source archives are under `data/raw/_archives/`, extracted sources under `data/raw/sources/`, inspection and exclusions under `data/interim/`, and cleaned datasets/splits under `data/processed/`.

To reproduce Phase 4 from the preserved raw sources:

```powershell
python scripts\prepare_datasets.py --root . --seed 20261017 --receipt-date 2026-10-17
python scripts\validate_prepared_data.py --root .
```

The preparation command never writes into `data/raw/`. It preserves raw and cleaned text separately, quarantines empty/ambiguous duplicate groups, and does not invent binary mappings for datasets whose documentation is insufficient.

## Safety and research integrity

- Never commit credentials, `.env`, raw datasets, model weights, caches, or generated artifacts.
- Preserve original data unchanged and verify it with a cryptographic hash.
- Separate reported findings, design choices, and unverified assumptions.
- Use event-, source-, time-, and near-duplicate-aware splits where the data supports them.
- Report accuracy together with precision, recall, F1, calibration, and error analysis.
- Treat confidence as a calibrated estimate, not a guarantee of truth.

See `docs/research/research_review.md`, `docs/architecture/system_design.md`, `docs/data/dataset_protocol.md`, and `docs/data/dataset_preparation_report.md` for the current evidence base, results, and open decisions.
