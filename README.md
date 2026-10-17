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

Phases 1-3 are complete at repository initialization:

- supplied research materials reviewed and indexed;
- proposed architecture, feature vector, assumptions, and risks documented;
- private GitHub repository and local clone established;
- reproducible Python environment instructions and initial checks added;
- dataset inspection tooling tested with synthetic fixtures.

Phase 4 is waiting for the user-supplied dataset. No replacement dataset has been selected or downloaded.

Not implemented: production APIs, data extraction services, classifiers, RAG, NLI, credibility scoring, fusion training, LLM calls, database services, frontend, deployment, and model evaluation.

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
scripts/       setup verification and non-mutating dataset inspection
tests/         initial automated checks and synthetic fixtures
```

## Local setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest
python scripts\verify_setup.py
```

Runtime dependencies intentionally remain empty until the real dataset and post-Phase-4 implementation choices justify them. Development dependencies are isolated in `requirements-dev.txt`.

## Dataset handoff

Place the original user-supplied dataset under `data/raw/`. Files under `data/` are ignored by Git by default. Before any transformation, record its SHA-256 hash and run:

```powershell
python scripts\dataset_inspect.py data\raw\YOUR_FILE.csv --output data\interim\inspection.json
```

The inspection command is read-only with respect to the input. Cleaning and splitting decisions must be made from the actual schema and documented under `docs/data/` before generating processed outputs.

## Safety and research integrity

- Never commit credentials, `.env`, raw datasets, model weights, caches, or generated artifacts.
- Preserve original data unchanged and verify it with a cryptographic hash.
- Separate reported findings, design choices, and unverified assumptions.
- Use event-, source-, time-, and near-duplicate-aware splits where the data supports them.
- Report accuracy together with precision, recall, F1, calibration, and error analysis.
- Treat confidence as a calibrated estimate, not a guarantee of truth.

See `docs/research/research_review.md`, `docs/architecture/system_design.md`, and `docs/data/dataset_protocol.md` for the current evidence base and open decisions.
