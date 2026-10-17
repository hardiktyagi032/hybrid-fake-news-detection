# Proposed system design

## Evidence-backed principles

The supplied literature supports several narrow conclusions: transformer and engineered-feature models can outperform baselines on particular datasets; long articles require an explicit span strategy; emotion, social context, propagation, and multimodal cues can add signal; random in-distribution performance can deteriorate under time or domain shift; external evidence improves factual grounding; and LLM outputs can hallucinate.

These findings do not establish that the complete proposed architecture, any named model, or XGBoost fusion will outperform alternatives on the user-supplied dataset.

## Proposed pipeline

1. Accept text, URL, claim, or document input.
2. Extract text while preserving the original input and provenance.
3. Normalize conservatively and retain both raw and model-specific representations.
4. Run classical and transformer classifiers as independent pattern signals.
5. Extract atomic claims without assigning truth.
6. Retrieve evidence lexically and semantically, fuse ranks, and rerank.
7. Apply NLI to each claim-evidence pair for supported, contradicted, or unknown status.
8. Compute source and cross-source features with transparent provenance.
9. Compute auxiliary context features without treating style or emotion as proof.
10. Build a versioned feature record.
11. Compare an auditable weighted baseline against a learned fusion model.
12. Calibrate confidence and preserve abstention/insufficient-evidence behavior.
13. Generate an explanation only from the decision record and cited evidence.

## Eleven-feature fusion contract

| Position | Feature | Proposed source | Open definition |
| ---: | --- | --- | --- |
| 1 | Classical fake probability | classical ML | Requires out-of-fold calibration |
| 2 | Transformer fake probability | transformer | Requires out-of-fold calibration |
| 3 | Supported claim ratio | NLI | Aggregation and claim weighting undecided |
| 4 | Contradicted claim ratio | NLI | Aggregation and claim weighting undecided |
| 5 | Unknown claim ratio | NLI | Must differ from API outage/missing retrieval |
| 6 | Evidence relevance | retriever/reranker | Scale and aggregation undecided |
| 7 | Source credibility | credibility module | Transparent rubric and missing-source policy undecided |
| 8 | Cross-source agreement | evidence analysis | Independence and publisher-family handling undecided |
| 9 | Clickbait | auxiliary analysis | Contextual feature, never direct proof |
| 10 | Emotion | auxiliary analysis | Contextual feature, never direct proof |
| 11 | Entity consistency | auxiliary analysis | Direction and scale require definition |

The supplied numeric vector is an illustrative example, not training data and not a validated operating point. The NLI ratios sum to one in that example, but the high fake probabilities and high support ratio disagree. The fusion layer must preserve and learn from disagreement instead of forcing artificial consistency.

## Required invariants

- Features have a documented orientation, range, version, timestamp, and provenance.
- Supported, contradicted, and unknown ratios sum to one within tolerance when at least one claim is evaluated.
- Missing evidence, no evidence, and service failure remain distinct states.
- Classifier probabilities are calibrated from out-of-fold predictions on the training partition only.
- Learned fusion never trains on predictions generated from the same samples used to fit its base models.
- Dataset splits group exact and near duplicates and should also consider event, source, and time.
- Evidence published after an article or label event is controlled during evaluation.
- Final explanations cite retrieved records and cannot introduce unsupported claims.

## Initial technology choices

The supplied documents propose React, Tailwind CSS, Vite, FastAPI, Python, PostgreSQL, Redis, BM25, BGE-M3, FAISS, a BGE reranker, DeBERTa-style encoders, Hugging Face, PyTorch, XGBoost or weighted fusion, and an OpenAI-backed explanation layer. These remain candidates. Dependency versions and final component choices will follow dataset inspection and later authorization.

## Risks and dependencies

- Source availability, API quotas, licensing, and terms may constrain retrieval.
- Publisher identity is not equivalent to article truth.
- Fact-check sources can introduce label leakage.
- Long documents can lose decisive text under naive truncation.
- Article-level labels may not align with claim-level NLI.
- Benchmarks often contain source, topic, temporal, or duplicate shortcuts.
- Multilingual and code-mixed performance cannot be inferred from English benchmarks.
- Confidence requires calibration and abstention testing.
- LLM explanations require evidence constraints and audit logs.
- The current 11-feature vector lacks explicit service-availability indicators; keep them in the decision record and evaluate whether the fusion schema must expand.
