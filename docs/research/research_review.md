# Research review and design implications

## Scope

The review covers 18 supplied research papers, one project technology/workflow document, one architecture/feasibility deck, the supplied frontend requirements, four architecture sketches, the proposed backend layout, and the 11-feature fusion example.

## Study-level synthesis

| Supplied study | Reported method and result | Limitation relevant to this project |
| --- | --- | --- |
| Real-time computational approach | Event/topic extraction, topic merging, and Spark; reports average topic and cluster reductions of 19.76% and 26.92% plus runtime gains | Source-specific heuristic evaluation; ambiguity in evolving events; not a claim-evidence verifier |
| CNN fake-news detection | GloVe plus CNN on a balanced 10,000-record sample; reports 92.15% accuracy | Random split and aggressive truncation/cleaning; no evidence grounding |
| BERT, ML, and manual fusion | Multilingual classifiers plus hard-coded domain, grammar, and swear-word weights | Hand-tuned weights and curated source lists are fragile; low Urdu results; credibility is not truth |
| Performance-oriented ML study | Reports classical, deep, transformer, and hybrid comparisons with a proposed 98.6% ensemble | Dataset and experimental reporting contain ambiguities; treat numbers as unverified until reproducible |
| Dravidian hybrid model | MuRIL features with DTCN, BiLSTM, attention, and adaptive cyclical learning; reports 93.97% average accuracy | Single curated dataset, fixed 128-token length, high complexity, no external fact verification |
| Emotion-driven multimodal network | Visual/text attention and emotion features; reports 0.94-0.98 accuracy across four reported datasets and positive ablations | Emotion is correlated context, not proof; domain and dataset-specific behavior; multimodal cost |
| WELFake ensemble study | BERT, RF, GRU, LSTM, and ensembles; reports 99% BERT and 100% after tuning | Near-perfect result needs leakage/duplicate audit; ensemble underperforms BERT; half-dataset usage unclear |
| Propagation-aware dual-adapter LLM | LLaMA adapters plus propagation graphs for zero-shot cross-domain rumor detection; reports gains across four transfer settings | Requires social propagation graphs unavailable for ordinary article/URL input; higher compute and recency |
| Cross-modal fake-video retrieval | Related video/text retrieval, information bottleneck, and LLM prompts; reports 85.2% on FakeSV | Video-specific; depends on high-quality related news; hallucinations persist; no model fine-tuning |
| FakeIDCA | Incremental ensemble for concept drift; reports strong temporal transfer and large static-model degradation | Time split construction and source effects need reproduction; operational drift monitoring adds complexity |
| Semantic plus engineered features | RoBERTa/CNN representation plus topic, category, sentiment, and entities; reports consistent gains on English and Persian data | Engineered features can encode dataset/source shortcuts; social context left for future work |
| Dual-BERT MWPBert | Headline BERT plus check-worthy body span BERT; reports 0.854 accuracy and improved recall on FakeNewsNet | Long-document selection remains heuristic; imbalanced dataset and random split; classification only |
| Generic ML detection study | Describes conventional text preprocessing and classifiers | Internal reporting appears inconsistent, including 75% and 94.5% accuracy claims; insufficient evidence for design selection |
| Hindi transformer ensemble | ELECTRA, mBERT, XLM-RoBERTa with majority-veto voting; reports over 90% accuracy | One-vs-all relabeling treats all non-fake hostility classes as factual; some ensemble gains are not significant |
| Transformer plus LLM reliability features | Gemini-generated factual/temporal features fed to a second BERT; reports 94% using a small subset | LLM features are ungrounded and may leak label/source cues; unusual 500-epoch report; needs independent validation |
| CNN-LSTM hybrid | Text and image pipelines; reports 96.1% text and 91.36% image accuracy | Small image set, separate modalities rather than a unified evidence task; random splits likely |
| Multimodal deep-learning survey | Reviews text, image, audio, video, social, and fusion approaches | Finds multimodal fusion, explainability, domain transfer, and dataset comparability remain open problems |
| Hybrid framework/project paper corpus | Supports combining classifiers and context features | Combining components remains a proposal until evaluated end to end |

## Findings that guide setup

- Establish classical baselines before choosing a complex model.
- Keep classifier signals separate from factual verification.
- Treat hybrid lexical/semantic retrieval as a candidate whose recall must be measured.
- Apply NLI after retrieval and retain an explicit unknown state.
- Keep LLMs out of the final truth decision unless later experiments justify a constrained role.
- Evaluate long-document strategies instead of assuming naive truncation.
- Preserve emotion, clickbait, style, credibility, and entity features as contextual signals.
- Make temporal, event, source, and duplicate leakage tests mandatory.
- Compare transparent weighted fusion with learned fusion; do not assume XGBoost is best.
- Add calibration and abstention evaluation to accuracy, precision, recall, and F1.

## Proposed design choices, not literature facts

- BM25 plus BGE-M3/FAISS combined with reciprocal rank fusion.
- BGE reranking after rank fusion.
- The exact 11-feature vector and its ordering.
- DeBERTa-v3 or ModernBERT as the main classifier.
- A DeBERTa-family NLI verifier.
- XGBoost as the learned decision engine.
- PostgreSQL, Redis, FastAPI, React, Tailwind, Vite, and the named external APIs.
- An OpenAI-backed explanation layer after the decision.

Each choice remains replaceable based on data, licensing, latency, reproducibility, and experiments.

## Unresolved decisions

- Real dataset schema, license, label semantics, languages, dates, and predefined splits.
- Article-level versus claim-level ground truth alignment.
- Evidence corpus scope, allowable publication-time window, and API availability.
- Credibility rubric and how to avoid encoding publisher identity as a label shortcut.
- Calibration method and abstention thresholds.
- Claim weighting and aggregation into support/contradiction/unknown ratios.
- Missingness behavior for APIs, sources, and short or non-factual inputs.
- Long-document selection strategy.
- Deployment latency and compute budget.
- Data retention, history deletion, export, and privacy requirements.

## Research integrity cautions

Reported results use different datasets, labels, modalities, splits, and metrics and are not directly comparable. Some supplied papers have unclear or internally inconsistent experimental descriptions. The repository therefore records their claims without adopting the reported performance as a project baseline. All model and fusion decisions require reproduction on leakage-controlled splits of the user-supplied dataset.
