# Planned backend layout

The supplied folder proposal is retained as the intended module boundary. Empty Python implementations were not created.

```text
backend/
  main.py                 setup status only
  config/                 safe environment settings
  api/                    future input, analysis, and health routes
  preprocessing/          future text, URL, PDF, and OCR extraction
  ml/                     future classical and transformer classifiers
  rag/                    future claims, sources, retrieval, fusion, reranking
  nli/                    future claim-evidence verification
  credibility/            future transparent source features
  auxiliary/              future clickbait, emotion, bias, style, language, entities
  fusion/                 future feature contract and decision engine
  llm/                    future grounded explanation
  database/               future persistence models and access
  cache/                  future cache policy
  utils/                  setup-safe reproducibility helpers
```

API contracts, database schema, external services, and model loaders remain unimplemented because they depend on dataset findings and later authorization.
