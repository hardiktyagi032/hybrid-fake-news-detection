# Planned backend layout

The supplied folder proposal is retained as the intended module boundary. Empty Python implementations were not created.

```text
backend/
  main.py                 setup status only
  config/                 safe environment settings
  api/                    future input, analysis, and health routes
  preprocessing/          future text, URL, PDF, and OCR extraction
  ml/                     implemented classical models/loader; transformer future
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

The classical package now contains the three approved TF-IDF pipelines plus a
checksum-validated local artifact registry and model-wise inference service.
API contracts, database schema, external services, and non-classical model
loaders remain unimplemented because they depend on later authorization.
