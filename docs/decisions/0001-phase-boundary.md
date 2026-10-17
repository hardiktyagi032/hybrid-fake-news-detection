# ADR 0001: Preserve the Phase 4 stopping boundary

## Status

Accepted.

## Decision

Repository initialization may include research synthesis, module boundaries, safe configuration, dataset-inspection utilities, and tests with synthetic fixtures. It must not include production APIs, trained models, retrieval, NLI, fusion implementation, LLM integration, frontend implementation, or deployment.

## Consequences

The repository is usable for dataset handoff and reproducible inspection while remaining honest about unimplemented components. Later work requires explicit user instruction after Phase 4.
