# Frontend functional scope

The future React interface must support the complete journey:

```text
Input -> Analyze -> ML + RAG + fact verification + LLM -> hybrid result -> evidence -> explanation -> follow-up chat
```

## Functional areas

- Fact-check input: new session, article text, URL, claim, upload, submit, clear, edit, follow-up, continue, and start over.
- Pipeline visibility: preprocessing, ML, retrieval, verification, LLM, fusion, and overall progress.
- Results: classification, confidence, summary, explanation, findings, supporting/contradicting evidence, unknown state, and expandable detail.
- ML detail: aggregate and model-level predictions, calibrated confidence, indicators, agreement, and technical detail.
- Evidence: source metadata, dates, relevance, stance, snippets, fact-check records, original links, and complete retrieval view.
- Conversation: evidence-grounded explanation, claim-to-evidence comparison, contradiction explanation, follow-up prompts, and context-aware continuity.
- History: search, date grouping, reopen, continue, delete, and rename.
- Dashboard: counts, classification distribution, activity, recent work, trends, and source overview.
- Source explorer: browse, search, filter, inspect, open, and view linked evidence.
- Navigation: new check, chat, history, dashboard, sources, settings, and profile.
- Settings: theme, language, explanation detail, technical/source visibility, history preferences, and privacy controls.
- System states: loading, progress, success, invalid input, error, no evidence, insufficient evidence, service unavailable, and retry.
- Utilities: copy, share, export, download, feedback, and report incorrect analysis.

## Acceptance constraints

- UI wording must not present relevance, confidence, or source reputation as proof.
- “No evidence” and “service unavailable” must render differently.
- Progress must reflect backend events rather than simulated certainty.
- Source links and snippets retain provenance and publication dates.
- Technical detail can be hidden without hiding the evidence behind the decision.
- History deletion and privacy controls require explicit confirmation and auditable behavior.

No frontend code exists yet.
