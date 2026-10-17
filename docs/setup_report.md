# Initial setup verification report

Date: 2026-10-17 (Asia/Calcutta)

## Repository

- Remote: `https://github.com/hardiktyagi032/hybrid-fake-news-detection`
- Visibility: private
- Local clone: verified in the current Codex workspace
- Default branch: `main`

## Environment

- Git: 2.55.0.windows.5
- Python: 3.14.5
- Virtual environment: `.venv`
- pip: 26.1.1
- pytest: 9.1.1
- Application runtime dependencies: none
- Phase 4 data dependencies: pandas 3.0.6, NumPy 2.5.4, openpyxl 3.1.5

## Verification performed

| Check | Result |
| --- | --- |
| Python test suite | 9 passed |
| Setup verifier | Passed; no required paths missing |
| Python bytecode compilation | Passed |
| Git whitespace validation | Passed |
| Ignore rules for `.env`, datasets, virtual environment, model weights, and frontend dependencies | Passed |
| GitHub Actions `initial-checks` | Passed on Linux with Python 3.12 |
| Phase 4 raw archive hash validation | Passed for 10 files |
| Phase 4 prepared output validation | Passed for 32 files and 8 dataset groups |
| Phase 4 reproducibility rerun | Passed; manifest SHA-256 unchanged |

The first sandboxed dependency installation attempt failed because external DNS was unavailable. The approved network execution succeeded. The first test attempt exposed a sandbox temporary-directory restriction and a direct-script import issue; both were corrected, and all checks passed on rerun.

The first remote workflow exposed that a root cache ignore rule also removed the planned `backend/cache/` boundary from a clean clone. The ignore rule was narrowed, the missing tracked marker was added, and the corrected workflow completed successfully.

## Phase boundary

The repository contains research and architecture documentation, safe configuration, setup status, dataset inspection/preparation/validation tooling, and synthetic tests. The user-supplied sources were prepared locally; raw and generated files remain ignored by Git. Detailed results are in `docs/data/dataset_preparation_report.md`.

It does not contain detection models, RAG, NLI, decision fusion, LLM calls, production backend APIs, frontend code, database services, or deployment configuration.
