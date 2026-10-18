"""Verify repository structure and setup-safe imports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import setup_status  # noqa: E402


REQUIRED = [
    "README.md",
    ".gitignore",
    ".env.example",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-data.txt",
    "requirements-ml.txt",
    "backend/main.py",
    "backend/ml/classical_model.py",
    "docs/research/research_review.md",
    "docs/architecture/system_design.md",
    "docs/data/dataset_protocol.md",
    "docs/data/dataset_preparation_report.md",
    "docs/ml/classical_baseline.md",
    "docs/ml/classical_results.md",
    "scripts/prepare_datasets.py",
    "scripts/validate_prepared_data.py",
    "scripts/train_classical.py",
    "scripts/evaluate_classical.py",
    "scripts/predict_classical.py",
    "notebooks/classical_ml_colab.ipynb",
    "data/raw",
    "data/interim",
    "data/processed",
    "frontend",
    "tests",
]


def main() -> int:
    missing = [item for item in REQUIRED if not (ROOT / item).exists()]
    payload = {
        "status": "ok" if not missing else "failed",
        "missing": missing,
        "setup": setup_status(),
    }
    print(json.dumps(payload, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
