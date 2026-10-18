from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "classical_ml_colab.ipynb"


def load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_colab_notebook_is_valid_clean_v4_json() -> None:
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    assert isinstance(cells, list) and cells
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)


def test_colab_notebook_uses_only_approved_classical_models() -> None:
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert 'APPROVED_MODELS = (\\"logistic_regression\\", \\"linear_svm\\")' in notebook_text
    assert "random_forest" not in notebook_text.casefold()
    assert "xgboost" not in notebook_text.casefold()


def test_colab_notebook_uses_public_clone_drive_validation_and_external_evaluation() -> None:
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    required_markers = [
        'https://github.com/{REPO_OWNER}/{REPO_NAME}.git',
        'drive.mount(\\"/content/drive\\")',
        "validate_prepared_data.py",
        "scripts.train_classical_multidataset",
        "scripts.predict_classical",
    ]
    assert all(marker in notebook_text for marker in required_markers)
    assert "GITHUB_TOKEN" not in notebook_text
    assert "ghp_" not in notebook_text


def test_colab_inference_uses_clean_subprocess() -> None:
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert "capture_output=True" in notebook_text
    assert "from backend.ml.classical_model import fake_probabilities" not in notebook_text
