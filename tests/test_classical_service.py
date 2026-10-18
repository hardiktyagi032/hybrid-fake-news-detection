from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from backend.ml.classical_service import (
    ClassicalArtifactError,
    ClassicalInput,
    ClassicalMLService,
    ClassicalModelRegistry,
    sha256_file,
)
from backend.ml.robust_classical_model import RobustRepresentationConfig, build_robust_pipeline


def create_model_set(root: Path) -> Path:
    model_set = "test_models"
    model_root = root / "models" / "classical" / model_set
    documents = [
        "official audited report with records",
        "fabricated viral hoax without evidence",
        "court publishes verified signed ruling",
        "invented miracle conspiracy shocks everyone",
        "agency releases documented public figures",
        "false anonymous story makes unsupported claim",
    ]
    labels = [0, 1, 0, 1, 0, 1]
    config = RobustRepresentationConfig(
        representation="word",
        min_df=1,
        word_max_features=100,
        random_forest_estimators=10,
        random_forest_n_jobs=1,
    )
    artifacts = {}
    validation = {}
    models = ["logistic_regression", "linear_svm", "random_forest"]
    for name in models:
        model = build_robust_pipeline(name, config)
        model.fit(documents, labels)
        path = model_root / name / "pipeline.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        artifacts[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        validation[name] = {"selected_threshold": {"threshold": 0.5}}
    manifest = {
        "model_set": model_set,
        "models": models,
        "selected_model": "logistic_regression",
        "artifacts": artifacts,
        "validation": validation,
        "datasets": ["synthetic_test_fixture"],
        "selection_metric": "test fixture",
    }
    (model_root / "benchmark.json").write_text(json.dumps(manifest), encoding="utf-8")
    return model_root


def test_service_runs_all_three_models_and_exposes_provenance(tmp_path: Path) -> None:
    create_model_set(tmp_path)
    service = ClassicalMLService(
        ClassicalModelRegistry(tmp_path, "test_models"), abstention_margin=0.0
    )
    result = service.analyze(
        ClassicalInput(text="official agency publishes an audited report"), top_k=3
    )

    assert set(result["models"]) == {
        "logistic_regression",
        "linear_svm",
        "random_forest",
    }
    assert result["classification"] in {"real", "fake", "uncertain"}
    assert result["preferred_model"] == "logistic_regression"
    assert result["task"] == "classical_content_classification"
    assert result["model_provenance"]["model_set"] == "test_models"
    assert result["models"]["random_forest"]["text_indicators"]["available"] is True
    assert "not evidence-based" in result["warning"]


def test_registry_rejects_modified_artifact(tmp_path: Path) -> None:
    model_root = create_model_set(tmp_path)
    path = model_root / "random_forest" / "pipeline.joblib"
    with path.open("ab") as handle:
        handle.write(b"changed")
    registry = ClassicalModelRegistry(tmp_path, "test_models")
    with pytest.raises(ClassicalArtifactError, match="Checksum mismatch"):
        registry.load("random_forest")


def test_service_validates_empty_and_oversized_input(tmp_path: Path) -> None:
    create_model_set(tmp_path)
    service = ClassicalMLService(
        ClassicalModelRegistry(tmp_path, "test_models"), max_characters=10
    )
    with pytest.raises(ValueError, match="non-empty"):
        service.analyze({"text": ""})
    with pytest.raises(ValueError, match="character limit"):
        service.analyze({"text": "a" * 11})
