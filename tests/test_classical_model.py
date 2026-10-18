from __future__ import annotations

import pandas as pd
import pytest

from backend.ml.classical_model import (
    ClassicalModelConfig,
    build_pipeline,
    combine_text_fields,
    evaluate_binary_classifier,
    fake_probabilities,
    validate_binary_labels,
)


def fixture_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title_clean": [
                "Official report confirms growth",
                "Fabricated miracle cure exposed",
                "Audited results released",
                "Secret hoax shocks everyone",
                "Court publishes verified ruling",
                "Invented conspiracy goes viral",
            ],
            "text_clean": [
                "Official report confirms growth with audited figures and named sources.",
                "Fabricated miracle cure exposed after experts found no supporting study.",
                "Audited results released by the agency with complete records.",
                "Secret hoax shocks everyone despite offering no documents or witnesses.",
                "Court publishes verified ruling and links the signed judgment.",
                "Invented conspiracy goes viral using anonymous and unsupported claims.",
            ],
            "label_binary_fake": [0, 1, 0, 1, 0, 1],
        }
    )


def test_combine_text_avoids_repeating_title_prefix() -> None:
    frame = pd.DataFrame(
        {
            "title_clean": ["A title", "Different title"],
            "text_clean": ["A title followed by its body.", "A separate body."],
        }
    )
    documents = combine_text_fields(frame)
    assert documents.iloc[0] == "A title followed by its body."
    assert documents.iloc[1] == "Different title\n\nA separate body."


def test_label_validation_rejects_non_binary_or_single_class() -> None:
    with pytest.raises(ValueError, match="both binary labels"):
        validate_binary_labels([1, 1])
    with pytest.raises(ValueError, match="only 0"):
        validate_binary_labels([0, "unknown", 1])


@pytest.mark.parametrize(
    "model_name", ["logistic_regression", "linear_svm", "random_forest"]
)
def test_approved_pipeline_returns_fake_probabilities_and_metrics(model_name: str) -> None:
    frame = fixture_frame()
    config = ClassicalModelConfig(max_features=100, min_df=1, ngram_max=1)
    model = build_pipeline(model_name, config)
    documents = combine_text_fields(frame)
    model.fit(documents, frame["label_binary_fake"])

    probabilities = fake_probabilities(model, documents)
    metrics = evaluate_binary_classifier(model, documents, frame["label_binary_fake"])
    assert probabilities.shape == (6,)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    assert metrics["records"] == 6
    assert set(metrics["per_class"]) == {"real_0", "fake_1"}
    assert len(metrics["confusion_matrix"]) == 2


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        build_pipeline("random_guess")


def test_evaluation_accepts_custom_threshold() -> None:
    frame = fixture_frame()
    model = build_pipeline(
        "logistic_regression",
        ClassicalModelConfig(max_features=100, min_df=1, ngram_max=1),
    )
    documents = combine_text_fields(frame)
    model.fit(documents, frame["label_binary_fake"])
    result = evaluate_binary_classifier(
        model,
        documents,
        frame["label_binary_fake"],
        threshold=0.4,
    )
    assert result["threshold"] == pytest.approx(0.4)


def test_evaluation_rejects_invalid_threshold() -> None:
    frame = fixture_frame()
    model = build_pipeline(
        "logistic_regression",
        ClassicalModelConfig(max_features=100, min_df=1, ngram_max=1),
    )
    documents = combine_text_fields(frame)
    model.fit(documents, frame["label_binary_fake"])
    with pytest.raises(ValueError, match="threshold"):
        evaluate_binary_classifier(
            model,
            documents,
            frame["label_binary_fake"],
            threshold=1.0,
        )
