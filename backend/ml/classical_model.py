"""TF-IDF classical baselines for article-level fake-news classification.

The model predicts content-pattern similarity to the training labels. It is not
an evidence-based fact verifier, and callers must not present its probability as
proof that an article is true or false.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


SUPPORTED_MODELS = ("logistic_regression", "linear_svm", "random_forest")


@dataclass(frozen=True)
class ClassicalModelConfig:
    """Reproducible configuration shared by the classical candidates."""

    max_features: int = 75_000
    min_df: int = 2
    max_df: float = 0.98
    ngram_min: int = 1
    ngram_max: int = 2
    sublinear_tf: bool = True
    class_weight: str | None = "balanced"
    random_state: int = 20_261_017
    logistic_c: float = 2.0
    svm_c: float = 1.0
    random_forest_estimators: int = 200
    random_forest_criterion: str = "gini"
    random_forest_max_depth: int | None = None
    random_forest_min_samples_leaf: int = 2
    random_forest_max_features: str | float = "sqrt"
    random_forest_n_jobs: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def combine_text_fields(frame: pd.DataFrame) -> pd.Series:
    """Build one document per row without repeating a title already in the body."""

    required = {"title_clean", "text_clean"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing text columns: {sorted(missing)}")

    titles = frame["title_clean"].fillna("").astype(str).str.strip()
    bodies = frame["text_clean"].fillna("").astype(str).str.strip()
    documents: list[str] = []
    for title, body in zip(titles, bodies, strict=True):
        if not title or body.casefold().startswith(title.casefold()):
            document = body or title
        elif not body:
            document = title
        else:
            document = f"{title}\n\n{body}"
        documents.append(document)

    result = pd.Series(documents, index=frame.index, dtype="string")
    if result.str.strip().eq("").any():
        raise ValueError("At least one record has no usable title or body text")
    return result


def validate_binary_labels(values: Iterable[Any]) -> np.ndarray:
    """Return integer labels after enforcing the convention 1=fake, 0=real."""

    series = pd.Series(values)
    if series.isna().any():
        raise ValueError("Binary labels contain missing values")
    try:
        labels = series.astype(int).to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError("Binary labels must contain only 0 (real) and 1 (fake)") from exc
    observed = set(np.unique(labels).tolist())
    if observed != {0, 1}:
        raise ValueError(f"Expected both binary labels {{0, 1}}, observed {sorted(observed)}")
    return labels


def build_pipeline(model_name: str, config: ClassicalModelConfig | None = None) -> Pipeline:
    """Create a word TF-IDF pipeline with a probabilistic binary classifier."""

    config = config or ClassicalModelConfig()
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
        sublinear_tf=config.sublinear_tf,
        dtype=np.float32,
    )

    if model_name == "logistic_regression":
        classifier: Any = LogisticRegression(
            C=config.logistic_c,
            class_weight=config.class_weight,
            max_iter=1_000,
            random_state=config.random_state,
            solver="liblinear",
        )
    elif model_name == "linear_svm":
        classifier = CalibratedClassifierCV(
            estimator=LinearSVC(
                C=config.svm_c,
                class_weight=config.class_weight,
                random_state=config.random_state,
            ),
            method="sigmoid",
            cv=3,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=config.random_forest_estimators,
            criterion=config.random_forest_criterion,
            max_depth=config.random_forest_max_depth,
            min_samples_leaf=config.random_forest_min_samples_leaf,
            max_features=config.random_forest_max_features,
            class_weight=config.class_weight,
            n_jobs=config.random_forest_n_jobs,
            random_state=config.random_state,
        )
    else:
        raise ValueError(f"Unsupported model {model_name!r}; choose one of {SUPPORTED_MODELS}")

    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def fake_probabilities(model: Pipeline, documents: Iterable[str]) -> np.ndarray:
    """Return P(fake) according to the project convention 1=fake."""

    probabilities = model.predict_proba(documents)
    classes = list(model.named_steps["classifier"].classes_)
    try:
        fake_index = classes.index(1)
    except ValueError as exc:
        raise ValueError(f"Model has no fake class (1); classes={classes}") from exc
    return np.asarray(probabilities[:, fake_index], dtype=float)


def evaluate_binary_classifier(
    model: Pipeline,
    documents: Iterable[str],
    labels: Iterable[Any],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate classification and probability quality with JSON-safe values."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    y_true = validate_binary_labels(labels)
    y_probability = fake_probabilities(model, documents)
    y_pred = (y_probability >= threshold).astype(int)
    precision_by_class, recall_by_class, f1_by_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "records": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "average_precision": float(average_precision_score(y_true, y_probability)),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
        "per_class": {
            "real_0": {
                "precision": float(precision_by_class[0]),
                "recall": float(recall_by_class[0]),
                "f1": float(f1_by_class[0]),
                "support": int(support[0]),
            },
            "fake_1": {
                "precision": float(precision_by_class[1]),
                "recall": float(recall_by_class[1]),
                "f1": float(f1_by_class[1]),
                "support": int(support[1]),
            },
        },
    }
