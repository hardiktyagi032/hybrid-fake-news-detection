"""Source-robust TF-IDF representations for the approved linear classifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


SOURCE_MARKERS = (
    "associated press",
    "alt news",
    "altnews",
    "ani news",
    "boom live",
    "boomlive",
    "dna india",
    "dnaindia",
    "factly",
    "india today",
    "ndtv",
    "news meter",
    "newsmeter",
    "pti",
    "reuters",
    "the quint",
    "the wire",
    "tribune india",
    "tribuneindia",
    "webqoof",
)

FRAMING_MARKERS = (
    "fact check",
    "fact-check",
    "fake news",
    "misinformation",
    "debunked",
)


class BoilerplateMasker(BaseEstimator, TransformerMixin):
    """Mask known outlet and optionally fact-check framing phrases."""

    def __init__(self, *, mask_sources: bool = True, mask_framing: bool = False) -> None:
        self.mask_sources = mask_sources
        self.mask_framing = mask_framing

    def fit(self, documents: Iterable[str], y: Any = None) -> "BoilerplateMasker":
        del documents, y
        return self

    def transform(self, documents: Iterable[str]) -> list[str]:
        phrases: tuple[str, ...] = ()
        if self.mask_sources:
            phrases += SOURCE_MARKERS
        if self.mask_framing:
            phrases += FRAMING_MARKERS
        pattern = None
        if phrases:
            escaped = sorted((re.escape(value) for value in phrases), key=len, reverse=True)
            pattern = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)
        output = []
        for document in documents:
            text = str(document)
            text = re.sub(r"https?://\S+|www\.\S+", " SOURCE_URL ", text, flags=re.IGNORECASE)
            if pattern is not None:
                text = pattern.sub(" SOURCE_MARKER ", text)
            output.append(text)
        return output


@dataclass(frozen=True)
class RobustRepresentationConfig:
    representation: str = "word_char"
    mask_sources: bool = True
    mask_framing: bool = False
    word_max_features: int = 80_000
    char_max_features: int = 80_000
    min_df: int = 2
    max_df: float = 0.98
    logistic_c: float = 2.0
    svm_c: float = 1.0
    random_forest_estimators: int = 200
    random_forest_criterion: str = "gini"
    random_forest_max_depth: int | None = None
    random_forest_min_samples_leaf: int = 2
    random_forest_max_features: str | float = "sqrt"
    random_forest_n_jobs: int = -1
    random_state: int = 20_261_017

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "mask_sources": self.mask_sources,
            "mask_framing": self.mask_framing,
            "word_max_features": self.word_max_features,
            "char_max_features": self.char_max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "logistic_c": self.logistic_c,
            "svm_c": self.svm_c,
            "random_forest_estimators": self.random_forest_estimators,
            "random_forest_criterion": self.random_forest_criterion,
            "random_forest_max_depth": self.random_forest_max_depth,
            "random_forest_min_samples_leaf": self.random_forest_min_samples_leaf,
            "random_forest_max_features": self.random_forest_max_features,
            "random_forest_n_jobs": self.random_forest_n_jobs,
            "random_state": self.random_state,
        }


def build_robust_pipeline(model_name: str, config: RobustRepresentationConfig) -> Pipeline:
    word = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.word_max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.char_max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    if config.representation == "word":
        features: Any = word
    elif config.representation == "char":
        features = char
    elif config.representation == "word_char":
        features = FeatureUnion([("word", word), ("char", char)], transformer_weights={"word": 1.0, "char": 0.5})
    else:
        raise ValueError("representation must be one of: word, char, word_char")

    if model_name == "logistic_regression":
        classifier: Any = LogisticRegression(
            C=config.logistic_c,
            class_weight=None,
            max_iter=1_000,
            random_state=config.random_state,
            solver="liblinear",
        )
    elif model_name == "linear_svm":
        classifier = CalibratedClassifierCV(
            estimator=LinearSVC(C=config.svm_c, class_weight=None, random_state=config.random_state),
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
            class_weight=None,
            n_jobs=config.random_forest_n_jobs,
            random_state=config.random_state,
        )
    else:
        raise ValueError(
            "Only logistic_regression, linear_svm, and random_forest are supported"
        )

    return Pipeline(
        [
            ("masker", BoilerplateMasker(mask_sources=config.mask_sources, mask_framing=config.mask_framing)),
            ("features", features),
            ("classifier", classifier),
        ]
    )
