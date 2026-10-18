"""Validated loading and inference for the approved classical models.

The service deliberately describes its output as a *content-classification
signal*.  It does not turn a text classifier into a factual-verification
system, and it exposes disagreement and low-margin predictions instead of
hiding them behind a forced binary answer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np

from backend.ml.classical_model import SUPPORTED_MODELS, fake_probabilities


DEFAULT_MODEL_SET = "multidataset_v2_robust"
MODEL_WARNING = (
    "Content-classification signal only; this is not evidence-based fact "
    "verification and must not be presented as proof that a claim is true or false."
)


class ClassicalArtifactError(RuntimeError):
    """Raised when a local model set is missing, malformed, or untrusted."""


@dataclass(frozen=True)
class ClassicalInput:
    """One title/body pair accepted by the classical inference service."""

    title: str = ""
    text: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClassicalInput":
        return cls(title=str(value.get("title", "")), text=str(value.get("text", "")))

    def document(self) -> str:
        return f"{self.title.strip()}\n\n{self.text.strip()}".strip()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ClassicalModelRegistry:
    """Load only manifest-declared, checksum-verified local model artifacts.

    ``joblib`` artifacts are pickle-based.  A matching digest proves integrity
    relative to the local manifest, not that an unknown manifest is safe.  The
    registry must therefore be pointed only at this project's trusted model
    outputs.
    """

    def __init__(
        self,
        repository_root: Path,
        model_set: str = DEFAULT_MODEL_SET,
    ) -> None:
        self.repository_root = repository_root.resolve()
        if not model_set or Path(model_set).name != model_set:
            raise ClassicalArtifactError("model_set must be a single directory name")
        self.model_set = model_set
        self.model_set_dir = (
            self.repository_root / "models" / "classical" / model_set
        ).resolve()
        self.benchmark_path = self.model_set_dir / "benchmark.json"
        self._manifest = self._read_and_validate_manifest()
        self._cache: dict[str, Any] = {}

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a defensive copy of the validated manifest."""

        return json.loads(json.dumps(self._manifest))

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(self._manifest["models"])

    @property
    def selected_model(self) -> str:
        return str(self._manifest["selected_model"])

    def threshold(self, model_name: str) -> float:
        self._validate_model_name(model_name)
        return float(
            self._manifest["validation"][model_name]["selected_threshold"]["threshold"]
        )

    def artifact_path(self, model_name: str) -> Path:
        self._validate_model_name(model_name)
        raw_path = Path(self._manifest["artifacts"][model_name]["path"])
        path = (self.repository_root / raw_path).resolve()
        try:
            path.relative_to(self.model_set_dir)
        except ValueError as exc:
            raise ClassicalArtifactError(
                f"Artifact for {model_name} escapes the configured model-set directory"
            ) from exc
        return path

    def load(self, model_name: str) -> Any:
        """Verify and lazily load a project-generated model pipeline."""

        self._validate_model_name(model_name)
        if model_name in self._cache:
            return self._cache[model_name]
        path = self.artifact_path(model_name)
        if not path.is_file():
            raise ClassicalArtifactError(f"Model artifact is missing: {path}")
        expected = self._manifest["artifacts"][model_name]["sha256"].lower()
        actual = sha256_file(path)
        if actual != expected:
            raise ClassicalArtifactError(
                f"Checksum mismatch for {model_name}: expected {expected}, got {actual}"
            )
        try:
            model = joblib.load(path)
        except Exception as exc:  # pragma: no cover - exact joblib errors vary
            raise ClassicalArtifactError(f"Could not load {model_name}: {exc}") from exc
        if not hasattr(model, "predict_proba"):
            raise ClassicalArtifactError(f"{model_name} artifact has no predict_proba method")
        self._cache[model_name] = model
        return model

    def describe(self) -> dict[str, Any]:
        """Return non-secret, JSON-safe model provenance."""

        return {
            "model_set": self.model_set,
            "created_at_utc": self._manifest.get("created_at_utc"),
            "models": list(self.model_names),
            "selected_model": self.selected_model,
            "thresholds": {name: self.threshold(name) for name in self.model_names},
            "artifact_sha256": {
                name: self._manifest["artifacts"][name]["sha256"]
                for name in self.model_names
            },
            "training_datasets": list(self._manifest.get("datasets", [])),
            "selection_metric": self._manifest.get("selection_metric"),
        }

    def _validate_model_name(self, model_name: str) -> None:
        if model_name not in self.model_names:
            raise KeyError(f"Unknown model {model_name!r}; expected one of {self.model_names}")

    def _read_and_validate_manifest(self) -> dict[str, Any]:
        if not self.benchmark_path.is_file():
            raise ClassicalArtifactError(f"Model manifest is missing: {self.benchmark_path}")
        try:
            value = json.loads(self.benchmark_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ClassicalArtifactError(f"Invalid model manifest: {exc}") from exc
        if not isinstance(value, dict):
            raise ClassicalArtifactError("Model manifest must contain a JSON object")
        if value.get("model_set") != self.model_set:
            raise ClassicalArtifactError("Manifest model_set does not match its directory")
        models = value.get("models")
        if (
            not isinstance(models, list)
            or not models
            or not set(models).issubset(SUPPORTED_MODELS)
            or len(models) != len(set(models))
        ):
            raise ClassicalArtifactError(
                f"Manifest contains unsupported or duplicate models; approved={SUPPORTED_MODELS}"
            )
        if value.get("selected_model") not in models:
            raise ClassicalArtifactError("Manifest selected_model is not declared")
        for name in models:
            try:
                artifact = value["artifacts"][name]
                digest = artifact["sha256"]
                threshold = value["validation"][name]["selected_threshold"]["threshold"]
            except (KeyError, TypeError) as exc:
                raise ClassicalArtifactError(f"Manifest entry for {name} is incomplete") from exc
            if not isinstance(artifact.get("path"), str):
                raise ClassicalArtifactError(f"Artifact path for {name} is invalid")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ClassicalArtifactError(f"SHA-256 digest for {name} is invalid")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise ClassicalArtifactError(f"SHA-256 digest for {name} is invalid") from exc
            if not isinstance(threshold, (int, float)) or not 0.0 < float(threshold) < 1.0:
                raise ClassicalArtifactError(f"Threshold for {name} must be between zero and one")
        return value


class ClassicalMLService:
    """Run model-wise and preferred-model classical text analysis."""

    def __init__(
        self,
        registry: ClassicalModelRegistry,
        *,
        abstention_margin: float = 0.08,
        max_characters: int = 200_000,
        max_batch_size: int = 100,
    ) -> None:
        if not 0.0 <= abstention_margin < 0.5:
            raise ValueError("abstention_margin must be in [0, 0.5)")
        if max_characters < 1 or max_batch_size < 1:
            raise ValueError("Input limits must be positive")
        self.registry = registry
        self.abstention_margin = abstention_margin
        self.max_characters = max_characters
        self.max_batch_size = max_batch_size

    def analyze(
        self,
        item: ClassicalInput | Mapping[str, Any],
        *,
        include_indicators: bool = True,
        top_k: int = 8,
    ) -> dict[str, Any]:
        """Analyze one document with both approved models."""

        if isinstance(item, Mapping):
            item = ClassicalInput.from_mapping(item)
        if not isinstance(item, ClassicalInput):
            raise TypeError("item must be ClassicalInput or a title/text mapping")
        document = self._validate_document(item.document())
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")

        model_outputs: dict[str, dict[str, Any]] = {}
        raw_classes: set[str] = set()
        for name in self.registry.model_names:
            model = self.registry.load(name)
            probability = float(fake_probabilities(model, [document])[0])
            threshold = self.registry.threshold(name)
            raw_class = "fake" if probability >= threshold else "real"
            raw_classes.add(raw_class)
            decision_margin = abs(probability - threshold)
            state = "uncertain" if decision_margin < self.abstention_margin else raw_class
            output: dict[str, Any] = {
                "model": name,
                "fake_probability": probability,
                "threshold": threshold,
                "predicted_class": raw_class,
                "classification": state,
                "predicted_class_probability": (
                    probability if raw_class == "fake" else 1.0 - probability
                ),
                "max_class_probability": max(probability, 1.0 - probability),
                "decision_margin": decision_margin,
            }
            if include_indicators:
                output["text_indicators"] = _linear_text_indicators(model, document, top_k)
            model_outputs[name] = output

        selected = self.registry.selected_model
        preferred = model_outputs[selected]
        agreement = len(raw_classes) == 1
        final_classification = preferred["classification"] if agreement else "uncertain"
        uncertainty_reasons: list[str] = []
        if not agreement:
            uncertainty_reasons.append("model_disagreement")
        if preferred["classification"] == "uncertain":
            uncertainty_reasons.append("preferred_model_low_margin")

        return {
            "task": "classical_content_classification",
            "classification": final_classification,
            "preferred_model": selected,
            "preferred_raw_prediction": preferred["predicted_class"],
            "fake_probability": preferred["fake_probability"],
            "predicted_class_probability": preferred["predicted_class_probability"],
            "max_class_probability": preferred["max_class_probability"],
            "agreement": agreement,
            "uncertainty_reasons": uncertainty_reasons,
            "models": model_outputs,
            "input": {
                "sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
                "characters": len(document),
                "words": len(document.split()),
            },
            "model_provenance": self.registry.describe(),
            "warning": MODEL_WARNING,
        }

    def analyze_batch(
        self,
        items: Iterable[ClassicalInput | Mapping[str, Any]],
        *,
        include_indicators: bool = False,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """Analyze a bounded batch while preserving input order."""

        batch = list(items)
        if not batch:
            raise ValueError("Batch must contain at least one item")
        if len(batch) > self.max_batch_size:
            raise ValueError(f"Batch exceeds the limit of {self.max_batch_size}")
        return [
            self.analyze(item, include_indicators=include_indicators, top_k=top_k)
            for item in batch
        ]

    def _validate_document(self, document: str) -> str:
        if not document:
            raise ValueError("A non-empty title or article body is required")
        if len(document) > self.max_characters:
            raise ValueError(f"Input exceeds the {self.max_characters}-character limit")
        return document


def _linear_coefficients(classifier: Any) -> np.ndarray | None:
    """Extract linear coefficients, averaging calibrated SVM folds when needed."""

    if hasattr(classifier, "coef_"):
        return np.asarray(classifier.coef_)[0]
    calibrated = getattr(classifier, "calibrated_classifiers_", None)
    if calibrated:
        coefficients = []
        for fitted in calibrated:
            estimator = getattr(fitted, "estimator", None)
            if estimator is not None and hasattr(estimator, "coef_"):
                coefficients.append(np.asarray(estimator.coef_)[0])
        if coefficients:
            return np.mean(np.stack(coefficients), axis=0)
    return None


def _linear_text_indicators(model: Any, document: str, top_k: int) -> dict[str, Any]:
    """Return local TF-IDF coefficient products as non-causal model indicators."""

    named_steps = getattr(model, "named_steps", {})
    features = named_steps.get("features")
    classifier = named_steps.get("classifier")
    if features is None or classifier is None or not hasattr(features, "get_feature_names_out"):
        return {"available": False, "reason": "pipeline_does_not_expose_linear_features"}
    masked_documents: Any = [document]
    masker = named_steps.get("masker")
    if masker is not None:
        masked_documents = masker.transform(masked_documents)
    matrix = features.transform(masked_documents)
    row = matrix.getrow(0)
    names = features.get_feature_names_out()
    coefficients = _linear_coefficients(classifier)
    if coefficients is None:
        importances = getattr(classifier, "feature_importances_", None)
        if importances is None or matrix.shape[1] != len(importances):
            return {"available": False, "reason": "feature_attribution_unavailable"}
        ranked = sorted(
            (
                (str(names[index]), float(weight * importances[index]))
                for index, weight in zip(row.indices, row.data, strict=True)
                if importances[index] > 0
            ),
            key=lambda value: value[1],
            reverse=True,
        )[:top_k]
        return {
            "available": True,
            "method": "active_tfidf_value_times_random_forest_global_importance",
            "interpretation": (
                "Active features ranked by global forest importance; direction, causation, "
                "and factual support cannot be inferred."
            ),
            "active_important_features": [
                {"feature": feature, "importance": importance}
                for feature, importance in ranked
            ],
        }
    if matrix.shape[1] != len(coefficients):
        return {"available": False, "reason": "linear_coefficients_unavailable"}

    values = [
        (str(names[index]), float(weight * coefficients[index]))
        for index, weight in zip(row.indices, row.data, strict=True)
    ]
    fake = sorted((item for item in values if item[1] > 0), key=lambda x: x[1], reverse=True)
    real = sorted((item for item in values if item[1] < 0), key=lambda x: x[1])
    return {
        "available": True,
        "method": "tfidf_value_times_linear_coefficient",
        "interpretation": "Associations learned from training text; not factual evidence or causation.",
        "toward_fake": [
            {"feature": feature, "contribution": contribution}
            for feature, contribution in fake[:top_k]
        ],
        "toward_real": [
            {"feature": feature, "contribution": contribution}
            for feature, contribution in real[:top_k]
        ],
    }
