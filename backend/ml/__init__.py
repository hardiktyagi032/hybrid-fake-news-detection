"""Classical and transformer model components."""

from backend.ml.classical_model import (
    ClassicalModelConfig,
    build_pipeline,
    combine_text_fields,
    evaluate_binary_classifier,
)
from backend.ml.classical_service import (
    ClassicalArtifactError,
    ClassicalInput,
    ClassicalMLService,
    ClassicalModelRegistry,
)

__all__ = [
    "ClassicalModelConfig",
    "build_pipeline",
    "combine_text_fields",
    "evaluate_binary_classifier",
    "ClassicalArtifactError",
    "ClassicalInput",
    "ClassicalMLService",
    "ClassicalModelRegistry",
]
