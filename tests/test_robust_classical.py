from __future__ import annotations

import pandas as pd
import pytest

from backend.ml.robust_classical_model import (
    BoilerplateMasker,
    RobustRepresentationConfig,
    build_robust_pipeline,
)
from scripts.train_classical_robust import robust_weights


def test_boilerplate_masker_masks_sources_but_preserves_framing_by_default() -> None:
    masker = BoilerplateMasker(mask_sources=True, mask_framing=False)
    transformed = masker.fit_transform(
        ["Reuters published a fact check at https://example.com/item"]
    )[0]
    assert "Reuters" not in transformed
    assert "https://" not in transformed
    assert "fact check" in transformed
    assert "SOURCE_MARKER" in transformed


def test_robust_pipeline_supports_only_approved_models() -> None:
    config = RobustRepresentationConfig(representation="word", word_max_features=100)
    assert build_robust_pipeline("logistic_regression", config).named_steps["classifier"] is not None
    assert build_robust_pipeline("linear_svm", config).named_steps["classifier"] is not None
    with pytest.raises(ValueError, match="Only logistic_regression and linear_svm"):
        build_robust_pipeline("random_forest", config)


def test_liar_extremes_receive_quarter_dataset_weight() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["core", "core", "liar_extremes", "liar_extremes"],
            "label_binary_fake": [0, 1, 0, 1],
        }
    )
    weights = robust_weights(frame)
    core_total = weights[frame["dataset"].eq("core").to_numpy()].sum()
    liar_total = weights[frame["dataset"].eq("liar_extremes").to_numpy()].sum()
    assert liar_total / core_total == pytest.approx(0.25)
