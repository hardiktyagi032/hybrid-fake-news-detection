from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.train_classical_multidataset import dataset_class_balanced_weights


def test_dataset_class_weights_equalize_group_influence() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["large"] * 6 + ["small"] * 4,
            "label_binary_fake": [0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
        }
    )
    weights = dataset_class_balanced_weights(frame)
    weighted = frame.assign(weight=weights).groupby(["dataset", "label_binary_fake"])["weight"].sum()
    assert np.allclose(weighted.to_numpy(), weighted.iloc[0])
    assert weights.mean() == pytest.approx(1.0)


def test_dataset_class_weights_require_both_labels_per_dataset() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["complete", "complete", "incomplete"],
            "label_binary_fake": [0, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="both real and fake"):
        dataset_class_balanced_weights(frame)
