from __future__ import annotations

import pandas as pd

from scripts.prepare_datasets import (
    assign_grouped_splits,
    canonical_frame,
    clean_text,
    enforce_official_split_clusters,
    prepare_text_frame,
)


def test_clean_text_normalizes_markup_unicode_and_whitespace() -> None:
    value = "\ufeff  <p>News&nbsp;item</p>\u200b\n\n\nSecond\tline  "
    assert clean_text(value) == "News item\n\nSecond line"


def test_conflicting_exact_text_is_quarantined() -> None:
    frame = canonical_frame(
        [
            {"dataset": "fixture", "record_id": "1", "text_original": "Same claim!", "label_raw": "FAKE"},
            {"dataset": "fixture", "record_id": "2", "text_original": "same claim", "label_raw": "REAL"},
            {"dataset": "fixture", "record_id": "3", "text_original": "Distinct claim", "label_raw": "REAL"},
        ]
    )
    prepared, exclusions, metrics = prepare_text_frame(frame)
    assert prepared["record_id"].tolist() == ["3"]
    assert metrics["conflicting_records_excluded"] == 2
    assert {row["reason"] for row in exclusions} == {"conflicting_labels_same_normalized_text"}


def test_grouped_split_is_deterministic_and_cluster_safe() -> None:
    frame = pd.DataFrame(
        {
            "near_cluster_id": ["a", "a", "b", "c", "d", "e", "f", "g"],
            "label_raw": ["0", "0", "0", "0", "1", "1", "1", "1"],
        }
    )
    first = assign_grouped_splits(frame, ("train", "validation", "test"), (0.5, 0.25, 0.25), 7)
    second = assign_grouped_splits(frame, ("train", "validation", "test"), (0.5, 0.25, 0.25), 7)
    assert first.tolist() == second.tolist()
    assigned = frame.assign(split=first)
    assert (assigned.groupby("near_cluster_id")["split"].nunique() == 1).all()


def test_official_split_keeps_evaluation_side_of_near_duplicate_cluster() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["fixture"] * 3,
            "record_id": ["train-copy", "test-copy", "train-distinct"],
            "text_fingerprint": ["a", "b", "c"],
            "label_raw": ["0", "0", "1"],
            "split_original": ["train", "test", "train"],
            "near_cluster_id": ["near-a", "near-a", "near-c"],
        }
    )
    prepared, exclusions, removed = enforce_official_split_clusters(frame)
    assert removed == 1
    assert set(prepared["record_id"]) == {"test-copy", "train-distinct"}
    assert exclusions[0]["reason"] == "near_duplicate_cross_split"
