"""Audit the multi-dataset classical models for leakage and robustness."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from backend.ml.classical_model import (
    ClassicalModelConfig,
    build_pipeline,
    combine_text_fields,
    fake_probabilities,
)
from scripts.prepare_datasets import near_duplicate_clusters
from scripts.train_classical_multidataset import (
    DEFAULT_DATASETS,
    MODEL_SET_NAME,
    dataset_class_balanced_weights,
)


AUDIT_COLUMNS = [
    "dataset",
    "record_id",
    "title_clean",
    "text_clean",
    "text_fingerprint",
    "label_binary_fake",
    "url",
    "source",
    "date",
    "split",
]


def load_audit_data(root: Path, datasets: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset in datasets:
        for split in ("train", "validation", "test"):
            path = root / "data" / "processed" / dataset / "splits" / f"{split}.csv"
            frame = pd.read_csv(path, usecols=AUDIT_COLUMNS, keep_default_na=False)
            if not frame["dataset"].eq(dataset).all() or not frame["split"].eq(split).all():
                raise ValueError(f"Unexpected dataset or split marker in {path}")
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["label_binary_fake"] = result["label_binary_fake"].astype(int)
    return result


def publisher_key(row: pd.Series) -> str:
    """Derive a coarse source key without placing it in the model features."""

    source = str(row.get("source", "")).strip().casefold()
    raw_url = str(row.get("url", "")).strip().casefold()
    if row["dataset"] == "bharatgather" and source:
        return re.sub(r"[^a-z0-9]+", "_", source).strip("_")
    if raw_url:
        candidate = raw_url if "://" in raw_url else f"//{raw_url}"
        parsed = urlparse(candidate)
        host = parsed.hostname or parsed.path.split("/")[0]
        host = re.sub(r"^www\.", "", host)
        key = re.sub(r"[^a-z0-9]+", "_", host).strip("_")
        if key:
            return key
    return re.sub(r"[^a-z0-9]+", "_", source).strip("_")


def exact_leakage(frame: pd.DataFrame) -> dict[str, Any]:
    grouped = frame.groupby("text_fingerprint", sort=False)
    cross_split = grouped["split"].nunique()
    cross_dataset = grouped["dataset"].nunique()
    leaked_fingerprints = set(cross_split[cross_split > 1].index)
    leaked_test = frame[
        frame["text_fingerprint"].isin(leaked_fingerprints) & frame["split"].eq("test")
    ]
    examples = []
    for fingerprint in list(sorted(leaked_fingerprints))[:20]:
        group = frame[frame["text_fingerprint"].eq(fingerprint)]
        examples.append(
            {
                "fingerprint": fingerprint,
                "datasets": sorted(group["dataset"].unique().tolist()),
                "splits": sorted(group["split"].unique().tolist()),
                "records": int(len(group)),
            }
        )
    return {
        "cross_split_fingerprints": int((cross_split > 1).sum()),
        "cross_dataset_fingerprints": int((cross_dataset > 1).sum()),
        "test_records_leaking_exactly_into_development": int(len(leaked_test)),
        "examples": examples,
    }


def near_duplicate_leakage(frame: pd.DataFrame) -> dict[str, Any]:
    cluster_ids, method_metrics = near_duplicate_clusters(
        frame["text_clean"].astype(str).tolist(),
        frame["text_fingerprint"].astype(str).tolist(),
    )
    working = frame[["dataset", "record_id", "split", "text_fingerprint"]].copy()
    working["global_near_cluster_id"] = cluster_ids
    grouped = working.groupby("global_near_cluster_id", sort=False)
    stats = grouped.agg(
        records=("record_id", "size"),
        split_count=("split", "nunique"),
        dataset_count=("dataset", "nunique"),
        fingerprint_count=("text_fingerprint", "nunique"),
    )
    cross_split_ids = set(
        stats[(stats["records"] > 1) & (stats["split_count"] > 1) & (stats["fingerprint_count"] > 1)].index
    )
    cross_dataset_ids = set(
        stats[(stats["records"] > 1) & (stats["dataset_count"] > 1) & (stats["fingerprint_count"] > 1)].index
    )
    leaked_test = working[
        working["global_near_cluster_id"].isin(cross_split_ids) & working["split"].eq("test")
    ]
    examples = []
    for cluster_id in list(sorted(cross_split_ids))[:20]:
        group = working[working["global_near_cluster_id"].eq(cluster_id)]
        examples.append(
            {
                "cluster_id": cluster_id,
                "datasets": sorted(group["dataset"].unique().tolist()),
                "splits": sorted(group["split"].unique().tolist()),
                "records": int(len(group)),
                "distinct_normalized_texts": int(group["text_fingerprint"].nunique()),
            }
        )
    return {
        **method_metrics,
        "cross_split_near_duplicate_clusters": len(cross_split_ids),
        "cross_dataset_near_duplicate_clusters": len(cross_dataset_ids),
        "test_records_near_duplicate_of_development": int(len(leaked_test)),
        "examples": examples,
    }


def metadata_shortcut_audit(frame: pd.DataFrame) -> dict[str, Any]:
    working = frame.copy()
    working["publisher_key"] = working.apply(publisher_key, axis=1)
    reports: dict[str, Any] = {}
    for dataset, dataset_frame in working.groupby("dataset", sort=True):
        train = dataset_frame[dataset_frame["split"].eq("train")]
        test = dataset_frame[dataset_frame["split"].eq("test")]
        usable_train = train[train["publisher_key"].ne("")]
        usable_test = test[test["publisher_key"].ne("")]
        if usable_train["publisher_key"].nunique() < 2 or usable_test.empty:
            reports[str(dataset)] = {
                "status": "not_evaluable",
                "reason": "fewer than two usable publisher/source groups",
            }
            continue
        global_label = int(usable_train["label_binary_fake"].mean() >= 0.5)
        mapping = (
            usable_train.groupby("publisher_key")["label_binary_fake"]
            .mean()
            .map(lambda value: int(value >= 0.5))
            .to_dict()
        )
        prediction = usable_test["publisher_key"].map(mapping).fillna(global_label).astype(int)
        labels = usable_test["label_binary_fake"].astype(int)
        seen = usable_test["publisher_key"].isin(mapping)
        group_rates = usable_train.groupby("publisher_key")["label_binary_fake"].agg(["size", "mean"])
        reports[str(dataset)] = {
            "status": "evaluated",
            "train_groups": int(usable_train["publisher_key"].nunique()),
            "test_groups": int(usable_test["publisher_key"].nunique()),
            "test_records_with_seen_group": int(seen.sum()),
            "test_seen_group_ratio": float(seen.mean()),
            "metadata_only_accuracy": float(accuracy_score(labels, prediction)),
            "metadata_only_macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
            "near_pure_train_groups": int(((group_rates["mean"] <= 0.05) | (group_rates["mean"] >= 0.95)).sum()),
        }
    return reports


def threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "real_precision": float(precision[0]),
        "real_recall": float(recall[0]),
        "real_f1": float(f1[0]),
        "real_support": int(support[0]),
        "fake_precision": float(precision[1]),
        "fake_recall": float(recall[1]),
        "fake_f1": float(f1[1]),
        "fake_support": int(support[1]),
    }


def threshold_and_error_audit(root: Path, frame: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}
    output_dir = root / "models" / "classical" / MODEL_SET_NAME / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = np.round(np.arange(0.20, 0.801, 0.01), 2)
    benchmark_path = root / "models" / "classical" / MODEL_SET_NAME / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    config = ClassicalModelConfig(**benchmark["config"])
    train = frame[frame["split"].eq("train")].copy()
    train_labels = train["label_binary_fake"].astype(int).to_numpy()
    train_weights = dataset_class_balanced_weights(train)
    for model_name in ("logistic_regression", "linear_svm", "random_forest"):
        # The persisted final pipeline was refit on train + validation. Rebuild a
        # train-only candidate so validation remains genuinely unseen here.
        model = build_pipeline(model_name, config)
        model.fit(
            combine_text_fields(train),
            train_labels,
            classifier__sample_weight=train_weights,
        )
        split_cache: dict[str, dict[str, Any]] = {}
        for split in ("validation", "test"):
            current = frame[frame["split"].eq(split)].copy()
            split_cache[split] = {
                "frame": current,
                "labels": current["label_binary_fake"].astype(int).to_numpy(),
                "probabilities": fake_probabilities(model, combine_text_fields(current)),
            }
        validation_rows = []
        for threshold in thresholds:
            dataset_scores = []
            for _, group in split_cache["validation"]["frame"].groupby("dataset", sort=True):
                indexes = group.index.to_numpy()
                positions = split_cache["validation"]["frame"].index.get_indexer(indexes)
                metrics = threshold_metrics(
                    group["label_binary_fake"].astype(int).to_numpy(),
                    split_cache["validation"]["probabilities"][positions],
                    float(threshold),
                )
                dataset_scores.append(metrics["macro_f1"])
            validation_rows.append(
                {
                    "threshold": float(threshold),
                    "mean_dataset_macro_f1": float(np.mean(dataset_scores)),
                    "worst_dataset_macro_f1": float(np.min(dataset_scores)),
                }
            )
        selected = max(
            validation_rows,
            key=lambda row: (row["mean_dataset_macro_f1"], row["worst_dataset_macro_f1"], -abs(row["threshold"] - 0.5)),
        )
        test_frame = split_cache["test"]["frame"].copy()
        test_frame["fake_probability"] = split_cache["test"]["probabilities"]
        test_frame["prediction"] = (test_frame["fake_probability"] >= selected["threshold"]).astype(int)
        test_frame["confidence"] = np.maximum(test_frame["fake_probability"], 1 - test_frame["fake_probability"])
        mistakes = test_frame[test_frame["prediction"].ne(test_frame["label_binary_fake"])].copy()
        mistakes = mistakes.sort_values("confidence", ascending=False)
        mistakes["text_excerpt"] = mistakes["text_clean"].astype(str).str.slice(0, 500)
        error_columns = [
            "dataset", "record_id", "label_binary_fake", "prediction", "fake_probability",
            "confidence", "source", "url", "date", "text_excerpt",
        ]
        mistakes[error_columns].head(200).to_csv(output_dir / f"{model_name}_errors.csv", index=False)
        per_dataset_test = {}
        for dataset, group in test_frame.groupby("dataset", sort=True):
            per_dataset_test[str(dataset)] = threshold_metrics(
                group["label_binary_fake"].astype(int).to_numpy(),
                group["fake_probability"].to_numpy(),
                float(selected["threshold"]),
            )
        report[model_name] = {
            "fit_scope": "train_only_for_unbiased_validation_and_test_audit",
            "validation_threshold_selection": selected,
            "validation_threshold_sweep": validation_rows,
            "test_at_selected_threshold": per_dataset_test,
            "test_mistakes": int(len(mistakes)),
            "high_confidence_test_mistakes_at_least_0_9": int((mistakes["confidence"] >= 0.9).sum()),
            "error_sample_path": str((output_dir / f"{model_name}_errors.csv").relative_to(root)).replace("\\", "/"),
        }
    return report


def logistic_feature_audit(root: Path) -> dict[str, Any]:
    path = root / "models" / "classical" / MODEL_SET_NAME / "logistic_regression" / "pipeline.joblib"
    model = joblib.load(path)
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    names = np.asarray(vectorizer.get_feature_names_out())
    coefficients = np.asarray(classifier.coef_[0])
    top_fake = np.argsort(coefficients)[-50:][::-1]
    top_real = np.argsort(coefficients)[:50]
    return {
        "top_fake_features": [
            {"feature": str(names[index]), "coefficient": float(coefficients[index])} for index in top_fake
        ],
        "top_real_features": [
            {"feature": str(names[index]), "coefficient": float(coefficients[index])} for index in top_real
        ],
    }


def run_audit(root: Path, datasets: list[str]) -> dict[str, Any]:
    root = root.resolve()
    frame = load_audit_data(root, datasets)
    report = {
        "status": "completed",
        "datasets": datasets,
        "records": int(len(frame)),
        "exact_duplicate_leakage": exact_leakage(frame),
        "near_duplicate_leakage": near_duplicate_leakage(frame),
        "publisher_source_shortcuts": metadata_shortcut_audit(frame),
        "thresholds_and_errors": threshold_and_error_audit(root, frame),
        "selected_logistic_features": logistic_feature_audit(root),
        "notes": [
            "Publisher/source metadata is audited but is not included as a model feature.",
            "Thresholds are selected only on validation data; test metrics are reported afterward.",
            "A high score does not establish factual correctness or source credibility.",
        ],
    }
    output = root / "models" / "classical" / MODEL_SET_NAME / "audit" / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    args = parser.parse_args()
    result = run_audit(args.root, args.datasets)
    summary = {
        "records": result["records"],
        "exact_duplicate_leakage": result["exact_duplicate_leakage"],
        "near_duplicate_leakage": result["near_duplicate_leakage"],
        "publisher_source_shortcuts": result["publisher_source_shortcuts"],
        "threshold_selection": {
            name: values["validation_threshold_selection"]
            for name, values in result["thresholds_and_errors"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
