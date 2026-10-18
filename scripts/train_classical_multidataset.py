"""Train the dataset-balanced approved classical classifiers."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from backend.ml.classical_model import (
    SUPPORTED_MODELS,
    ClassicalModelConfig,
    build_pipeline,
    combine_text_fields,
    evaluate_binary_classifier,
    validate_binary_labels,
)
from scripts.train_classical import sha256_file, validate_dataset_manifest


DEFAULT_DATASETS = ("isot", "ifnd", "indian_fake_news", "bharatgather")
MODEL_SET_NAME = "multidataset_v1"
TRAINING_COLUMNS = ["dataset", "title_clean", "text_clean", "label_binary_fake", "split"]


def load_combined_split(root: Path, datasets: list[str], split: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset in datasets:
        validate_dataset_manifest(root, dataset)
        path = root / "data" / "processed" / dataset / "splits" / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Prepared split not found: {path}")
        frame = pd.read_csv(
            path,
            usecols=TRAINING_COLUMNS,
            dtype="string",
            keep_default_na=False,
        )
        if not frame["dataset"].eq(dataset).all():
            raise ValueError(f"Unexpected dataset identifier in {path}")
        if not frame["split"].eq(split).all():
            raise ValueError(f"Unexpected split identifier in {path}")
        validate_binary_labels(frame["label_binary_fake"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def dataset_class_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give every dataset/class pair equal total influence, with mean weight one."""

    required = {"dataset", "label_binary_fake"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing weighting columns: {sorted(missing)}")
    labels = validate_binary_labels(frame["label_binary_fake"])
    keys = pd.DataFrame(
        {
            "dataset": frame["dataset"].astype(str).to_numpy(),
            "label": labels,
        }
    )
    group_sizes = keys.groupby(["dataset", "label"]).size()
    expected_groups = len(keys["dataset"].unique()) * 2
    if len(group_sizes) != expected_groups:
        raise ValueError("Every dataset must contain both real and fake labels")
    weights = np.asarray(
        [1.0 / float(group_sizes.loc[(dataset, label)]) for dataset, label in keys.itertuples(index=False)],
        dtype=float,
    )
    return weights / weights.mean()


def evaluate_by_dataset(model: Any, frame: pd.DataFrame) -> dict[str, Any]:
    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset, group in frame.groupby("dataset", sort=True):
        per_dataset[str(dataset)] = evaluate_binary_classifier(
            model,
            combine_text_fields(group),
            group["label_binary_fake"],
        )
    macro_f1_values = [metrics["f1_macro"] for metrics in per_dataset.values()]
    balanced_accuracy_values = [metrics["balanced_accuracy"] for metrics in per_dataset.values()]
    return {
        "selection_score_mean_dataset_macro_f1": float(np.mean(macro_f1_values)),
        "mean_dataset_balanced_accuracy": float(np.mean(balanced_accuracy_values)),
        "worst_dataset_macro_f1": float(np.min(macro_f1_values)),
        "pooled": evaluate_binary_classifier(
            model,
            combine_text_fields(frame),
            frame["label_binary_fake"],
        ),
        "per_dataset": per_dataset,
    }


def run_training(
    root: Path,
    datasets: list[str],
    model_names: list[str],
    config: ClassicalModelConfig,
) -> dict[str, Any]:
    root = root.resolve()
    unknown_models = sorted(set(model_names).difference(SUPPORTED_MODELS))
    if unknown_models:
        raise ValueError(f"Unsupported models: {unknown_models}")
    if len(datasets) < 2 or len(datasets) != len(set(datasets)):
        raise ValueError("Provide at least two unique datasets")

    manifest_path = root / "data" / "processed" / "manifest.json"
    train = load_combined_split(root, datasets, "train")
    validation = load_combined_split(root, datasets, "validation")
    test = load_combined_split(root, datasets, "test")
    train_text = combine_text_fields(train)
    train_labels = validate_binary_labels(train["label_binary_fake"])
    train_weights = dataset_class_balanced_weights(train)

    validation_metrics: dict[str, dict[str, Any]] = {}
    for model_name in model_names:
        print(f"Training dataset-balanced candidate: {model_name}", flush=True)
        model = build_pipeline(model_name, config)
        model.fit(train_text, train_labels, classifier__sample_weight=train_weights)
        validation_metrics[model_name] = evaluate_by_dataset(model, validation)
        score = validation_metrics[model_name]["selection_score_mean_dataset_macro_f1"]
        print(f"Validation mean dataset macro-F1 {model_name}: {score:.6f}", flush=True)

    selected_model = max(
        model_names,
        key=lambda name: (
            validation_metrics[name]["selection_score_mean_dataset_macro_f1"],
            validation_metrics[name]["worst_dataset_macro_f1"],
        ),
    )

    development = pd.concat([train, validation], ignore_index=True)
    development_text = combine_text_fields(development)
    development_labels = validate_binary_labels(development["label_binary_fake"])
    development_weights = dataset_class_balanced_weights(development)
    output_dir = root / "models" / "classical" / MODEL_SET_NAME
    test_metrics: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, str]] = {}

    for model_name in model_names:
        print(f"Refitting final dataset-balanced model: {model_name}", flush=True)
        model = build_pipeline(model_name, config)
        model.fit(
            development_text,
            development_labels,
            classifier__sample_weight=development_weights,
        )
        test_metrics[model_name] = evaluate_by_dataset(model, test)
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "pipeline.joblib"
        joblib.dump(model, model_path, compress=3)
        artifacts[model_name] = {
            "path": str(model_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(model_path),
        }

    result: dict[str, Any] = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_set": MODEL_SET_NAME,
        "models": model_names,
        "datasets": datasets,
        "label_convention": {"0": "real", "1": "fake"},
        "weighting": "equal total weight per dataset and class",
        "selection_metric": "mean per-dataset validation macro-F1, then worst-dataset macro-F1",
        "selected_model": selected_model,
        "records": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "development_refit": int(len(development)),
            "test": int(len(test)),
        },
        "validation_metrics": validation_metrics,
        "held_out_test_metrics": test_metrics,
        "config": config.to_dict(),
        "artifacts": artifacts,
        "manifest_sha256": sha256_file(manifest_path),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "limitations": [
            "This remains content classification, not evidence-based fact verification.",
            "Collection and labeling policies differ across datasets.",
            "The four test splits were not used for fitting or model selection.",
            "Future evaluation still needs temporal and source-disjoint data.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--models", nargs="+", choices=SUPPORTED_MODELS, default=list(SUPPORTED_MODELS)
    )
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_261_017)
    args = parser.parse_args()
    config = ClassicalModelConfig(
        max_features=args.max_features,
        class_weight=None,
        random_state=args.seed,
    )
    run_training(args.root, args.datasets, args.models, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
