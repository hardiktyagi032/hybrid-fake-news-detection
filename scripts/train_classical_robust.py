"""Train source-robust Logistic Regression, Linear SVM, and Random Forest models."""

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
from sklearn.metrics import f1_score

from backend.ml.classical_model import (
    SUPPORTED_MODELS,
    combine_text_fields,
    evaluate_binary_classifier,
    fake_probabilities,
)
from backend.ml.robust_classical_model import RobustRepresentationConfig, build_robust_pipeline
from scripts.select_robust_representation import load_liar_extremes
from scripts.train_classical import sha256_file
from scripts.train_classical_multidataset import (
    DEFAULT_DATASETS,
    dataset_class_balanced_weights,
    load_combined_split,
)


MODEL_SET_NAME = "multidataset_v2_robust"
LIAR_WEIGHT = 0.25


def robust_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = dataset_class_balanced_weights(frame)
    liar = frame["dataset"].eq("liar_extremes").to_numpy()
    weights[liar] *= LIAR_WEIGHT
    return weights / weights.mean()


def threshold_summary(labels: np.ndarray, probabilities: np.ndarray, datasets: np.ndarray) -> dict[str, Any]:
    candidates = []
    for threshold in np.round(np.arange(0.20, 0.801, 0.01), 2):
        scores = []
        for dataset in sorted(set(datasets.tolist())):
            mask = datasets == dataset
            scores.append(
                f1_score(labels[mask], probabilities[mask] >= threshold, average="macro", zero_division=0)
            )
        candidates.append(
            {
                "threshold": float(threshold),
                "mean_dataset_macro_f1": float(np.mean(scores)),
                "worst_dataset_macro_f1": float(np.min(scores)),
            }
        )
    return max(
        candidates,
        key=lambda row: (
            row["mean_dataset_macro_f1"],
            row["worst_dataset_macro_f1"],
            -abs(row["threshold"] - 0.5),
        ),
    )


def fit_frame(model: Any, frame: pd.DataFrame) -> Any:
    model.fit(
        combine_text_fields(frame),
        frame["label_binary_fake"].astype(int).to_numpy(),
        classifier__sample_weight=robust_weights(frame),
    )
    return model


def run_training(
    root: Path,
    datasets: list[str],
    model_names: list[str],
    *,
    merge_existing: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    unknown = sorted(set(model_names).difference(SUPPORTED_MODELS))
    if unknown or not model_names or len(model_names) != len(set(model_names)):
        raise ValueError(f"Models must be unique approved values; invalid={unknown}")
    config = RobustRepresentationConfig(representation="word", mask_sources=True, mask_framing=False)
    liar_train = load_liar_extremes(root)
    output_dir = root / "models" / "classical" / MODEL_SET_NAME
    existing: dict[str, Any] = {}
    benchmark_path = output_dir / "benchmark.json"
    if merge_existing and benchmark_path.is_file():
        existing = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if existing.get("model_set") != MODEL_SET_NAME or existing.get("datasets") != datasets:
            raise ValueError("Existing benchmark is not compatible with this training request")
    validation: dict[str, Any] = dict(existing.get("validation", {}))

    for model_name in model_names:
        fold_probabilities = []
        fold_labels = []
        fold_datasets = []
        fold_metrics: dict[str, Any] = {}
        for held_out in datasets:
            fit_datasets = [name for name in datasets if name != held_out]
            train = load_combined_split(root, fit_datasets, "train")
            train = pd.concat([train, liar_train], ignore_index=True)
            held_out_validation = load_combined_split(root, [held_out], "validation")
            print(f"Robust validation: {model_name}, held out {held_out}", flush=True)
            model = fit_frame(build_robust_pipeline(model_name, config), train)
            probabilities = fake_probabilities(model, combine_text_fields(held_out_validation))
            labels = held_out_validation["label_binary_fake"].astype(int).to_numpy()
            fold_probabilities.append(probabilities)
            fold_labels.append(labels)
            fold_datasets.append(np.repeat(held_out, len(labels)))
            fold_metrics[held_out] = evaluate_binary_classifier(
                model, combine_text_fields(held_out_validation), labels
            )
        labels = np.concatenate(fold_labels)
        probabilities = np.concatenate(fold_probabilities)
        dataset_names = np.concatenate(fold_datasets)
        selected_threshold = threshold_summary(labels, probabilities, dataset_names)
        validation[model_name] = {
            "protocol": "dataset-held-out validation folds",
            "folds_at_0_5": fold_metrics,
            "selected_threshold": selected_threshold,
        }

    all_model_names = [
        name for name in SUPPORTED_MODELS if name in set(existing.get("models", [])) | set(model_names)
    ]
    selected_model = max(
        all_model_names,
        key=lambda name: (
            validation[name]["selected_threshold"]["mean_dataset_macro_f1"],
            validation[name]["selected_threshold"]["worst_dataset_macro_f1"],
        ),
    )
    core_development = pd.concat(
        [
            load_combined_split(root, datasets, "train"),
            load_combined_split(root, datasets, "validation"),
        ],
        ignore_index=True,
    )
    development = pd.concat([core_development, liar_train], ignore_index=True)
    test = load_combined_split(root, datasets, "test")
    artifacts: dict[str, Any] = dict(existing.get("artifacts", {}))
    test_metrics: dict[str, Any] = dict(existing.get("held_out_test_metrics", {}))
    for model_name in model_names:
        print(f"Final robust refit: {model_name}", flush=True)
        model = fit_frame(build_robust_pipeline(model_name, config), development)
        threshold = validation[model_name]["selected_threshold"]["threshold"]
        test_metrics[model_name] = {
            dataset: evaluate_binary_classifier(
                model,
                combine_text_fields(group),
                group["label_binary_fake"],
                threshold=threshold,
            )
            for dataset, group in test.groupby("dataset", sort=True)
        }
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "pipeline.joblib"
        joblib.dump(model, model_path, compress=3)
        artifacts[model_name] = {
            "path": str(model_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(model_path),
        }
    report = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_set": MODEL_SET_NAME,
        "models": all_model_names,
        "selected_model": selected_model,
        "datasets": datasets,
        "liar_binary_mapping": {"true": 0, "false": 1, "pants-fire": 1},
        "liar_excluded_labels": ["mostly-true", "half-true", "barely-true"],
        "liar_dataset_weight": LIAR_WEIGHT,
        "representation": config.to_dict(),
        "selection_metric": "mean dataset-held-out validation macro-F1 after validation-only threshold selection",
        "validation": validation,
        "held_out_test_metrics": test_metrics,
        "records": {
            "core_development": int(len(core_development)),
            "liar_train_extremes": int(len(liar_train)),
            "final_fit": int(len(development)),
            "test": int(len(test)),
        },
        "artifacts": artifacts,
        "update_mode": "merge_existing" if merge_existing else "full_rebuild",
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "limitations": [
            "Source-disjoint validation remains weak on the worst dataset.",
            "LIAR endpoint mapping is a documented model-training choice, not a change to prepared data.",
            "The test collections have already been inspected during prior baseline work and are not pristine for future iteration.",
            "Content-pattern classification is not evidence-based fact verification.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Train only requested models and preserve compatible models already in benchmark.json.",
    )
    args = parser.parse_args()
    run_training(args.root, args.datasets, args.models, merge_existing=args.merge_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
