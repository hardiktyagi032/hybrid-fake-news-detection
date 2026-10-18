"""Train, select, and evaluate TF-IDF classical fake-news baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(root: Path, dataset: str, split: str) -> pd.DataFrame:
    path = root / "data" / "processed" / dataset / "splits" / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared split not found: {path}")
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    required = {"title_clean", "text_clean", "label_binary_fake", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if not frame["split"].eq(split).all():
        raise ValueError(f"{path} contains records assigned to a different split")
    validate_binary_labels(frame["label_binary_fake"])
    return frame


def validate_dataset_manifest(root: Path, dataset: str) -> tuple[Path, dict[str, Any]]:
    manifest_path = root / "data" / "processed" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        report = manifest["datasets"][dataset]
    except KeyError as exc:
        raise ValueError(f"Dataset {dataset!r} is not present in the Phase 4 manifest") from exc
    if report.get("role") != "text_classification":
        raise ValueError(f"Dataset {dataset!r} is not a text-classification dataset")
    if report.get("label_mapping_verified") is not True:
        raise ValueError(f"Dataset {dataset!r} has no verified binary label mapping; training refused")
    return manifest_path, report


def train_candidates(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model_names: list[str],
    config: ClassicalModelConfig,
) -> tuple[str, dict[str, dict[str, Any]]]:
    train_text = combine_text_fields(train)
    validation_text = combine_text_fields(validation)
    y_train = validate_binary_labels(train["label_binary_fake"])
    candidate_metrics: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        print(f"Training candidate: {model_name}", flush=True)
        pipeline = build_pipeline(model_name, config)
        pipeline.fit(train_text, y_train)
        candidate_metrics[model_name] = evaluate_binary_classifier(
            pipeline, validation_text, validation["label_binary_fake"]
        )
        print(
            f"Validation {model_name}: macro-F1={candidate_metrics[model_name]['f1_macro']:.6f}",
            flush=True,
        )

    winner = max(
        model_names,
        key=lambda name: (candidate_metrics[name]["f1_macro"], candidate_metrics[name]["roc_auc"]),
    )
    return winner, candidate_metrics


def run_training(
    root: Path,
    dataset: str,
    model_names: list[str],
    config: ClassicalModelConfig,
) -> dict[str, Any]:
    root = root.resolve()
    unknown = sorted(set(model_names).difference(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    if len(model_names) != len(set(model_names)):
        raise ValueError("Model candidates must be unique")

    manifest_path, dataset_report = validate_dataset_manifest(root, dataset)
    train = load_split(root, dataset, "train")
    validation = load_split(root, dataset, "validation")
    test = load_split(root, dataset, "test")
    winner, candidate_metrics = train_candidates(train, validation, model_names, config)

    development = pd.concat([train, validation], ignore_index=True)
    development_text = combine_text_fields(development)
    development_labels = validate_binary_labels(development["label_binary_fake"])
    test_text = combine_text_fields(test)
    dataset_output_dir = root / "models" / "classical" / dataset
    held_out_metrics: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, str]] = {}

    for model_name in model_names:
        print(f"Refitting final model on train + validation: {model_name}", flush=True)
        final_model = build_pipeline(model_name, config)
        final_model.fit(development_text, development_labels)
        held_out_metrics[model_name] = evaluate_binary_classifier(
            final_model, test_text, test["label_binary_fake"]
        )
        model_dir = dataset_output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "pipeline.joblib"
        joblib.dump(final_model, model_path, compress=3)
        artifacts[model_name] = {
            "path": str(model_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(model_path),
        }
    result: dict[str, Any] = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "label_convention": {"0": "real", "1": "fake"},
        "dataset_label_semantics": dataset_report["label_semantics"],
        "manifest_sha256": sha256_file(manifest_path),
        "selection_metric": "validation f1_macro, then validation roc_auc",
        "candidate_validation_metrics": candidate_metrics,
        "selected_model": winner,
        "final_training_records": int(len(development)),
        "held_out_test_metrics": held_out_metrics,
        "config": config.to_dict(),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifacts": artifacts,
        "limitations": [
            "This is content classification, not evidence-based fact verification.",
            "ISOT may contain publisher, topic, and writing-style cues that inflate in-dataset performance.",
            "The test split was used once after model selection; external and cross-dataset evaluation remain required.",
            "Predicted probability is model confidence under this dataset, not a guarantee of factual truth.",
        ],
    }
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    (dataset_output_dir / "benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    for model_name in model_names:
        model_report = {
            "dataset": dataset,
            "model": model_name,
            "selected": model_name == winner,
            "validation_metrics": candidate_metrics[model_name],
            "held_out_test_metrics": held_out_metrics[model_name],
            "artifact": artifacts[model_name],
            "config": config.to_dict(),
        }
        (dataset_output_dir / model_name / "metrics.json").write_text(
            json.dumps(model_report, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", default="isot")
    parser.add_argument(
        "--models", nargs="+", choices=SUPPORTED_MODELS, default=list(SUPPORTED_MODELS)
    )
    parser.add_argument("--max-features", type=int, default=75_000)
    parser.add_argument("--seed", type=int, default=20_261_017)
    args = parser.parse_args()
    config = ClassicalModelConfig(max_features=args.max_features, random_state=args.seed)
    run_training(args.root, args.dataset, args.models, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
