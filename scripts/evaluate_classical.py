"""Evaluate trained classical pipelines on other verified binary datasets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from backend.ml.classical_model import SUPPORTED_MODELS, combine_text_fields, evaluate_binary_classifier
from scripts.train_classical import load_split, validate_dataset_manifest


def run_external_evaluation(
    root: Path,
    training_dataset: str,
    target_datasets: list[str],
    model_names: list[str],
) -> dict[str, Any]:
    root = root.resolve()
    if training_dataset in target_datasets:
        raise ValueError("External targets must not include the training dataset")
    results: dict[str, dict[str, Any]] = {}

    for target in target_datasets:
        _, report = validate_dataset_manifest(root, target)
        test = load_split(root, target, "test")
        target_result: dict[str, Any] = {
            "records": int(len(test)),
            "label_semantics": report["label_semantics"],
            "models": {},
        }
        for model_name in model_names:
            model_path = (
                root
                / "models"
                / "classical"
                / training_dataset
                / model_name
                / "pipeline.joblib"
            )
            if not model_path.is_file():
                raise FileNotFoundError(f"Trained model not found: {model_path}")
            model = joblib.load(model_path)
            target_result["models"][model_name] = evaluate_binary_classifier(
                model,
                combine_text_fields(test),
                test["label_binary_fake"],
            )
        results[target] = target_result

    output = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_dataset": training_dataset,
        "targets": results,
        "interpretation": (
            "Cross-dataset performance estimates transfer only; differences in collection, "
            "language, genre, and labeling policy can all affect these metrics."
        ),
    }
    output_path = root / "models" / "classical" / training_dataset / "external_evaluation.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--training-dataset", default="isot")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["ifnd", "indian_fake_news", "bharatgather"],
    )
    parser.add_argument(
        "--models", nargs="+", choices=SUPPORTED_MODELS, default=list(SUPPORTED_MODELS)
    )
    args = parser.parse_args()
    run_external_evaluation(args.root, args.training_dataset, args.targets, args.models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
