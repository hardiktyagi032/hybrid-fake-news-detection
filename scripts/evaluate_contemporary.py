"""Evaluate saved classical models on the frozen contemporary diagnostic set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from backend.ml.classical_model import fake_probabilities


def score(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (probabilities >= threshold).astype(int)
    return {
        "records": int(len(labels)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, prediction, labels=[0, 1]).astype(int).tolist(),
    }


def run_evaluation(root: Path) -> dict[str, Any]:
    root = root.resolve()
    fixture = json.loads((root / "evaluation" / "contemporary_2026.json").read_text(encoding="utf-8"))
    records = fixture["records"]
    documents = [record["text"] for record in records]
    labels = np.asarray([record["label_binary_fake"] for record in records], dtype=int)
    candidates: dict[str, tuple[Path, float]] = {}
    for version, model_set in (
        ("v1", "multidataset_v1"),
        ("v2", "multidataset_v2_robust"),
    ):
        model_root = root / "models" / "classical" / model_set
        benchmark_path = model_root / "benchmark.json"
        if not benchmark_path.is_file():
            continue
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        for model_name in benchmark.get("models", []):
            threshold = float(
                benchmark.get("validation", {})
                .get(model_name, {})
                .get("selected_threshold", {})
                .get("threshold", 0.5)
            )
            candidates[f"{version}_{model_name}"] = (
                model_root / model_name / "pipeline.joblib",
                threshold,
            )
    results = {}
    predictions = []
    skipped = {}
    for name, (path, threshold) in candidates.items():
        if not path.is_file():
            skipped[name] = f"model artifact not found: {path.relative_to(root)}"
            continue
        model = joblib.load(path)
        probabilities = fake_probabilities(model, documents)
        results[name] = score(labels, probabilities, threshold)
        for record, probability in zip(records, probabilities, strict=True):
            predictions.append(
                {
                    "model": name,
                    "id": record["id"],
                    "label_binary_fake": record["label_binary_fake"],
                    "fake_probability": float(probability),
                    "prediction": int(probability >= threshold),
                }
            )
    report = {
        "status": "completed",
        "fixture": fixture["name"],
        "fixture_created_date": fixture["created_date"],
        "limitations": fixture["limitations"],
        "results": results,
        "skipped": skipped,
        "predictions": predictions,
    }
    output = root / "models" / "classical" / "contemporary_2026_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run_evaluation(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
