"""Export false positives and false negatives from a saved classical model set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.ml.classical_model import combine_text_fields, fake_probabilities
from scripts.train_classical_multidataset import DEFAULT_DATASETS


def run_export(root: Path, model_set: str) -> dict[str, Any]:
    root = root.resolve()
    model_root = root / "models" / "classical" / model_set
    benchmark = json.loads((model_root / "benchmark.json").read_text(encoding="utf-8"))
    datasets = list(benchmark.get("datasets", DEFAULT_DATASETS))
    test = pd.concat(
        [
            pd.read_csv(root / "data" / "processed" / dataset / "splits" / "test.csv", keep_default_na=False)
            for dataset in datasets
        ],
        ignore_index=True,
    )
    test["label_binary_fake"] = test["label_binary_fake"].astype(int)
    output_dir = model_root / "errors"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for model_name in benchmark["models"]:
        threshold = float(
            benchmark.get("validation", {})
            .get(model_name, {})
            .get("selected_threshold", {})
            .get("threshold", 0.5)
        )
        model = joblib.load(model_root / model_name / "pipeline.joblib")
        probabilities = fake_probabilities(model, combine_text_fields(test))
        predictions = (probabilities >= threshold).astype(int)
        errors = test[predictions != test["label_binary_fake"].astype(int).to_numpy()].copy()
        errors["fake_probability"] = probabilities[predictions != test["label_binary_fake"].astype(int).to_numpy()]
        errors["prediction"] = predictions[predictions != test["label_binary_fake"].astype(int).to_numpy()]
        errors["threshold"] = threshold
        errors["decision_margin"] = np.abs(errors["fake_probability"] - threshold)
        errors["error_type"] = np.where(errors["prediction"].eq(1), "false_positive", "false_negative")
        errors["text_excerpt"] = errors["text_clean"].astype(str).str.slice(0, 500)
        errors = errors.sort_values("decision_margin", ascending=False)
        columns = [
            "dataset",
            "record_id",
            "error_type",
            "label_binary_fake",
            "prediction",
            "fake_probability",
            "threshold",
            "decision_margin",
            "source",
            "url",
            "date",
            "text_excerpt",
        ]
        path = output_dir / f"{model_name}_test_errors.csv"
        errors[columns].to_csv(path, index=False)
        counts = errors["error_type"].value_counts()
        summary[model_name] = {
            "threshold": threshold,
            "errors": int(len(errors)),
            "false_positives": int(counts.get("false_positive", 0)),
            "false_negatives": int(counts.get("false_negative", 0)),
            "large_margin_errors_at_least_0_4": int((errors["decision_margin"] >= 0.4).sum()),
            "path": str(path.relative_to(root)).replace("\\", "/"),
        }
    report = {"status": "completed", "model_set": model_set, "summary": summary}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-set", default="multidataset_v2_robust")
    args = parser.parse_args()
    run_export(args.root, args.model_set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
