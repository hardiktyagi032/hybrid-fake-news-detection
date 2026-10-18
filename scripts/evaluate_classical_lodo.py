"""Evaluate classical models by leaving each source dataset out of fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.ml.classical_model import (
    SUPPORTED_MODELS,
    ClassicalModelConfig,
    build_pipeline,
    combine_text_fields,
)
from scripts.train_classical_multidataset import (
    DEFAULT_DATASETS,
    dataset_class_balanced_weights,
    evaluate_by_dataset,
    load_combined_split,
)


def run_lodo(
    root: Path,
    datasets: list[str],
    model_names: list[str],
    config: ClassicalModelConfig,
) -> dict[str, Any]:
    root = root.resolve()
    results: dict[str, dict[str, Any]] = {name: {} for name in model_names}
    for held_out in datasets:
        training_datasets = [name for name in datasets if name != held_out]
        train = pd.concat(
            [
                load_combined_split(root, training_datasets, "train"),
                load_combined_split(root, training_datasets, "validation"),
            ],
            ignore_index=True,
        )
        test = load_combined_split(root, [held_out], "test")
        weights = dataset_class_balanced_weights(train)
        for model_name in model_names:
            print(f"Fitting {model_name}; held-out dataset={held_out}", flush=True)
            model = build_pipeline(model_name, config)
            model.fit(
                combine_text_fields(train),
                train["label_binary_fake"].astype(int).to_numpy(),
                classifier__sample_weight=weights,
            )
            results[model_name][held_out] = evaluate_by_dataset(model, test)["pooled"]

    summary = {}
    for model_name, model_results in results.items():
        macro_f1 = [metrics["f1_macro"] for metrics in model_results.values()]
        summary[model_name] = {
            "mean_held_out_dataset_macro_f1": float(np.mean(macro_f1)),
            "worst_held_out_dataset_macro_f1": float(np.min(macro_f1)),
        }
    report = {
        "status": "completed",
        "protocol": "For each fold, fit on train+validation from three datasets and evaluate the untouched test split of the fourth.",
        "datasets": datasets,
        "config": config.to_dict(),
        "results": results,
        "summary": summary,
    }
    output = root / "models" / "classical" / "multidataset_v1" / "audit" / "leave_one_dataset_out.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
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
    parser.add_argument("--max-features", type=int, default=100_000)
    args = parser.parse_args()
    report = run_lodo(
        args.root,
        args.datasets,
        args.models,
        ClassicalModelConfig(max_features=args.max_features, class_weight=None),
    )
    print(json.dumps({"summary": report["summary"], "results": report["results"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
