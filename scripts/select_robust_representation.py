"""Select a robust feature representation using dataset-held-out validation folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.ml.classical_model import combine_text_fields, evaluate_binary_classifier
from backend.ml.robust_classical_model import RobustRepresentationConfig, build_robust_pipeline
from scripts.train_classical_multidataset import DEFAULT_DATASETS, dataset_class_balanced_weights, load_combined_split


CANDIDATES = {
    "word_unmasked": RobustRepresentationConfig(representation="word", mask_sources=False),
    "word_source_masked": RobustRepresentationConfig(representation="word", mask_sources=True),
    "word_source_and_framing_masked": RobustRepresentationConfig(
        representation="word", mask_sources=True, mask_framing=True
    ),
    "word_char_source_masked": RobustRepresentationConfig(representation="word_char", mask_sources=True),
    "word_source_masked_plus_liar_extremes": RobustRepresentationConfig(
        representation="word", mask_sources=True
    ),
    "word_source_masked_plus_liar_extremes_025": RobustRepresentationConfig(
        representation="word", mask_sources=True
    ),
    "word_source_masked_plus_liar_extremes_050": RobustRepresentationConfig(
        representation="word", mask_sources=True
    ),
}


def load_liar_extremes(root: Path) -> pd.DataFrame:
    """Map only LIAR's clear endpoint labels for model-training augmentation."""

    path = root / "data" / "processed" / "liar" / "splits" / "train.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    mapping = {"true": 0, "false": 1, "pants-fire": 1}
    frame = frame[frame["label_raw"].isin(mapping)].copy()
    frame["dataset"] = "liar_extremes"
    frame["label_binary_fake"] = frame["label_raw"].map(mapping).astype(int)
    return frame


def run_selection(
    root: Path,
    datasets: list[str],
    candidate_names: list[str] | None = None,
    merge_existing: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    output = root / "models" / "classical" / "robust_selection.json"
    results: dict[str, dict[str, Any]] = {}
    if merge_existing and output.is_file():
        results.update(json.loads(output.read_text(encoding="utf-8")).get("results", {}))
    selected_names = candidate_names or list(CANDIDATES)
    unknown = sorted(set(selected_names).difference(CANDIDATES))
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}")
    for candidate_name in selected_names:
        config = CANDIDATES[candidate_name]
        folds: dict[str, Any] = {}
        print(f"Candidate: {candidate_name}", flush=True)
        for held_out in datasets:
            fit_datasets = [name for name in datasets if name != held_out]
            train = load_combined_split(root, fit_datasets, "train")
            if "plus_liar_extremes" in candidate_name:
                train = pd.concat([train, load_liar_extremes(root)], ignore_index=True)
            validation = load_combined_split(root, [held_out], "validation")
            model = build_robust_pipeline("logistic_regression", config)
            weights = dataset_class_balanced_weights(train)
            liar_weight = 1.0
            if candidate_name.endswith("_025"):
                liar_weight = 0.25
            elif candidate_name.endswith("_050"):
                liar_weight = 0.50
            if liar_weight != 1.0:
                weights[train["dataset"].eq("liar_extremes").to_numpy()] *= liar_weight
                weights /= weights.mean()
            model.fit(
                combine_text_fields(train),
                train["label_binary_fake"].astype(int).to_numpy(),
                classifier__sample_weight=weights,
            )
            metrics = evaluate_binary_classifier(
                model,
                combine_text_fields(validation),
                validation["label_binary_fake"],
            )
            folds[held_out] = metrics
            print(f"  held out {held_out}: macro-F1={metrics['f1_macro']:.4f}", flush=True)
        scores = [metrics["f1_macro"] for metrics in folds.values()]
        results[candidate_name] = {
            "config": config.to_dict(),
            "folds": folds,
            "mean_dataset_macro_f1": float(np.mean(scores)),
            "worst_dataset_macro_f1": float(np.min(scores)),
        }
    selected = max(
        results,
        key=lambda name: (
            results[name]["mean_dataset_macro_f1"],
            results[name]["worst_dataset_macro_f1"],
        ),
    )
    report = {
        "status": "completed",
        "selection_data": "validation split of a dataset excluded entirely from that fold's training",
        "model": "logistic_regression",
        "selected_candidate": selected,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--candidates", nargs="+", choices=sorted(CANDIDATES))
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()
    report = run_selection(args.root, args.datasets, args.candidates, args.merge_existing)
    summary = {
        name: {
            "mean_dataset_macro_f1": values["mean_dataset_macro_f1"],
            "worst_dataset_macro_f1": values["worst_dataset_macro_f1"],
        }
        for name, values in report["results"].items()
    }
    print(json.dumps({"selected_candidate": report["selected_candidate"], "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
