"""Tune Random Forest only on source-held-out validation folds.

The held-out test splits and contemporary diagnostic are not consulted during
selection.  This keeps the comparison honest while exploring settings that are
meaningful for high-dimensional sparse TF-IDF text.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from backend.ml.classical_model import combine_text_fields, fake_probabilities
from backend.ml.robust_classical_model import RobustRepresentationConfig, build_robust_pipeline
from scripts.select_robust_representation import load_liar_extremes
from scripts.train_classical_multidataset import DEFAULT_DATASETS, load_combined_split
from scripts.train_classical_robust import fit_frame, threshold_summary


BASE_CONFIG = RobustRepresentationConfig(
    representation="word",
    mask_sources=True,
    mask_framing=False,
)

CANDIDATES = {
    "sqrt_leaf2_gini_mask_framing_120": replace(
        BASE_CONFIG,
        mask_framing=True,
        random_forest_estimators=120,
        random_forest_min_samples_leaf=2,
        random_forest_max_features="sqrt",
        random_forest_criterion="gini",
    ),
    "sqrt_leaf1_gini_120": replace(
        BASE_CONFIG,
        random_forest_estimators=120,
        random_forest_min_samples_leaf=1,
        random_forest_max_features="sqrt",
        random_forest_criterion="gini",
    ),
    "sqrt_leaf2_logloss_120": replace(
        BASE_CONFIG,
        random_forest_estimators=120,
        random_forest_min_samples_leaf=2,
        random_forest_max_features="sqrt",
        random_forest_criterion="log_loss",
    ),
    "one_percent_leaf2_logloss_120": replace(
        BASE_CONFIG,
        random_forest_estimators=120,
        random_forest_min_samples_leaf=2,
        random_forest_max_features=0.01,
        random_forest_criterion="log_loss",
    ),
}


def run_tuning(
    root: Path,
    datasets: list[str],
    candidate_names: list[str],
    *,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    unknown = sorted(set(candidate_names).difference(CANDIDATES))
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}")
    liar_train = load_liar_extremes(root)
    output = root / "models" / "classical" / "multidataset_v2_robust" / "random_forest_tuning.json"
    results: dict[str, Any] = {}
    if reuse_existing and output.is_file():
        results.update(json.loads(output.read_text(encoding="utf-8")).get("results", {}))
    for candidate_name in candidate_names:
        if reuse_existing and candidate_name in results:
            print(f"Reusing completed candidate: {candidate_name}", flush=True)
            continue
        config = CANDIDATES[candidate_name]
        fold_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        all_labels = []
        all_probabilities = []
        all_datasets = []
        for held_out in datasets:
            print(f"Random Forest tuning: {candidate_name}; held out {held_out}", flush=True)
            train = load_combined_split(root, [name for name in datasets if name != held_out], "train")
            train = pd.concat([train, liar_train], ignore_index=True)
            validation = load_combined_split(root, [held_out], "validation")
            model = fit_frame(build_robust_pipeline("random_forest", config), train)
            labels = validation["label_binary_fake"].astype(int).to_numpy()
            probabilities = fake_probabilities(model, combine_text_fields(validation))
            fold_cache[held_out] = (labels, probabilities)
            all_labels.append(labels)
            all_probabilities.append(probabilities)
            all_datasets.append(np.repeat(held_out, len(labels)))
        labels = np.concatenate(all_labels)
        probabilities = np.concatenate(all_probabilities)
        dataset_names = np.concatenate(all_datasets)
        selected = threshold_summary(labels, probabilities, dataset_names)
        threshold = selected["threshold"]
        folds = {
            dataset: {
                "records": int(len(fold_labels)),
                "macro_f1": float(
                    f1_score(
                        fold_labels,
                        fold_probabilities >= threshold,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
            for dataset, (fold_labels, fold_probabilities) in fold_cache.items()
        }
        results[candidate_name] = {
            "config": config.to_dict(),
            "selected_threshold": selected,
            "folds_at_selected_threshold": folds,
        }
    benchmark_path = output.parent / "benchmark.json"
    if benchmark_path.is_file():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        baseline = benchmark.get("validation", {}).get("random_forest")
        if baseline:
            results["baseline_200_sqrt_leaf2_gini"] = {
                "config": benchmark.get("representation", {}),
                "selected_threshold": baseline["selected_threshold"],
                "folds_at_0_5": baseline.get("folds_at_0_5", {}),
                "note": "Previously completed full baseline retained for fair comparison.",
            }
    selected_candidate = max(
        results,
        key=lambda name: (
            results[name]["selected_threshold"]["mean_dataset_macro_f1"],
            results[name]["selected_threshold"]["worst_dataset_macro_f1"],
        ),
    )
    report = {
        "status": "completed",
        "selection_protocol": "dataset-held-out validation only; no test or contemporary data",
        "datasets": datasets,
        "selected_candidate": selected_candidate,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--candidates", nargs="+", choices=sorted(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    run_tuning(args.root, args.datasets, args.candidates, reuse_existing=args.reuse_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
