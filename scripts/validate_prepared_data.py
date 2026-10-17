"""Validate Phase 4 outputs against their manifest and leakage constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from scripts.prepare_datasets import OUTPUT_COLUMNS, sha256_file
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from prepare_datasets import OUTPUT_COLUMNS, sha256_file


def validate(root: Path) -> dict[str, object]:
    data_root = root / "data"
    manifest_path = data_root / "processed" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[str] = []

    for raw_file in manifest["raw_archives"]:
        path = root / raw_file["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != raw_file["bytes"]:
            raise ValueError(f"Raw byte size changed: {path}")
        if sha256_file(path) != raw_file["sha256"]:
            raise ValueError(f"Raw SHA-256 changed: {path}")
    checks.append(f"raw_archive_hashes:{len(manifest['raw_archives'])}")

    for relative, expected_hash in manifest["output_sha256"].items():
        path = (data_root / "processed" / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Prepared output SHA-256 mismatch: {path}")
    checks.append(f"output_hashes:{len(manifest['output_sha256'])}")

    validated_datasets = 0
    for name, report in manifest["datasets"].items():
        if name == "bs_detector":
            sources = pd.read_csv(data_root / "processed" / name / "source_categories.csv", dtype="string")
            posts = pd.read_csv(data_root / "processed" / name / "unlabeled_posts.csv", dtype="string")
            if len(sources) != report["prepared_source_records"]:
                raise ValueError("BS Detector source count does not match manifest")
            if len(posts) != report["prepared_unlabeled_posts"]:
                raise ValueError("BS Detector post count does not match manifest")
            validated_datasets += 1
            continue

        clean_path = data_root / "processed" / name / "cleaned.csv"
        frame = pd.read_csv(clean_path, dtype="string", keep_default_na=False)
        if list(frame.columns) != OUTPUT_COLUMNS:
            raise ValueError(f"Unexpected columns in {clean_path}")
        if len(frame) != report["prepared_records"]:
            raise ValueError(f"Record count mismatch for {name}")
        if frame["text_clean"].eq("").any():
            raise ValueError(f"Empty cleaned text remains in {name}")
        if frame["text_fingerprint"].duplicated().any():
            raise ValueError(f"Exact normalized duplicates remain in {name}")
        if (frame.groupby("near_cluster_id")["split"].nunique() > 1).any():
            raise ValueError(f"Near-duplicate cluster leakage remains in {name}")

        split_ids: list[set[str]] = []
        split_total = 0
        for split in sorted(frame["split"].unique()):
            split_path = data_root / "processed" / name / "splits" / f"{split}.csv"
            split_frame = pd.read_csv(split_path, dtype="string", keep_default_na=False)
            if list(split_frame.columns) != OUTPUT_COLUMNS:
                raise ValueError(f"Unexpected columns in {split_path}")
            identifiers = set(split_frame["record_id"])
            if len(identifiers) != len(split_frame):
                raise ValueError(f"Duplicate record IDs in {split_path}")
            split_ids.append(identifiers)
            split_total += len(split_frame)
        if split_total != len(frame):
            raise ValueError(f"Split totals do not match cleaned data for {name}")
        for index, left in enumerate(split_ids):
            for right in split_ids[index + 1 :]:
                if left.intersection(right):
                    raise ValueError(f"Record IDs overlap across splits for {name}")
        validated_datasets += 1

    checks.append(f"dataset_outputs:{validated_datasets}")
    return {"status": "passed", "checks": checks, "seed": manifest["seed"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = validate(args.root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
