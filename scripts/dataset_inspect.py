"""Read-only structural inspection for a user-supplied CSV, TSV, JSON, or JSONL file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from backend.utils.helpers import comparison_text, sha256_file


NULL_TOKENS = {"", "null", "none", "nan", "n/a", "na"}


def _read_delimited(path: Path, delimiter: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError("JSON input must be an array of objects")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(row)
    return rows


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return _read_delimited(path, ",")
    if suffix == ".tsv":
        return _read_delimited(path, "\t")
    if suffix == ".json":
        return _read_json(path)
    if suffix in {".jsonl", ".ndjson"}:
        return _read_jsonl(path)
    raise ValueError(f"Unsupported format: {suffix or '<none>'}. Supported: CSV, TSV, JSON, JSONL")


def _is_missing(value: Any) -> bool:
    return value is None or comparison_text(value) in NULL_TOKENS


def _likely_roles(columns: Iterable[str]) -> dict[str, list[str]]:
    roles = {"text": [], "label": [], "url": [], "date": [], "id": [], "metadata": []}
    for column in columns:
        name = column.casefold()
        if any(token in name for token in ("label", "class", "target", "verdict", "rating")):
            roles["label"].append(column)
        elif any(token in name for token in ("url", "link", "uri")):
            roles["url"].append(column)
        elif any(token in name for token in ("date", "time", "published", "created")):
            roles["date"].append(column)
        elif name == "id" or name.endswith("_id"):
            roles["id"].append(column)
        elif any(token in name for token in ("text", "content", "article", "claim", "title", "headline", "statement")):
            roles["text"].append(column)
        else:
            roles["metadata"].append(column)
    return roles


def inspect_file(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = load_rows(path)
    columns = sorted({str(key) for row in rows for key in row})
    roles = _likely_roles(columns)
    canonical_rows = [json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) for row in rows]

    profiles: dict[str, Any] = {}
    for column in columns:
        values = [row.get(column) for row in rows]
        non_missing = [value for value in values if not _is_missing(value)]
        normalized = [comparison_text(value) for value in non_missing]
        counts = Counter(normalized)
        profiles[column] = {
            "missing": len(values) - len(non_missing),
            "non_missing": len(non_missing),
            "unique_normalized": len(counts),
            "duplicate_value_records": sum(count - 1 for count in counts.values() if count > 1),
            "sample_values": [str(value)[:160] for value in non_missing[:3]],
        }

    label_distributions = {
        column: dict(Counter(comparison_text(row.get(column)) for row in rows if not _is_missing(row.get(column))))
        for column in roles["label"]
    }

    return {
        "input": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "format": path.suffix.casefold().lstrip("."),
        },
        "records": len(rows),
        "columns": columns,
        "likely_roles": roles,
        "column_profiles": profiles,
        "exact_duplicate_records": len(canonical_rows) - len(set(canonical_rows)),
        "label_distributions": label_distributions,
        "notes": [
            "Role detection is name-based and must be confirmed by a human.",
            "Whitespace/case normalization is used only for duplicate analysis; source values are unchanged.",
            "Near-duplicate, leakage, label-consistency, and split analysis require dataset-specific decisions.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_file(args.input)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
