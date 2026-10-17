from __future__ import annotations

import csv
from pathlib import Path

from scripts.dataset_inspect import inspect_file


def test_csv_inspection_profiles_records_without_modifying_source(tmp_path: Path) -> None:
    source = tmp_path / "synthetic_fixture.csv"
    rows = [
        {"title": "Claim A", "text": "Same text", "label": "fake", "url": "https://a.test"},
        {"title": "Claim A", "text": " same  text ", "label": "fake", "url": "https://a.test"},
        {"title": "Claim B", "text": "Different", "label": "real", "url": ""},
    ]
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    before = source.read_bytes()

    report = inspect_file(source)

    assert report["records"] == 3
    assert report["likely_roles"]["label"] == ["label"]
    assert report["likely_roles"]["text"] == ["text", "title"]
    assert report["column_profiles"]["text"]["duplicate_value_records"] == 1
    assert report["column_profiles"]["url"]["missing"] == 1
    assert report["label_distributions"]["label"] == {"fake": 2, "real": 1}
    assert source.read_bytes() == before


def test_exact_duplicate_records_are_reported(tmp_path: Path) -> None:
    source = tmp_path / "synthetic_fixture.csv"
    source.write_text("text,label\nrepeat,fake\nrepeat,fake\n", encoding="utf-8")
    report = inspect_file(source)
    assert report["exact_duplicate_records"] == 1
