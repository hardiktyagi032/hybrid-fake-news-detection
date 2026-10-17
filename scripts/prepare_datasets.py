"""Prepare the user-supplied fake-news datasets without merging their semantics.

The pipeline preserves raw downloads, creates reversible cleaned copies, quarantines
invalid/conflicting rows, and writes deterministic leakage-aware splits. It does not
train a model or select a final training corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import unicodedata
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import numpy as np


PIPELINE_VERSION = "1.0.0"
DEFAULT_SEED = 20261017
NEAR_DUPLICATE_MAX_HAMMING = 3
NEAR_DUPLICATE_MIN_TOKENS = 6
NEAR_DUPLICATE_MAX_TOKENS = 5000
NEAR_DUPLICATE_MAX_FEATURES = 128
MAX_BAND_BUCKET = 2000

OUTPUT_COLUMNS = [
    "dataset",
    "record_id",
    "title_original",
    "title_clean",
    "text_original",
    "text_clean",
    "text_fingerprint",
    "near_cluster_id",
    "label_raw",
    "label_binary_fake",
    "url",
    "source",
    "date",
    "language",
    "split_original",
    "split",
    "metadata_json",
]

LIAR_COLUMNS = [
    "source_record_id",
    "label",
    "statement",
    "subjects",
    "speaker",
    "speaker_job_title",
    "state_info",
    "party_affiliation",
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_fire_count",
    "context",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_url: str
    license: str
    label_semantics: str
    label_mapping_verified: bool
    role: str = "text_classification"


DATASET_SPECS = {
    "isot": DatasetSpec(
        "isot",
        "https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset/data",
        "Kaggle metadata: MIT; underlying article copyright still applies",
        "File membership: Fake.csv=fake and True.csv=real; fake indicator 1=fake, 0=real.",
        True,
    ),
    "indian_fake_news": DatasetSpec(
        "indian_fake_news",
        "https://www.kaggle.com/datasets/imbikramsaha/fake-real-news",
        "CC0 1.0 (Kaggle metadata)",
        "FAKE=fake and REAL=real; fake indicator 1=fake, 0=real.",
        True,
    ),
    "fri": DatasetSpec(
        "fri",
        "https://github.com/shivani-tufchi/Fake-News-Classification",
        "No license file found in supplied repository",
        "Raw labels are 0/1, but the repository does not define which value means fake.",
        False,
    ),
    "ifnd": DatasetSpec(
        "ifnd",
        "https://github.com/sonalgarg174/Dataset",
        "Research use only (repository README)",
        "README: TRUE=true and Fake=fake; fake indicator 1=fake, 0=real.",
        True,
    ),
    "bharatfakenewskosh": DatasetSpec(
        "bharatfakenewskosh",
        "https://www.kaggle.com/datasets/man2191989/bharatfakenewskosh",
        "CC BY-NC 4.0 (Kaggle metadata)",
        "Workbook labels are booleans, but observed counts conflict with the published class totals; mapping is not asserted.",
        False,
    ),
    "bharatgather": DatasetSpec(
        "bharatgather",
        "https://huggingface.co/datasets/l3cube-pune/BharatGather",
        "CC BY 4.0",
        "Dataset card: 1=true and 0=false; fake indicator 1 for raw label 0.",
        True,
    ),
    "liar": DatasetSpec(
        "liar",
        "https://github.com/tfs4/liar_dataset",
        "Research use only; original sources retain copyright",
        "Six ordered truthfulness labels are preserved; no binary mapping is imposed.",
        False,
    ),
    "bs_detector": DatasetSpec(
        "bs_detector",
        "https://github.com/thiagovas/bs-detector-dataset",
        "LGPL-3.0 repository license",
        "Source-category metadata and unlabeled posts; not an article-level truth dataset.",
        False,
        role="source_credibility_auxiliary",
    ),
}


def scalar(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def clean_text(value: Any) -> str:
    """Apply reversible, model-neutral Unicode/markup/whitespace cleanup."""

    text = scalar(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\x00", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = "".join(character if character in "\n\t" or unicodedata.category(character) != "Cc" else " " for character in text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t\r\f\v]*\n[ \t\r\f\v]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def comparison_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value)).casefold()
    text = re.sub(r"https?://\S+", " URL ", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combine_text(title: Any, body: Any) -> str:
    title_text = scalar(title).strip()
    body_text = scalar(body).strip()
    if title_text and body_text and comparison_text(title_text) != comparison_text(body_text):
        return f"{title_text}\n\n{body_text}"
    return body_text or title_text


def metadata_json(**values: Any) -> str:
    payload = {key: scalar(value) for key, value in values.items() if scalar(value) != ""}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[OUTPUT_COLUMNS].fillna("").astype("string")


def read_csv_fallback(path: Path, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False, **kwargs)


def load_isot(source_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    folder = source_root / "isot-fake-news-dataset" / "News_Dataset"
    parts: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for filename, raw_label, fake_label in (("Fake.csv", "FAKE", "1"), ("True.csv", "REAL", "0")):
        frame = read_csv_fallback(folder / filename)
        parts.append(frame)
        for index, row in frame.iterrows():
            title = scalar(row["title"])
            body = scalar(row["text"])
            rows.append(
                {
                    "dataset": "isot",
                    "record_id": f"{filename}:{index + 2}",
                    "title_original": title,
                    "text_original": combine_text(title, body),
                    "label_raw": raw_label,
                    "label_binary_fake": fake_label,
                    "date": row["date"],
                    "source": row["subject"],
                    "language": "English",
                    "metadata_json": metadata_json(subject=row["subject"], source_file=filename),
                }
            )
    combined = pd.concat(parts, ignore_index=True)
    return canonical_frame(rows), raw_profile(combined)


def load_indian_fake_news(source_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = source_root / "indian-fake-news" / "news_dataset.csv"
    frame = read_csv_fallback(path)
    rows = []
    for index, row in frame.iterrows():
        raw_label = scalar(row["label"]).upper()
        binary = {"FAKE": "1", "REAL": "0"}.get(raw_label, "")
        rows.append(
            {
                "dataset": "indian_fake_news",
                "record_id": f"news_dataset.csv:{index + 2}",
                "text_original": row["text"],
                "label_raw": raw_label,
                "label_binary_fake": binary,
                "language": "English",
            }
        )
    return canonical_frame(rows), raw_profile(frame)


def load_fri(source_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = source_root / "fake-news-classification" / "Fake-News-Classification-main" / "FRI dataset.csv"
    frame = read_csv_fallback(path)
    rows = [
        {
            "dataset": "fri",
            "record_id": f"FRI dataset.csv:{index + 2}",
            "text_original": row["ArticleBody"],
            "label_raw": row["Label"],
            "language": "English",
        }
        for index, row in frame.iterrows()
    ]
    return canonical_frame(rows), raw_profile(frame)


def load_ifnd(source_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = source_root / "ifnd-dataset" / "Dataset-main" / "IFND.csv"
    frame = read_csv_fallback(path)
    rows = []
    for index, row in frame.iterrows():
        raw_label = scalar(row["Label"])
        binary = {"fake": "1", "true": "0"}.get(raw_label.casefold(), "")
        rows.append(
            {
                "dataset": "ifnd",
                "record_id": scalar(row["id"]) or f"IFND.csv:{index + 2}",
                "title_original": row["Statement"],
                "text_original": row["Statement"],
                "label_raw": raw_label,
                "label_binary_fake": binary,
                "url": row["Web"],
                "source": row["Category"],
                "date": row["Date"],
                "language": "English",
                "metadata_json": metadata_json(image_url=row["Image"], category=row["Category"]),
            }
        )
    return canonical_frame(rows), raw_profile(frame)


def load_bharatfakenewskosh(source_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = source_root / "bharatfakenewskosh" / "bharatfakenewskosh (3).xlsx"
    frame = pd.read_excel(path, sheet_name="A")
    rows = []
    for index, row in frame.iterrows():
        title = scalar(row["Eng_Trans_Statement"]) or scalar(row["Statement"])
        body = scalar(row["Eng_Trans_News_Body"]) or scalar(row["News Body"])
        rows.append(
            {
                "dataset": "bharatfakenewskosh",
                "record_id": scalar(row["id"]) or f"sheet-A:{index + 2}",
                "title_original": title,
                "text_original": combine_text(title, body),
                "label_raw": row["Label"],
                "url": row["Fact_Check_Link"],
                "source": row["Fact_Check_Source"],
                "date": row["Publish_Date"],
                "language": row["Language"],
                "metadata_json": metadata_json(
                    author=row["Author_Name"],
                    source_type=row["Source_Type"],
                    media_link=row["Media_Link"],
                    category=row["News_Category"],
                    region=row["Region"],
                    platform=row["Platform"],
                    has_text=row["Text"],
                    has_video=row["Video"],
                    has_image=row["Image"],
                ),
            }
        )
    profile = raw_profile(frame)
    profile["workbook_sheet_used"] = "A"
    profile["other_sheets"] = ["Sheet1 (malformed staging data; not prepared)", "Sheet3 (empty)"]
    return canonical_frame(rows), profile


def load_bharatgather(source_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    folder = source_root / "bharatgather"
    frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for split, filename in (("train", "train.json"), ("test", "test.json")):
        with (folder / filename).open("r", encoding="utf-8") as stream:
            frame = pd.DataFrame(json.load(stream))
        frames.append(frame.assign(_split=split))
        for index, row in frame.iterrows():
            raw_label = scalar(row["label"])
            binary = {"0": "1", "1": "0"}.get(raw_label, "")
            rows.append(
                {
                    "dataset": "bharatgather",
                    "record_id": scalar(row["id"]) or f"{filename}:{index + 1}",
                    "title_original": row["headline"],
                    "text_original": row["training_text"],
                    "label_raw": raw_label,
                    "label_binary_fake": binary,
                    "url": row["url"],
                    "source": row["src"],
                    "language": row["language"],
                    "split_original": split,
                    "metadata_json": metadata_json(
                        tone=row["tone"],
                        content_type=row["content_type"],
                        full_article=row["full_article"],
                    ),
                }
            )
    return canonical_frame(rows), raw_profile(pd.concat(frames, ignore_index=True))


def load_liar(source_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    folder = source_root / "liar-dataset" / "liar_dataset-master"
    frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for split, filename in (("train", "train.tsv"), ("validation", "valid.tsv"), ("test", "test.tsv")):
        frame = pd.read_csv(
            folder / filename,
            sep="\t",
            header=None,
            names=LIAR_COLUMNS,
            dtype="string",
            keep_default_na=False,
        )
        frames.append(frame.assign(_split=split))
        for index, row in frame.iterrows():
            rows.append(
                {
                    "dataset": "liar",
                    "record_id": scalar(row["source_record_id"]) or f"{filename}:{index + 1}",
                    "title_original": row["statement"],
                    "text_original": row["statement"],
                    "label_raw": row["label"],
                    "source": row["speaker"],
                    "language": "English",
                    "split_original": split,
                    "metadata_json": metadata_json(
                        subjects=row["subjects"],
                        speaker=row["speaker"],
                        speaker_job_title=row["speaker_job_title"],
                        state=row["state_info"],
                        party=row["party_affiliation"],
                        context=row["context"],
                        barely_true_count=row["barely_true_count"],
                        false_count=row["false_count"],
                        half_true_count=row["half_true_count"],
                        mostly_true_count=row["mostly_true_count"],
                        pants_fire_count=row["pants_fire_count"],
                    ),
                }
            )
    return canonical_frame(rows), raw_profile(pd.concat(frames, ignore_index=True))


def load_bs_detector(source_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    folder = source_root / "bs-detector-dataset" / "bs-detector-dataset-master"
    source_columns = ["source_id", "domain", "category_primary", "category_secondary", "category_tertiary", "notes"]
    sources = pd.read_csv(folder / "data_final.csv", header=None, names=source_columns, dtype="string", keep_default_na=False)
    sources["domain_clean"] = sources["domain"].map(lambda value: clean_text(value).casefold())
    sources = sources[sources["domain_clean"] != ""].drop_duplicates(subset=["domain_clean"], keep="first")

    post_columns = ["post_id", "title", "time_marker", "publisher", "url"]
    posts = pd.read_csv(folder / "posts_data.csv", header=None, names=post_columns, dtype="string", keep_default_na=False)
    posts["title_clean"] = posts["title"].map(clean_text)
    posts["publisher_clean"] = posts["publisher"].map(clean_text)
    posts = posts.drop_duplicates(subset=["post_id"], keep="first")
    profile = {
        "source_table": raw_profile(sources[source_columns]),
        "unlabeled_posts": raw_profile(posts[post_columns]),
    }
    return sources, posts, profile


LOADERS = {
    "isot": load_isot,
    "indian_fake_news": load_indian_fake_news,
    "fri": load_fri,
    "ifnd": load_ifnd,
    "bharatfakenewskosh": load_bharatfakenewskosh,
    "bharatgather": load_bharatgather,
    "liar": load_liar,
}


def raw_profile(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "records": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "missing_by_column": {
            str(column): int(count)
            for column, count in frame.isna().sum().items()
            if int(count) > 0
        },
        "exact_full_row_duplicates": int(frame.duplicated().sum()),
    }


def _simhash64(text: str) -> tuple[int, int]:
    tokens = re.findall(r"\w+", comparison_text(text), flags=re.UNICODE)[:NEAR_DUPLICATE_MAX_TOKENS]
    token_count = len(tokens)
    if token_count < NEAR_DUPLICATE_MIN_TOKENS:
        return 0, token_count
    features: list[str]
    if token_count >= 20:
        available = token_count - 2
        feature_count = min(available, NEAR_DUPLICATE_MAX_FEATURES)
        indices = sorted({min(available - 1, (position * available) // feature_count) for position in range(feature_count)})
        features = [" ".join(tokens[index : index + 3]) for index in indices]
    else:
        features = tokens
    hashes = []
    for feature in features:
        encoded = feature.encode("utf-8")
        high = zlib.crc32(encoded, 0x9E3779B9)
        low = zlib.crc32(encoded, 0x85EBCA6B)
        hashes.append((high << 32) | low)
    bit_matrix = np.unpackbits(
        np.asarray(hashes, dtype=np.uint64).view(np.uint8),
        bitorder="little",
    ).reshape(len(hashes), 64)
    majority = bit_matrix.sum(axis=0) * 2 >= len(hashes)
    value = sum((1 << bit) for bit, enabled in enumerate(majority.tolist()) if enabled)
    return value, token_count


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def near_duplicate_clusters(texts: Sequence[str], fingerprints: Sequence[str]) -> tuple[list[str], dict[str, int]]:
    union_find = UnionFind(len(texts))
    bands: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(4)]
    hashes: list[int] = [0] * len(texts)
    eligible = 0
    skipped_large_bucket = 0

    for index, text in enumerate(texts):
        value, token_count = _simhash64(text)
        hashes[index] = value
        if token_count < NEAR_DUPLICATE_MIN_TOKENS:
            continue
        eligible += 1
        candidates: set[int] = set()
        keys = [(value >> (band * 16)) & 0xFFFF for band in range(4)]
        for band, key in enumerate(keys):
            bucket = bands[band][key]
            if len(bucket) <= MAX_BAND_BUCKET:
                candidates.update(bucket)
            else:
                skipped_large_bucket += 1
        for candidate in candidates:
            if (value ^ hashes[candidate]).bit_count() <= NEAR_DUPLICATE_MAX_HAMMING:
                union_find.union(index, candidate)
        for band, key in enumerate(keys):
            bands[band][key].append(index)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(texts)):
        groups[union_find.find(index)].append(index)
    cluster_ids = [""] * len(texts)
    near_clusters = 0
    near_records = 0
    for members in groups.values():
        representative = min(fingerprints[index] for index in members)
        cluster_id = sha256_text(f"near:{representative}")[:16]
        for index in members:
            cluster_ids[index] = cluster_id
        if len(members) > 1:
            near_clusters += 1
            near_records += len(members)
    return cluster_ids, {
        "simhash_eligible_records": eligible,
        "near_duplicate_clusters": near_clusters,
        "near_duplicate_records": near_records,
        "large_bucket_checks_skipped": skipped_large_bucket,
    }


def exclusion_rows(frame: pd.DataFrame, reason: str) -> list[dict[str, str]]:
    return [
        {
            "dataset": scalar(row["dataset"]),
            "record_id": scalar(row["record_id"]),
            "reason": reason,
            "text_sha256": scalar(row["text_fingerprint"]),
            "label_raw": scalar(row["label_raw"]),
            "split_original": scalar(row["split_original"]),
        }
        for _, row in frame.iterrows()
    ]


def prepare_text_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]], dict[str, Any]]:
    working = frame.copy()
    working["title_clean"] = working["title_original"].map(clean_text)
    working["text_clean"] = working["text_original"].map(clean_text)
    working["_comparison"] = working["text_clean"].map(comparison_text)
    working["text_fingerprint"] = working["_comparison"].map(sha256_text)
    exclusions: list[dict[str, str]] = []

    invalid = working["_comparison"] == ""
    exclusions.extend(exclusion_rows(working[invalid], "empty_text_after_cleanup"))
    working = working[~invalid].copy()

    conflict_keys: set[str] = set()
    for fingerprint, group in working.groupby("text_fingerprint", sort=False):
        labels = {scalar(label) for label in group["label_raw"] if scalar(label)}
        if len(labels) > 1:
            conflict_keys.add(fingerprint)
    conflict_mask = working["text_fingerprint"].isin(conflict_keys)
    exclusions.extend(exclusion_rows(working[conflict_mask], "conflicting_labels_same_normalized_text"))
    working = working[~conflict_mask].copy()

    split_priority = {"test": 0, "validation": 1, "train": 2, "": 3}
    working["_priority"] = working["split_original"].map(lambda value: split_priority.get(scalar(value), 3))
    working = working.sort_values(["text_fingerprint", "_priority", "record_id"], kind="stable")
    duplicate_mask = working.duplicated(subset=["text_fingerprint"], keep="first")
    exclusions.extend(exclusion_rows(working[duplicate_mask], "duplicate_normalized_text"))
    working = working[~duplicate_mask].copy().reset_index(drop=True)

    cluster_ids, near_metrics = near_duplicate_clusters(
        working["text_clean"].tolist(),
        working["text_fingerprint"].tolist(),
    )
    working["near_cluster_id"] = cluster_ids
    short_records = int((working["text_clean"].str.len() < 20).sum())
    metrics = {
        "raw_records": int(len(frame)),
        "empty_text_excluded": int(invalid.sum()),
        "conflicting_text_groups": len(conflict_keys),
        "conflicting_records_excluded": int(conflict_mask.sum()),
        "normalized_duplicates_excluded": int(duplicate_mask.sum()),
        "short_text_records_retained": short_records,
        **near_metrics,
    }
    return working.drop(columns=["_comparison", "_priority"]), exclusions, metrics


def assign_grouped_splits(
    frame: pd.DataFrame,
    split_names: Sequence[str],
    ratios: Sequence[float],
    seed: int,
) -> pd.Series:
    if len(split_names) != len(ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("split names and ratios must align and sum to one")
    labels = sorted({scalar(value) for value in frame["label_raw"]})
    totals = Counter(scalar(value) for value in frame["label_raw"])
    targets = {
        split: {label: totals[label] * ratio for label in labels}
        for split, ratio in zip(split_names, ratios, strict=True)
    }
    target_total = {split: len(frame) * ratio for split, ratio in zip(split_names, ratios, strict=True)}
    current = {split: Counter() for split in split_names}
    current_total = Counter()
    assignment: dict[str, str] = {}

    groups = []
    for cluster_id, group in frame.groupby("near_cluster_id", sort=False):
        label_counts = Counter(scalar(value) for value in group["label_raw"])
        tie = sha256_text(f"{seed}:{cluster_id}")
        groups.append((scalar(cluster_id), len(group), label_counts, tie))
    groups.sort(key=lambda item: (-item[1], item[3]))

    for cluster_id, size, label_counts, _ in groups:
        best_split = split_names[0]
        best_score = -float("inf")
        for split in split_names:
            label_deficit = sum(
                count * (targets[split][label] - current[split][label]) / max(targets[split][label], 1.0)
                for label, count in label_counts.items()
            )
            total_deficit = size * (target_total[split] - current_total[split]) / max(target_total[split], 1.0)
            score = label_deficit + total_deficit
            if score > best_score:
                best_score = score
                best_split = split
        assignment[cluster_id] = best_split
        current[best_split].update(label_counts)
        current_total[best_split] += size
    return frame["near_cluster_id"].map(assignment).astype("string")


def enforce_official_split_clusters(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, str]], int]:
    priority = {"test": 0, "validation": 1, "train": 2}
    remove_indices: list[int] = []
    for _, group in frame.groupby("near_cluster_id", sort=False):
        splits = {scalar(value) for value in group["split_original"] if scalar(value)}
        if len(splits) <= 1:
            continue
        keep_split = min(splits, key=lambda value: priority.get(value, 99))
        remove_indices.extend(group.index[group["split_original"] != keep_split].tolist())
    removed = frame.loc[remove_indices] if remove_indices else frame.iloc[0:0]
    exclusions = exclusion_rows(removed, "near_duplicate_cross_split")
    return frame.drop(index=remove_indices).reset_index(drop=True), exclusions, len(remove_indices)


def split_frame(name: str, frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, list[dict[str, str]], dict[str, Any]]:
    exclusions: list[dict[str, str]] = []
    metrics: dict[str, Any] = {}
    if name == "liar":
        frame, removed, count = enforce_official_split_clusters(frame)
        exclusions.extend(removed)
        frame["split"] = frame["split_original"]
        metrics["near_duplicate_cross_split_excluded"] = count
    elif name == "bharatgather":
        frame, removed, count = enforce_official_split_clusters(frame)
        exclusions.extend(removed)
        frame["split"] = frame["split_original"]
        train_mask = frame["split_original"] == "train"
        train_assignments = assign_grouped_splits(
            frame[train_mask],
            ("train", "validation"),
            (0.9, 0.1),
            seed,
        )
        frame.loc[train_mask, "split"] = train_assignments.values
        metrics["near_duplicate_cross_split_excluded"] = count
        metrics["split_policy"] = "Official test retained; deterministic grouped validation derived from remaining official train."
    else:
        frame["split"] = assign_grouped_splits(
            frame,
            ("train", "validation", "test"),
            (0.8, 0.1, 0.1),
            seed,
        )
        metrics["split_policy"] = "Deterministic grouped 80/10/10 split, approximately stratified by raw label."
    return frame, exclusions, metrics


def distribution(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(scalar(value) for value in values).items()))


def split_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, group in frame.groupby("split", sort=True):
        result[scalar(split)] = {
            "records": int(len(group)),
            "label_raw": distribution(group["label_raw"]),
            "label_binary_fake": distribution(value for value in group["label_binary_fake"] if scalar(value)),
        }
    return result


def verify_prepared_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("prepared dataset is empty")
    if frame["text_clean"].map(bool).eq(False).any():
        raise ValueError("prepared dataset contains empty cleaned text")
    if frame["text_fingerprint"].duplicated().any():
        raise ValueError("normalized exact duplicates remain")
    cluster_splits = frame.groupby("near_cluster_id")["split"].nunique()
    if (cluster_splits > 1).any():
        raise ValueError("near-duplicate cluster leakage remains across splits")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def prepare_bs_detector(source_root: Path, processed_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sources, posts, profile = load_bs_detector(source_root)
    output_folder = processed_root / "bs_detector"
    write_csv(sources, output_folder / "source_categories.csv")
    write_csv(posts, output_folder / "unlabeled_posts.csv")
    report = {
        "role": DATASET_SPECS["bs_detector"].role,
        "raw_profile": profile,
        "prepared_source_records": int(len(sources)),
        "prepared_unlabeled_posts": int(len(posts)),
        "split_policy": "No train/validation/test split; the source is auxiliary credibility metadata plus unlabeled posts.",
    }
    return report, {
        "bs_detector/source_categories.csv": sha256_file(output_folder / "source_categories.csv"),
        "bs_detector/unlabeled_posts.csv": sha256_file(output_folder / "unlabeled_posts.csv"),
    }


def raw_archive_manifest(raw_root: Path) -> list[dict[str, Any]]:
    archive_root = raw_root / "_archives"
    return [
        {
            "path": path.relative_to(raw_root.parent.parent).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(archive_root.rglob("*"))
        if path.is_file()
    ]


def cross_dataset_overlaps(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    label_maps: dict[str, dict[str, set[str]]] = {}
    for name, frame in frames.items():
        mapping: dict[str, set[str]] = defaultdict(set)
        for _, row in frame.iterrows():
            mapping[scalar(row["text_fingerprint"])].add(scalar(row["label_raw"]))
        label_maps[name] = mapping
    results = []
    names = sorted(label_maps)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            common = set(label_maps[left]).intersection(label_maps[right])
            if not common:
                continue
            pairs: Counter[str] = Counter()
            for fingerprint in common:
                for left_label in label_maps[left][fingerprint]:
                    for right_label in label_maps[right][fingerprint]:
                        pairs[f"{left_label} <> {right_label}"] += 1
            results.append(
                {
                    "left": left,
                    "right": right,
                    "normalized_exact_text_overlap": len(common),
                    "label_pairs": dict(sorted(pairs.items())),
                }
            )
    return results


def run_pipeline(root: Path, seed: int = DEFAULT_SEED, receipt_date: str = "2026-10-17") -> dict[str, Any]:
    data_root = root / "data"
    raw_root = data_root / "raw"
    source_root = raw_root / "sources"
    interim_root = data_root / "interim"
    processed_root = data_root / "processed"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing extracted source directory: {source_root}")
    interim_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    raw_files_before = raw_archive_manifest(raw_root)
    reports: dict[str, Any] = {}
    prepared_frames: dict[str, pd.DataFrame] = {}
    all_exclusions: list[dict[str, str]] = []
    output_hashes: dict[str, str] = {}

    for name, loader in LOADERS.items():
        frame, profile = loader(source_root)
        prepared, exclusions, quality_metrics = prepare_text_frame(frame)
        prepared, split_exclusions, split_metrics = split_frame(name, prepared, seed)
        exclusions.extend(split_exclusions)
        verify_prepared_frame(prepared)
        prepared = prepared[OUTPUT_COLUMNS].sort_values(["split", "record_id"], kind="stable").reset_index(drop=True)
        prepared_frames[name] = prepared
        all_exclusions.extend(exclusions)

        folder = processed_root / name
        write_csv(prepared, folder / "cleaned.csv")
        for split, split_data in prepared.groupby("split", sort=True):
            write_csv(split_data, folder / "splits" / f"{split}.csv")

        output_hashes[f"{name}/cleaned.csv"] = sha256_file(folder / "cleaned.csv")
        for split in sorted(set(prepared["split"])):
            split_path = folder / "splits" / f"{split}.csv"
            output_hashes[f"{name}/splits/{split}.csv"] = sha256_file(split_path)

        spec = DATASET_SPECS[name]
        reports[name] = {
            "role": spec.role,
            "source_url": spec.source_url,
            "license": spec.license,
            "label_semantics": spec.label_semantics,
            "label_mapping_verified": spec.label_mapping_verified,
            "raw_profile": profile,
            "quality": quality_metrics,
            **split_metrics,
            "prepared_records": int(len(prepared)),
            "excluded_records": len(exclusions),
            "label_raw_distribution": distribution(prepared["label_raw"]),
            "label_binary_fake_distribution": distribution(
                value for value in prepared["label_binary_fake"] if scalar(value)
            ),
            "splits": split_distribution(prepared),
        }

    bs_report, bs_hashes = prepare_bs_detector(source_root, processed_root)
    reports["bs_detector"] = {
        "source_url": DATASET_SPECS["bs_detector"].source_url,
        "license": DATASET_SPECS["bs_detector"].license,
        "label_semantics": DATASET_SPECS["bs_detector"].label_semantics,
        "label_mapping_verified": False,
        **bs_report,
    }
    output_hashes.update(bs_hashes)

    exclusions_frame = pd.DataFrame(
        all_exclusions,
        columns=["dataset", "record_id", "reason", "text_sha256", "label_raw", "split_original"],
    ).sort_values(["dataset", "reason", "record_id"], kind="stable")
    write_csv(exclusions_frame, interim_root / "exclusions.csv")
    output_hashes["../interim/exclusions.csv"] = sha256_file(interim_root / "exclusions.csv")

    inspection = {
        "pipeline_version": PIPELINE_VERSION,
        "seed": seed,
        "receipt_date": receipt_date,
        "datasets": reports,
        "cross_dataset_normalized_exact_overlaps": cross_dataset_overlaps(prepared_frames),
    }
    inspection_path = interim_root / "inspection.json"
    inspection_path.write_text(json.dumps(inspection, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_hashes["../interim/inspection.json"] = sha256_file(inspection_path)

    raw_files_after = raw_archive_manifest(raw_root)
    if raw_files_before != raw_files_after:
        raise RuntimeError("Raw archive hashes changed during preparation")

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "seed": seed,
        "receipt_date": receipt_date,
        "split_ratios_default": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "near_duplicate_method": {
            "method": "64-bit SimHash over normalized word trigrams (tokens for short texts)",
            "maximum_hamming_distance": NEAR_DUPLICATE_MAX_HAMMING,
            "minimum_tokens": NEAR_DUPLICATE_MIN_TOKENS,
            "maximum_tokens_considered": NEAR_DUPLICATE_MAX_TOKENS,
            "maximum_features_sampled": NEAR_DUPLICATE_MAX_FEATURES,
            "note": "Potential near-duplicate clusters are kept within one split; raw data is unchanged.",
        },
        "raw_archives": raw_files_after,
        "datasets": reports,
        "cross_dataset_normalized_exact_overlaps": inspection["cross_dataset_normalized_exact_overlaps"],
        "output_sha256": dict(sorted(output_hashes.items())),
    }
    manifest_path = processed_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--receipt-date", default="2026-10-17")
    args = parser.parse_args()
    manifest = run_pipeline(args.root.resolve(), seed=args.seed, receipt_date=args.receipt_date)
    summary = {
        "pipeline_version": manifest["pipeline_version"],
        "seed": manifest["seed"],
        "datasets": {
            name: {
                "prepared_records": report.get("prepared_records", report.get("prepared_source_records")),
                "excluded_records": report.get("excluded_records", 0),
                "splits": report.get("splits", {}),
            }
            for name, report in manifest["datasets"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
