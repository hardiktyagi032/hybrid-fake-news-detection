"""Low-risk helpers for reproducibility and conservative text comparison."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison_text(value: object) -> str:
    """Normalize only whitespace and case for duplicate analysis.

    This function is not a model preprocessing step and must not overwrite source text.
    """

    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()
