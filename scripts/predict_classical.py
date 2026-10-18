"""Run one trusted local classical-model artifact against article text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from backend.ml.classical_model import fake_probabilities


def predict(model_path: Path, title: str, text: str, threshold: float = 0.5) -> dict[str, Any]:
    """Return one JSON-safe prediction from a project-generated artifact."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    document = f"{title.strip()}\n\n{text.strip()}".strip()
    if not document:
        raise ValueError("A title or article body is required")

    # joblib uses pickle internally. Only project-generated, trusted artifacts
    # may be loaded here; never load an artifact supplied by an unknown party.
    model = joblib.load(model_path)
    fake_probability = float(fake_probabilities(model, [document])[0])
    predicted_class = "fake" if fake_probability >= threshold else "real"
    confidence = fake_probability if predicted_class == "fake" else 1.0 - fake_probability
    return {
        "model": model_path.parent.name,
        "fake_probability": fake_probability,
        "threshold": threshold,
        "predicted_class": predicted_class,
        "confidence": confidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--text", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(predict(args.model_path, args.title, args.text, args.threshold), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
