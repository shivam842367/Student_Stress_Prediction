"""Command-line prediction entrypoint for the trained student stress model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.preprocess import load_preprocessor_artifacts, prepare_features
    from src.utils import build_default_input, get_model_artifact_paths, load_pickle
except ImportError:  # pragma: no cover - support direct execution
    from preprocess import load_preprocessor_artifacts, prepare_features
    from utils import build_default_input, get_model_artifact_paths, load_pickle


def predict_from_input(payload: dict[str, object]) -> tuple[str, dict[str, float]]:
    """Predict stress level from a user-provided payload."""
    encoders, numeric_features, categorical_features, scaler, target_encoder = load_preprocessor_artifacts()
    model = load_pickle(get_model_artifact_paths()["best_model"])

    frame = pd.DataFrame([payload])
    X = prepare_features(frame, encoders, numeric_features, categorical_features, scaler)
    probabilities = model.predict_proba(X)[0]
    classes = target_encoder.classes_
    probs = {str(label): round(float(prob), 3) for label, prob in zip(classes, probabilities)}
    prediction = target_encoder.inverse_transform([int(np.argmax(probabilities))])[0]
    return str(prediction), probs


def main() -> None:
    """Run a sample prediction using the default input profile."""
    payload = build_default_input()
    prediction, probabilities = predict_from_input(payload)
    print(f"Prediction: {prediction}")
    print(json.dumps(probabilities, indent=2))


if __name__ == "__main__":
    import numpy as np

    main()
