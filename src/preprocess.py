"""Preprocessing utilities for train/test split, encoding, scaling and artifact export."""

from __future__ import annotations

import pickle
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from src.utils import MODELS_DIR, get_feature_metadata, load_dataframe, save_json, save_pickle
except ImportError:  # pragma: no cover - support direct execution
    from utils import MODELS_DIR, get_feature_metadata, load_dataframe, save_json, save_pickle

from src.utils import load_pickle


class Preprocessor:
    """Fit and transform student data for machine learning models."""

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data.copy()
        self.metadata = get_feature_metadata()
        self.numeric_features = self.metadata["numeric"]
        self.categorical_features = self.metadata["categorical"]
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        self.categorical_encoders: dict[str, dict[str, int]] = {}
        self.feature_names: list[str] = []

    def fit(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Fit the preprocessor on the training split and return encoded features."""
        data = self.data.copy()
        features = self.numeric_features + self.categorical_features
        X = data[features]
        y = data["Stress_Level"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        for column in self.categorical_features:
            train_values = X_train[column].astype(str)
            unique_values = sorted(train_values.dropna().unique())
            encoder = {value: index for index, value in enumerate(unique_values)}
            self.categorical_encoders[column] = encoder

        X_train_processed = self._prepare_frame(X_train)
        X_test_processed = self._prepare_frame(X_test)
        self.scaler.fit(X_train_processed)
        self.target_encoder.fit(y_train)

        X_train_scaled = self._scale_frame(X_train_processed)
        X_test_scaled = self._scale_frame(X_test_processed)
        self.feature_names = list(X_train_scaled.columns)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Encode and impute a dataframe using fitted mappings."""
        processed = frame.copy()
        for column in self.categorical_features:
            processed[column] = processed[column].astype(str)
            processed[column] = processed[column].map(self.categorical_encoders.get(column, {}))
            processed[column] = processed[column].fillna(-1).astype(int)

        for column in self.numeric_features:
            processed[column] = pd.to_numeric(processed[column], errors="coerce")
            processed[column] = processed[column].fillna(processed[column].median())

        return processed[self.numeric_features + self.categorical_features]

    def _scale_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaler to a prepared dataframe."""
        return pd.DataFrame(self.scaler.transform(frame), columns=frame.columns, index=frame.index)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform a new dataframe using stored encoders and scaler."""
        processed = self._prepare_frame(frame)
        return self._scale_frame(processed)

    def save_artifacts(self, path: Path | None = None) -> None:
        """Persist preprocessing artifacts to disk."""
        target_path = path or MODELS_DIR
        save_json(self.categorical_encoders, target_path / "categorical_encoders.json")
        save_json({"numeric": self.numeric_features, "categorical": self.categorical_features}, target_path / "feature_columns.json")
        save_pickle(self.scaler, target_path / "scaler.pkl")
        save_pickle(self.target_encoder, target_path / "target_encoder.pkl")


def preprocess_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the dataset, fit the preprocessor and save artifacts."""
    data = load_dataframe(path)
    preprocessor = Preprocessor(data)
    X_train, X_test, y_train, y_test = preprocessor.fit()
    preprocessor.save_artifacts()
    return X_train, X_test, y_train, y_test


def load_preprocessor_artifacts() -> tuple[dict[str, dict[str, int]], list[str], list[str], Any, Any]:
    """Load preprocessing artifacts for inference."""
    encoders = json.loads((MODELS_DIR / "categorical_encoders.json").read_text(encoding="utf-8"))
    feature_info = json.loads((MODELS_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    scaler = load_pickle(MODELS_DIR / "scaler.pkl")
    target_encoder = load_pickle(MODELS_DIR / "target_encoder.pkl")
    return encoders, feature_info["numeric"], feature_info["categorical"], scaler, target_encoder


def prepare_features(frame: pd.DataFrame, encoders: dict[str, dict[str, int]], numeric_features: list[str], categorical_features: list[str], scaler: Any) -> pd.DataFrame:
    """Prepare a feature dataframe for model inference and app predictions."""
    prepared = frame.copy()
    for column in categorical_features:
        prepared[column] = prepared[column].astype(str)
        prepared[column] = prepared[column].map(encoders.get(column, {}))
        prepared[column] = prepared[column].fillna(-1).astype(int)

    for column in numeric_features:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared[column] = prepared[column].fillna(prepared[column].median())

    prepared = prepared[numeric_features + categorical_features]
    return pd.DataFrame(scaler.transform(prepared), columns=prepared.columns, index=prepared.index)
