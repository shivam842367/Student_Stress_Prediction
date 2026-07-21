"""Utility helpers for the student stress prediction project."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"


def ensure_directories() -> None:
    """Create required directories if they do not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: str | Path) -> Any:
    """Load JSON data from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: str | Path) -> None:
    """Persist JSON data to disk."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Persist a Python object using joblib."""
    joblib.dump(obj, path)


def load_pickle(path: str | Path) -> Any:
    """Load a Python object from disk."""
    return joblib.load(path)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a dataframe to CSV."""
    df.to_csv(path, index=False)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a dataframe from CSV."""
    return pd.read_csv(path)


def get_feature_metadata() -> dict[str, list[str]]:
    """Return the modeling feature groups used by the training and app pipelines."""
    numeric_features = [
        "age",
        "sleep_hours",
        "study_hours",
        "exercise_days",
        "social_hours",
        "screen_time_hours",
        "commute_minutes",
        "part_time_hours",
        "caffeine_intake",
        "mindfulness_minutes",
        "financial_stress_score",
        "academic_pressure_score",
        "family_support_score",
        "peer_support_score",
        "self_efficacy_score",
        "weekly_meals",
        "home_noise_score",
        "internet_speed",
        "assignment_load",
        "class_attendance",
    ]
    categorical_features = [
        "persona",
        "major",
        "housing_type",
        "relationship_status",
        "transportation_mode",
        "part_time_job",
        "study_location",
        "club_membership",
        "diet_type",
        "scholarship_status",
    ]
    return {"numeric": numeric_features, "categorical": categorical_features}


def build_default_input() -> dict[str, Any]:
    """Create a default user input dictionary for the Streamlit app."""
    metadata = get_feature_metadata()
    defaults = {
        "age": 20,
        "sleep_hours": 6.5,
        "study_hours": 5.0,
        "exercise_days": 3.0,
        "social_hours": 4.0,
        "screen_time_hours": 4.5,
        "commute_minutes": 20.0,
        "part_time_hours": 0.0,
        "caffeine_intake": 1.5,
        "mindfulness_minutes": 10.0,
        "financial_stress_score": 3.0,
        "academic_pressure_score": 4.0,
        "family_support_score": 6.0,
        "peer_support_score": 5.0,
        "self_efficacy_score": 6.0,
        "weekly_meals": 14.0,
        "home_noise_score": 3.0,
        "internet_speed": 100.0,
        "assignment_load": 6.0,
        "class_attendance": 80.0,
        "persona": "Average",
        "major": "Computer Science",
        "housing_type": "Dorm",
        "relationship_status": "Single",
        "transportation_mode": "Walk",
        "part_time_job": "No",
        "study_location": "Library",
        "club_membership": "Yes",
        "diet_type": "Balanced",
        "scholarship_status": "No",
    }
    return {key: defaults[key] for key in metadata["numeric"] + metadata["categorical"]}


def get_model_artifact_paths() -> dict[str, Path]:
    """Return the standard artifact paths used by the project."""
    return {
        "dataset": DATA_DIR / "student_dataset.csv",
        "dictionary": DATA_DIR / "data_dictionary.md",
        "preprocessor": MODELS_DIR / "preprocessor_config.json",
        "scaler": MODELS_DIR / "scaler.pkl",
        "target_encoder": MODELS_DIR / "target_encoder.pkl",
        "best_model": MODELS_DIR / "best_model.pkl",
        "metrics": MODELS_DIR / "model_metrics.json",
        "feature_importance": MODELS_DIR / "feature_importance.csv",
        "shap_summary": MODELS_DIR / "shap_summary.png",
    }
