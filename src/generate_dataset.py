"""Generate a realistic synthetic dataset for student lifestyle and stress prediction."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.utils import DATA_DIR, ensure_directories, save_dataframe
except ImportError:  # pragma: no cover - support direct execution
    from utils import DATA_DIR, ensure_directories, save_dataframe


def sample_persona(rng: np.random.Generator) -> str:
    """Sample a student persona with realistic probabilities."""
    personas = ["Topper", "Average", "Athlete", "Struggling", "Working Student", "Social Butterfly"]
    probabilities = [0.18, 0.25, 0.16, 0.14, 0.14, 0.13]
    return rng.choice(personas, p=probabilities)


def clip(value: float, low: float, high: float) -> float:
    """Clamp a value to a range."""
    return max(low, min(high, value))


def build_dataset(n_students: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset with 50+ meaningful features and a hidden stress target."""
    rng = np.random.default_rng(seed)
    records = []

    for _ in range(n_students):
        persona = sample_persona(rng)

        if persona == "Topper":
            sleep_mean, sleep_sd = 7.0, 0.7
            study_mean, study_sd = 6.8, 1.4
            exercise_mean, exercise_sd = 3.0, 1.4
            social_mean, social_sd = 3.0, 1.3
            screen_mean, screen_sd = 4.2, 1.0
            commute_mean, commute_sd = 18.0, 8.0
            part_time_mean, part_time_sd = 0.0, 0.0
            caffeine_mean, caffeine_sd = 1.0, 0.7
            mindfulness_mean, mindfulness_sd = 18.0, 8.0
            financial_mean, financial_sd = 2.5, 1.4
            pressure_mean, pressure_sd = 5.2, 1.2
            family_support_mean, family_support_sd = 7.0, 1.2
            peer_support_mean, peer_support_sd = 6.5, 1.2
            self_efficacy_mean, self_efficacy_sd = 7.5, 1.2
            assignment_mean, assignment_sd = 5.0, 1.3
            attendance_mean, attendance_sd = 90.0, 6.0
        elif persona == "Athlete":
            sleep_mean, sleep_sd = 6.2, 0.8
            study_mean, study_sd = 4.8, 1.2
            exercise_mean, exercise_sd = 6.2, 1.2
            social_mean, social_sd = 3.8, 1.3
            screen_mean, screen_sd = 4.6, 1.1
            commute_mean, commute_sd = 25.0, 10.0
            part_time_mean, part_time_sd = 0.0, 0.0
            caffeine_mean, caffeine_sd = 1.4, 0.8
            mindfulness_mean, mindfulness_sd = 12.0, 6.0
            financial_mean, financial_sd = 3.0, 1.5
            pressure_mean, pressure_sd = 4.8, 1.1
            family_support_mean, family_support_sd = 6.2, 1.3
            peer_support_mean, peer_support_sd = 6.3, 1.3
            self_efficacy_mean, self_efficacy_sd = 6.8, 1.3
            assignment_mean, assignment_sd = 4.2, 1.2
            attendance_mean, attendance_sd = 86.0, 7.0
        elif persona == "Working Student":
            sleep_mean, sleep_sd = 5.8, 0.8
            study_mean, study_sd = 4.0, 1.2
            exercise_mean, exercise_sd = 2.0, 1.0
            social_mean, social_sd = 2.5, 1.1
            screen_mean, screen_sd = 5.3, 1.2
            commute_mean, commute_sd = 35.0, 12.0
            part_time_mean, part_time_sd = 18.0, 5.0
            caffeine_mean, caffeine_sd = 2.0, 1.1
            mindfulness_mean, mindfulness_sd = 8.0, 4.0
            financial_mean, financial_sd = 6.3, 1.6
            pressure_mean, pressure_sd = 6.4, 1.1
            family_support_mean, family_support_sd = 5.0, 1.4
            peer_support_mean, peer_support_sd = 4.8, 1.2
            self_efficacy_mean, self_efficacy_sd = 5.6, 1.3
            assignment_mean, assignment_sd = 6.0, 1.5
            attendance_mean, attendance_sd = 78.0, 8.0
        elif persona == "Struggling":
            sleep_mean, sleep_sd = 5.0, 0.9
            study_mean, study_sd = 3.0, 1.1
            exercise_mean, exercise_sd = 1.4, 0.8
            social_mean, social_sd = 2.0, 1.0
            screen_mean, screen_sd = 6.4, 1.3
            commute_mean, commute_sd = 30.0, 10.0
            part_time_mean, part_time_sd = 10.0, 4.0
            caffeine_mean, caffeine_sd = 2.6, 1.1
            mindfulness_mean, mindfulness_sd = 5.0, 3.0
            financial_mean, financial_sd = 7.0, 1.5
            pressure_mean, pressure_sd = 7.4, 1.0
            family_support_mean, family_support_sd = 4.0, 1.4
            peer_support_mean, peer_support_sd = 3.8, 1.2
            self_efficacy_mean, self_efficacy_sd = 4.3, 1.3
            assignment_mean, assignment_sd = 8.0, 1.4
            attendance_mean, attendance_sd = 68.0, 9.0
        else:
            sleep_mean, sleep_sd = 6.5, 0.8
            study_mean, study_sd = 4.8, 1.2
            exercise_mean, exercise_sd = 2.8, 1.1
            social_mean, social_sd = 4.2, 1.5
            screen_mean, screen_sd = 5.0, 1.1
            commute_mean, commute_sd = 20.0, 9.0
            part_time_mean, part_time_sd = 4.0, 3.0
            caffeine_mean, caffeine_sd = 1.6, 0.9
            mindfulness_mean, mindfulness_sd = 12.0, 6.0
            financial_mean, financial_sd = 3.5, 1.5
            pressure_mean, pressure_sd = 5.6, 1.2
            family_support_mean, family_support_sd = 6.0, 1.2
            peer_support_mean, peer_support_sd = 5.8, 1.3
            self_efficacy_mean, self_efficacy_sd = 6.2, 1.3
            assignment_mean, assignment_sd = 5.5, 1.3
            attendance_mean, attendance_sd = 84.0, 7.0

        age = int(round(clip(rng.normal(20.5, 2.0), 17, 30)))
        sleep_hours = clip(rng.normal(sleep_mean, sleep_sd), 4.0, 10.0)
        study_hours = clip(rng.normal(study_mean, study_sd), 1.0, 15.0)
        exercise_days = clip(rng.normal(exercise_mean, exercise_sd), 0.0, 7.0)
        social_hours = clip(rng.normal(social_mean, social_sd), 0.0, 12.0)
        screen_time_hours = clip(rng.normal(screen_mean, screen_sd), 1.0, 10.0)
        commute_minutes = clip(rng.normal(commute_mean, commute_sd), 5.0, 90.0)
        part_time_hours = clip(rng.normal(part_time_mean, part_time_sd), 0.0, 40.0)
        caffeine_intake = clip(rng.normal(caffeine_mean, caffeine_sd), 0.0, 5.0)
        mindfulness_minutes = clip(rng.normal(mindfulness_mean, mindfulness_sd), 0.0, 60.0)
        financial_stress_score = clip(rng.normal(financial_mean, financial_sd), 1.0, 10.0)
        academic_pressure_score = clip(rng.normal(pressure_mean, pressure_sd), 1.0, 10.0)
        family_support_score = clip(rng.normal(family_support_mean, family_support_sd), 1.0, 10.0)
        peer_support_score = clip(rng.normal(peer_support_mean, peer_support_sd), 1.0, 10.0)
        self_efficacy_score = clip(rng.normal(self_efficacy_mean, self_efficacy_sd), 1.0, 10.0)
        weekly_meals = int(round(clip(rng.normal(14.0, 3.0), 3, 21)))
        home_noise_score = int(round(clip(rng.normal(3.0, 1.6), 1, 10)))
        internet_speed = int(round(clip(rng.normal(90.0, 25.0), 20, 200)))
        assignment_load = clip(rng.normal(assignment_mean, assignment_sd), 1.0, 10.0)
        class_attendance = int(round(clip(rng.normal(attendance_mean, attendance_sd), 40, 100)))
        sleep_quality_score = int(round(clip(rng.normal(6.4, 1.8), 1, 10)))
        appetite_change = int(round(clip(rng.normal(2.0, 1.4), 0, 10)))
        headaches_per_week = int(round(clip(rng.normal(1.7, 1.2), 0, 10)))
        anxiety_score = int(round(clip(rng.normal(4.0, 2.2), 1, 10)))
        depression_score = int(round(clip(rng.normal(3.3, 2.0), 1, 10)))
        monthly_income = int(round(clip(rng.normal(1800.0, 900.0), 200, 5000)))
        monthly_expenses = int(round(clip(rng.normal(1600.0, 700.0), 300, 5000)))
        extracurricular_hours = clip(rng.normal(4.5, 2.2), 0.0, 20.0)
        course_difficulty = int(round(clip(rng.normal(6.0, 1.8), 1, 10)))
        gpa = clip(rng.normal(3.2, 0.5), 1.5, 4.0)
        relationship_status = rng.choice(["Single", "In Relationship", "Married"], p=[0.72, 0.24, 0.04])
        housing_type = rng.choice(["Dorm", "Apartment", "Family Home"], p=[0.45, 0.35, 0.20])
        transportation_mode = rng.choice(["Walk", "Bus", "Bike", "Car"], p=[0.35, 0.35, 0.10, 0.20])
        major = rng.choice(["Computer Science", "Business", "Engineering", "Arts", "Nursing", "Biology", "Psychology"], p=[0.22, 0.18, 0.16, 0.12, 0.11, 0.10, 0.11])
        part_time_job = rng.choice(["No", "Yes"], p=[0.55, 0.45])
        study_location = rng.choice(["Library", "Dorm Room", "Cafe", "Home"], p=[0.30, 0.25, 0.20, 0.25])
        club_membership = rng.choice(["No", "Yes"], p=[0.52, 0.48])
        diet_type = rng.choice(["Balanced", "Vegetarian", "Fast Food", "High Protein"], p=[0.45, 0.15, 0.20, 0.20])
        scholarship_status = rng.choice(["No", "Yes"], p=[0.75, 0.25])
        social_media_minutes = int(round(clip(rng.normal(120.0, 60.0), 30, 300)))
        gaming_hours = clip(rng.normal(2.0, 1.8), 0.0, 10.0)
        sleep_debt = max(0.0, 8.0 - sleep_hours) + 0.3 * part_time_hours / 10.0
        academic_load = assignment_load + 0.6 * study_hours
        social_demand = social_hours + 0.3 * social_media_minutes / 60.0
        hidden_score = (
            -0.55 * (sleep_hours - 7.0)
            + 0.25 * (academic_pressure_score - 5.0)
            + 0.18 * (financial_stress_score - 5.0)
            + 0.16 * (screen_time_hours - 4.0)
            + 0.12 * (caffeine_intake - 1.3)
            + 0.13 * (assignment_load - 5.0)
            + 0.10 * (part_time_hours / 10.0)
            + 0.12 * (anxiety_score - 4.0)
            + 0.10 * (sleep_debt)
            + 0.08 * (home_noise_score - 3.0)
            + 0.09 * (academic_load / 10.0)
            + 0.07 * (social_demand / 8.0)
            - 0.10 * (family_support_score - 5.0)
            - 0.08 * (self_efficacy_score - 5.0)
            + 0.08 * (mindfulness_minutes < 10) * 1.0
            + 0.05 * (class_attendance < 80) * 1.0
            + rng.normal(0.0, 0.9)
        )
        p_stress = 1.0 / (1.0 + math.exp(-hidden_score))
        noise_draw = rng.uniform(0.0, 1.0)
        if p_stress + noise_draw * 0.16 < 0.34:
            stress_level = "Low"
        elif p_stress + noise_draw * 0.16 > 0.72:
            stress_level = "High"
        else:
            stress_level = "Moderate"

        records.append(
            {
                "student_id": f"STU-{len(records) + 1:05d}",
                "persona": persona,
                "age": age,
                "sleep_hours": round(sleep_hours, 2),
                "study_hours": round(study_hours, 2),
                "exercise_days": round(exercise_days, 2),
                "social_hours": round(social_hours, 2),
                "screen_time_hours": round(screen_time_hours, 2),
                "commute_minutes": round(commute_minutes, 2),
                "part_time_hours": round(part_time_hours, 2),
                "caffeine_intake": round(caffeine_intake, 2),
                "mindfulness_minutes": round(mindfulness_minutes, 2),
                "financial_stress_score": round(financial_stress_score, 2),
                "academic_pressure_score": round(academic_pressure_score, 2),
                "family_support_score": round(family_support_score, 2),
                "peer_support_score": round(peer_support_score, 2),
                "self_efficacy_score": round(self_efficacy_score, 2),
                "weekly_meals": weekly_meals,
                "home_noise_score": home_noise_score,
                "internet_speed": internet_speed,
                "assignment_load": round(assignment_load, 2),
                "class_attendance": class_attendance,
                "sleep_quality_score": sleep_quality_score,
                "appetite_change": appetite_change,
                "headaches_per_week": headaches_per_week,
                "anxiety_score": anxiety_score,
                "depression_score": depression_score,
                "monthly_income": monthly_income,
                "monthly_expenses": monthly_expenses,
                "extracurricular_hours": round(extracurricular_hours, 2),
                "course_difficulty": course_difficulty,
                "gpa": round(gpa, 2),
                "major": major,
                "housing_type": housing_type,
                "relationship_status": relationship_status,
                "transportation_mode": transportation_mode,
                "part_time_job": part_time_job,
                "study_location": study_location,
                "club_membership": club_membership,
                "diet_type": diet_type,
                "scholarship_status": scholarship_status,
                "social_media_minutes": social_media_minutes,
                "gaming_hours": round(gaming_hours, 2),
                "Stress_Level": stress_level,
            }
        )

    df = pd.DataFrame.from_records(records)
    return df


def create_data_dictionary(df: pd.DataFrame, path: Path) -> None:
    """Write a markdown data dictionary for the generated dataset."""
    features = list(df.columns)
    lines = ["# Data Dictionary", "", "This dataset contains synthetic student lifestyle and stress behavior profiles.", "", "## Target", "", "- Stress_Level: categorical outcome with Low, Moderate, High.", "", "## Features", ""]
    for column in features:
        if column == "Stress_Level":
            continue
        if column == "student_id":
            lines.append(f"- {column}: unique student identifier.")
        else:
            lines.append(f"- {column}: feature used to infer the student stress profile.")
    lines.extend(["", "## Notes", "", "- The dataset is synthetic and designed to avoid target leakage.", "- The target is generated from hidden weighted signals and random noise."])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Generate the dataset and save it to disk."""
    ensure_directories()
    df = build_dataset()
    save_dataframe(df, DATA_DIR / "student_dataset.csv")
    create_data_dictionary(df, DATA_DIR / "data_dictionary.md")
    print(f"Saved synthetic dataset to {DATA_DIR / 'student_dataset.csv'}")


if __name__ == "__main__":
    main()
