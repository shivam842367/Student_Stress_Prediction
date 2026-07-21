"""Streamlit dashboard for student stress prediction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import predict_from_input
from src.utils import build_default_input, get_model_artifact_paths

st.set_page_config(page_title="Student Stress Prediction", page_icon="🧠", layout="wide")


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, object]:
    """Load model comparison metrics from disk."""
    path = get_model_artifact_paths()["metrics"]
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load the generated student dataset for analytics."""
    path = get_model_artifact_paths()["dataset"]
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def render_sidebar_form() -> dict[str, object]:
    """Create the sidebar form for student profile inputs."""
    with st.sidebar:
        st.title("Student Profile")
        st.caption("Enter a realistic student profile to estimate stress risk.")
        defaults = build_default_input()
        values = {}
        values["age"] = st.number_input("Age", 17, 30, int(defaults["age"]))
        values["sleep_hours"] = st.slider("Sleep Hours", 4.0, 10.0, float(defaults["sleep_hours"]))
        values["study_hours"] = st.slider("Study Hours", 1.0, 15.0, float(defaults["study_hours"]))
        values["exercise_days"] = st.slider("Exercise Days", 0.0, 7.0, float(defaults["exercise_days"]))
        values["social_hours"] = st.slider("Social Hours", 0.0, 12.0, float(defaults["social_hours"]))
        values["screen_time_hours"] = st.slider("Screen Time Hours", 1.0, 10.0, float(defaults["screen_time_hours"]))
        values["commute_minutes"] = st.slider("Commute Minutes", 5.0, 90.0, float(defaults["commute_minutes"]))
        values["part_time_hours"] = st.slider("Part-time Hours", 0.0, 40.0, float(defaults["part_time_hours"]))
        values["caffeine_intake"] = st.slider("Caffeine Intake", 0.0, 5.0, float(defaults["caffeine_intake"]))
        values["mindfulness_minutes"] = st.slider("Mindfulness Minutes", 0.0, 60.0, float(defaults["mindfulness_minutes"]))
        values["financial_stress_score"] = st.slider("Financial Stress Score", 1.0, 10.0, float(defaults["financial_stress_score"]))
        values["academic_pressure_score"] = st.slider("Academic Pressure", 1.0, 10.0, float(defaults["academic_pressure_score"]))
        values["family_support_score"] = st.slider("Family Support", 1.0, 10.0, float(defaults["family_support_score"]))
        values["peer_support_score"] = st.slider("Peer Support", 1.0, 10.0, float(defaults["peer_support_score"]))
        values["self_efficacy_score"] = st.slider("Self-efficacy", 1.0, 10.0, float(defaults["self_efficacy_score"]))
        values["weekly_meals"] = st.slider("Weekly Meals", 3, 21, int(defaults["weekly_meals"]))
        values["home_noise_score"] = st.slider("Home Noise", 1, 10, int(defaults["home_noise_score"]))
        values["internet_speed"] = st.slider("Internet Speed", 20, 200, int(defaults["internet_speed"]))
        values["assignment_load"] = st.slider("Assignment Load", 1.0, 10.0, float(defaults["assignment_load"]))
        values["class_attendance"] = st.slider("Class Attendance", 40, 100, int(defaults["class_attendance"]))
        values["persona"] = st.selectbox("Persona", ["Topper", "Average", "Athlete", "Struggling", "Working Student", "Social Butterfly"], index=1)
        values["major"] = st.selectbox("Major", ["Computer Science", "Business", "Engineering", "Arts", "Nursing", "Biology", "Psychology"], index=0)
        values["housing_type"] = st.selectbox("Housing Type", ["Dorm", "Apartment", "Family Home"], index=0)
        values["relationship_status"] = st.selectbox("Relationship Status", ["Single", "In Relationship", "Married"], index=0)
        values["transportation_mode"] = st.selectbox("Transportation", ["Walk", "Bus", "Bike", "Car"], index=0)
        values["part_time_job"] = st.selectbox("Part-time Job", ["No", "Yes"], index=0)
        values["study_location"] = st.selectbox("Study Location", ["Library", "Dorm Room", "Cafe", "Home"], index=0)
        values["club_membership"] = st.selectbox("Club Membership", ["No", "Yes"], index=1)
        values["diet_type"] = st.selectbox("Diet Type", ["Balanced", "Vegetarian", "Fast Food", "High Protein"], index=0)
        values["scholarship_status"] = st.selectbox("Scholarship", ["No", "Yes"], index=0)
        return values


def render_recommendations(prediction: str, probabilities: dict[str, float]) -> None:
    """Display personalized recommendations based on the predicted stress level."""
    st.subheader("Personalized Recommendations")
    if prediction == "High":
        recommendations = [
            "Prioritize sleep recovery and reduce late-night screen time.",
            "Schedule a short mindfulness routine and speak with an academic advisor if load remains high.",
            "Consider reducing part-time hours or adding lighter study blocks to lower overload.",
        ]
    elif prediction == "Moderate":
        recommendations = [
            "Maintain steady routines around sleep, meals and exercise.",
            "Protect one daily focus block and set boundaries around social media use.",
            "Use peer support and counseling resources early rather than waiting for escalation.",
        ]
    else:
        recommendations = [
            "Maintain healthy habits and continue structured routines.",
            "Use reflection prompts to sustain balance and monitor stress triggers.",
            "Keep building resilience through exercise, mindfulness and social support.",
        ]
    for item in recommendations:
        st.write(f"- {item}")

    st.caption(f"Confidence: {max(probabilities.values()):.1%}")


def main() -> None:
    """Render the Streamlit dashboard."""
    st.title("Student Lifestyle & Stress Prediction")
    st.markdown("A production-ready dashboard for estimating student stress risk from lifestyle and support indicators.")

    payload = render_sidebar_form()
    if st.button("Predict Stress Level", type="primary"):
        prediction, probabilities = predict_from_input(payload)
        st.subheader("Prediction Outcome")
        st.metric(label="Predicted Stress Level", value=prediction)
        st.progress(0.5 if prediction == "Moderate" else 0.8 if prediction == "High" else 0.3)
        st.write("### Probability Breakdown")
        st.bar_chart(pd.DataFrame(probabilities, index=[0]).T)
        render_recommendations(prediction, probabilities)

    with st.expander("Dataset Overview"):
        df = load_dataset()
        if df.empty:
            st.info("Run the data generation step to create the dataset.")
        else:
            st.dataframe(df.head(10))
            col1, col2, col3 = st.columns(3)
            col1.metric("Students", len(df))
            col2.metric("Features", len(df.columns) - 1)
            col3.metric("Stress Classes", df["Stress_Level"].nunique())

    with st.expander("Model Performance"):
        metrics = load_metrics()
        if metrics:
            st.json(metrics)
        else:
            st.info("Model training has not been completed yet.")


if __name__ == "__main__":
    main()
