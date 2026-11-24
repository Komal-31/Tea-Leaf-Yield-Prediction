
from pathlib import Path

import mlflow.sklearn
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "tea_leaf_v1"

FEATURES = [
    "worker_exp_yr",
    "rain_mm_week",
    "slope_deg",
    "temp_c",
    "bush_age_yr",
    "hours_worked",
    "flush",
    "section",
]


@st.cache_resource
def load_model():
    return mlflow.sklearn.load_model(str(MODEL_PATH))


st.set_page_config(
    page_title="Tea Leaf Yield Prediction",
    page_icon="🍃",
    layout="centered",
)

st.title("🍃 Tea Leaf Yield Prediction")
st.write("Predict tea leaf yield per worker using machine learning.")

st.subheader("Enter Plantation Details")

with st.form("prediction_form"):
    worker_exp_yr = st.number_input(
        "Worker Experience (years)", min_value=0.0, value=48.0
    )
    rain_mm_week = st.number_input(
        "Weekly Rainfall (mm)", min_value=0.0, value=285.0
    )
    slope_deg = st.number_input(
        "Slope (degrees)", min_value=0.0, max_value=90.0, value=21.0
    )
    temp_c = st.number_input(
        "Temperature (°C)", value=15.9
    )
    bush_age_yr = st.number_input(
        "Bush Age (years)", min_value=0.0, value=10.0
    )
    hours_worked = st.number_input(
        "Hours Worked", min_value=0.0, value=5.4
    )
    flush = st.text_input("Flush", value="monsoon")
    section = st.text_input("Section", value="River Side")

    submitted = st.form_submit_button("Predict Yield")

if submitted:
    try:
        model = load_model()

        input_data = pd.DataFrame(
            [[
                worker_exp_yr,
                rain_mm_week,
                slope_deg,
                temp_c,
                bush_age_yr,
                hours_worked,
                flush,
                section,
            ]],
            columns=FEATURES,
        )

        prediction = model.predict(input_data)[0]

        st.success("Prediction completed!")
        st.metric(
            "Predicted Tea Leaf Yield",
            f"{float(prediction):.2f} kg per worker",
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
