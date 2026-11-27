
import streamlit as st

from serving import (
    build_feature_dataframe,
    load_serving_config,
    load_serving_model,
)

SERVING_CONFIG = load_serving_config()
MODEL_VERSION = SERVING_CONFIG["model_version"]
MODEL_DIR = SERVING_CONFIG["model_dir"]
FEATURES = SERVING_CONFIG["features"]
MAX_HOURS_WORKED = SERVING_CONFIG["max_hours_worked"]
MIN_RAIN_MM_WEEK = SERVING_CONFIG["min_rain_mm_week"]


@st.cache_resource
def load_model():
    return load_serving_model(SERVING_CONFIG)


def validate_inputs(rain_mm_week: float, hours_worked: float) -> None:
    """Validate assignment-specific bounds for Streamlit inputs."""
    if rain_mm_week < MIN_RAIN_MM_WEEK:
        raise ValueError(f"Weekly rainfall must be >= {MIN_RAIN_MM_WEEK:g} mm.")
    if hours_worked < 0 or hours_worked > MAX_HOURS_WORKED:
        raise ValueError(f"Hours worked must be between 0 and {MAX_HOURS_WORKED:g}.")


def predict_yield(payload: dict) -> float:
    """Validate inputs, load cached champion model, and return predicted yield."""
    validate_inputs(
        rain_mm_week=float(payload["rain_mm_week"]),
        hours_worked=float(payload["hours_worked"]),
    )
    model = load_model()
    input_data = build_feature_dataframe(payload, FEATURES)
    return float(model.predict(input_data)[0])


st.set_page_config(
    page_title="Tea Leaf Yield Prediction",
    page_icon="🍃",
    layout="centered",
)

st.title("🍃 Tea Leaf Yield Prediction")
st.write("Predict tea leaf yield per worker using machine learning.")
st.caption(
    f"Model Version: v{MODEL_VERSION} (@{SERVING_CONFIG['champion_alias']}) — `{MODEL_DIR}`"
)

try:
    load_model()
except Exception as exc:
    st.error(f"Failed to load champion model v{MODEL_VERSION}: {exc}")

st.subheader("Enter Plantation Details")

with st.form("prediction_form"):
    worker_exp_yr = st.number_input(
        "Worker Experience (years)", min_value=0.0, value=12.0
    )
    rain_mm_week = st.number_input(
        "Weekly Rainfall (mm)", min_value=MIN_RAIN_MM_WEEK, value=150.0
    )
    slope_deg = st.number_input(
        "Slope (degrees)", min_value=0.0, max_value=90.0, value=17.0
    )
    temp_c = st.number_input(
        "Temperature (°C)", value=25.0
    )
    bush_age_yr = st.number_input(
        "Bush Age (years)", min_value=0.0, value=10.0
    )
    hours_worked = st.number_input(
        "Hours Worked",
        min_value=0.0,
        max_value=MAX_HOURS_WORKED,
        value=6.0,
    )
    flush = st.text_input("Flush", value="monsoon")
    section = st.text_input("Section", value="River Side")

    submitted = st.form_submit_button("Predict Yield")

if submitted:
    try:
        payload = {
            "worker_exp_yr": worker_exp_yr,
            "rain_mm_week": rain_mm_week,
            "slope_deg": slope_deg,
            "temp_c": temp_c,
            "bush_age_yr": bush_age_yr,
            "hours_worked": hours_worked,
            "flush": flush,
            "section": section,
        }
        prediction = predict_yield(payload)

        st.success("Prediction completed!")
        st.metric(
            f"Predicted Tea Leaf Yield (Model v{MODEL_VERSION})",
            f"{prediction:.2f} kg per worker",
        )

    except ValueError as e:
        st.error(f"Invalid input: {e}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
