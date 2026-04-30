from __future__ import annotations

import pickle
from pathlib import Path
import requests 

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ARTIFACT_DIR = Path("artifacts")
ADVANCED_MODEL_PATH = ARTIFACT_DIR / "cancer_severity_pipeline.joblib"
LEGACY_MODEL_PATH = Path("model.pkl")

# 👉 PASTE YOUR HUGGINGFACE LINK HERE
MODEL_URL = "https://huggingface.co/vickygupta966/cancer-prediction-model/blob/main/model.pkl"

FEATURES = [
    "Age",
    "Gender",
    "Country_Region",
    "Year",
    "Genetic_Risk",
    "Air_Pollution",
    "Alcohol_Use",
    "Smoking",
    "Obesity_Level",
    "Cancer_Type",
    "Cancer_Stage",
]


@st.cache_resource
def load_model():
    # 1. Try advanced model (if exists)
    if ADVANCED_MODEL_PATH.exists():
        bundle = joblib.load(ADVANCED_MODEL_PATH)
        return bundle["model"], bundle.get("metadata", {}), "advanced"

    # 2. Download model.pkl from Hugging Face if not exists
    if not LEGACY_MODEL_PATH.exists():
        with st.spinner("Downloading model... please wait ⏳"):
            try:
                r = requests.get(MODEL_URL)

                if r.status_code != 200:
                    st.error("❌ Failed to download model from Hugging Face")
                    st.stop()

                with open("model.pkl", "wb") as f:
                    f.write(r.content)

            except Exception as e:
                st.error(f"❌ Download error: {e}")
                st.stop()

    # 3. Load model.pkl
    try:
        with open("model.pkl", "rb") as file:
            return pickle.load(file), {}, "legacy"
    except Exception as e:
        st.error(f"❌ Model loading error: {e}")
        st.stop()


def severity_band(score: float) -> tuple[str, str]:
    if score >= 7:
        return "High severity", "error"
    if score >= 4:
        return "Moderate severity", "warning"
    return "Low severity", "success"


def build_input_row() -> pd.DataFrame:
    st.sidebar.header("Patient Details")

    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=45)
    year = st.sidebar.number_input("Year", min_value=2015, max_value=2024, value=2024)

    gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
    country = st.sidebar.selectbox(
        "Country / Region",
        ["Australia", "Brazil", "Canada", "China", "France", "Germany", "India", "Pakistan", "UK", "USA"],
    )
    cancer_type = st.sidebar.selectbox(
        "Cancer Type",
        ["Breast", "Cervical", "Colon", "Leukemia", "Liver", "Lung", "Prostate", "Skin"],
    )
    cancer_stage = st.sidebar.selectbox(
        "Cancer Stage",
        ["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"],
    )

    st.sidebar.header("Risk Factors")
    genetic_risk = st.sidebar.slider("Genetic Risk", 0.0, 10.0, 5.0, 0.1)
    air_pollution = st.sidebar.slider("Air Pollution", 0.0, 10.0, 5.0, 0.1)
    alcohol = st.sidebar.slider("Alcohol Use", 0.0, 10.0, 5.0, 0.1)
    smoking = st.sidebar.slider("Smoking", 0.0, 10.0, 5.0, 0.1)
    obesity = st.sidebar.slider("Obesity Level", 0.0, 10.0, 5.0, 0.1)

    return pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender,
                "Country_Region": country,
                "Year": year,
                "Genetic_Risk": genetic_risk,
                "Air_Pollution": air_pollution,
                "Alcohol_Use": alcohol,
                "Smoking": smoking,
                "Obesity_Level": obesity,
                "Cancer_Type": cancer_type,
                "Cancer_Stage": cancer_stage,
            }
        ],
        columns=FEATURES,
    )


def predict(model, row: pd.DataFrame, mode: str) -> float:
    if mode == "advanced":
        return float(np.clip(model.predict(row)[0], 0, 10))

    # Legacy fallback keeps the old model usable, but it is less reliable because
    # its categorical encodings were not saved with the model.
    gender_map = {"Female": 0, "Male": 1, "Other": 2}
    country_map = {
        "UK": 0,
        "China": 1,
        "Pakistan": 2,
        "Brazil": 3,
        "USA": 4,
        "Australia": 5,
        "Canada": 6,
        "France": 7,
        "Germany": 8,
        "India": 9,
    }
    cancer_type_map = {
        "Lung": 0,
        "Leukemia": 1,
        "Breast": 2,
        "Colon": 3,
        "Skin": 4,
        "Liver": 5,
        "Cervical": 6,
        "Prostate": 7,
    }
    stage_map = {"Stage 0": 0, "Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}

    r = row.iloc[0]
    legacy_input = np.array(
        [
            [
                r["Age"],
                gender_map[r["Gender"]],
                country_map[r["Country_Region"]],
                r["Year"],
                r["Genetic_Risk"],
                r["Air_Pollution"],
                r["Alcohol_Use"],
                r["Smoking"],
                r["Obesity_Level"],
                cancer_type_map[r["Cancer_Type"]],
                stage_map[r["Cancer_Stage"]],
            ]
        ]
    )
    return float(np.clip(model.predict(legacy_input)[0], 0, 10))


st.set_page_config(page_title="Cancer Severity Prediction", page_icon="🔬", layout="wide")

st.title("Cancer Severity Prediction")
st.caption("Advanced ML pipeline app with preprocessing, validation, and safer prediction inputs.")

model, metadata, mode = load_model()
input_row = build_input_row()

if mode == "missing":
    st.error(
        "No model found. Run `python src/train_advanced_model.py --data <your_csv_path>` "
        "to create artifacts/cancer_severity_pipeline.joblib."
    )
    st.stop()

if mode == "legacy":
    st.warning(
        "Using legacy model.pkl. For better predictions, train the advanced pipeline so categorical encoders "
        "are saved with the model."
    )

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Current Input")
    st.dataframe(input_row, hide_index=True, use_container_width=True)

    if st.button("Predict Severity", type="primary"):
        score = predict(model, input_row, mode)
        label, state = severity_band(score)

        if state == "error":
            st.error(f"{label}: {score:.2f} / 10")
        elif state == "warning":
            st.warning(f"{label}: {score:.2f} / 10")
        else:
            st.success(f"{label}: {score:.2f} / 10")

        st.progress(min(max(score / 10, 0), 1))

with right:
    st.subheader("Model Information")
    st.write(f"Active model: `{mode}`")

    if metadata:
        st.metric("Test R²", f"{metadata.get('test_r2', 0):.3f}")
        st.metric("Test RMSE", f"{metadata.get('test_rmse', 0):.3f}")
        st.metric("Test MAE", f"{metadata.get('test_mae', 0):.3f}")
        st.write("Best model:", metadata.get("best_model_name", "Unknown"))

    st.info(
        "This project is for ML demonstration and learning. It should not be used as medical advice "
        "or as a clinical decision system."
    )
