from __future__ import annotations

import pickle
from pathlib import Path
import requests 
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ARTIFACT_DIR = Path("artifacts")
ADVANCED_MODEL_PATH = ARTIFACT_DIR / "cancer_severity_pipeline.joblib"
LEGACY_MODEL_PATH = Path("model.pkl")

# Hugging Face model link
MODEL_URL = "https://huggingface.co/vickygupta966/cancer-prediction-model/resolve/main/model.pkl"

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
    """Load model from local file or download from Hugging Face"""
    
    # For Streamlit Cloud, always ensure model is downloaded
    if not LEGACY_MODEL_PATH.exists():
        with st.spinner("Downloading model from Hugging Face... ⏳"):
            try:
                response = requests.get(MODEL_URL)
                if response.status_code == 200:
                    with open("model.pkl", "wb") as f:
                        f.write(response.content)
                    st.success("✅ Model downloaded successfully!")
                else:
                    st.error(f"❌ Failed to download model. Status: {response.status_code}")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Download error: {e}")
                st.stop()
    
    # Try advanced model first (if exists in artifacts)
    if ADVANCED_MODEL_PATH.exists():
        try:
            bundle = joblib.load(ADVANCED_MODEL_PATH)
            return bundle["model"], bundle.get("metadata", {}), "advanced"
        except Exception as e:
            st.warning(f"Could not load advanced model: {e}")
    
    # Load legacy model.pkl
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
            return model, {}, "legacy"
    except Exception as e:
        st.error(f"❌ Model loading error: {e}")
        st.stop()


def severity_band(score: float) -> tuple[str, str]:
    """Convert severity score to label and color"""
    if score >= 7:
        return "High severity", "error"
    if score >= 4:
        return "Moderate severity", "warning"
    return "Low severity", "success"


def build_input_row() -> pd.DataFrame:
    """Create input row from sidebar widgets"""
    st.sidebar.header("Patient Details")

    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=45, step=1)
    year = st.sidebar.number_input("Year", min_value=2015, max_value=2024, value=2024, step=1)

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

    # Create DataFrame with correct column order
    input_data = {
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
    
    return pd.DataFrame([input_data])


def predict(model, row: pd.DataFrame, mode: str) -> float:
    """Make prediction with proper data handling"""
    try:
        if mode == "advanced":
            # Create a clean copy with the correct column order
            X_input = row[FEATURES].copy()
            
            # The pipeline expects a DataFrame with column names
            # Pass the DataFrame directly, not numpy array
            prediction = model.predict(X_input)[0]
            
            return float(np.clip(prediction, 0, 10))

        # Legacy mode (old model.pkl)
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
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        # Debug info to help troubleshoot
        st.write("Debug Info:")
        st.write(f"- Mode: {mode}")
        st.write(f"- Input columns: {list(row.columns)}")
        if mode == "advanced":
            st.write(f"- Model type: {type(model)}")
            if hasattr(model, 'named_steps'):
                st.write(f"- Pipeline steps: {list(model.named_steps.keys())}")
        raise


# Streamlit UI Configuration
st.set_page_config(page_title="Cancer Severity Prediction", page_icon="🔬", layout="wide")

st.title("🎗️ Cancer Severity Prediction")
st.caption("Advanced ML pipeline for predicting cancer severity based on patient data")

# Load model
model, metadata, mode = load_model()

# Build input form
input_row = build_input_row()

# Display warnings for legacy mode
if mode == "legacy":
    st.warning("⚠️ Using legacy model. For better predictions, use the advanced pipeline with saved encoders.")

# Main layout
left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("📊 Current Patient Input")
    st.dataframe(input_row, hide_index=True, use_container_width=True)

    if st.button("🔮 Predict Severity", type="primary", use_container_width=True):
        with st.spinner("Calculating prediction..."):
            score = predict(model, input_row, mode)
            label, state = severity_band(score)

        # Display result with appropriate styling
        if state == "error":
            st.error(f"🚨 {label}: {score:.2f} / 10")
        elif state == "warning":
            st.warning(f"⚠️ {label}: {score:.2f} / 10")
        else:
            st.success(f"✅ {label}: {score:.2f} / 10")

        # Progress bar
        st.progress(min(max(score / 10, 0), 1))
        
        # Additional info
        st.caption(f"Score range: 0 (lowest severity) - 10 (highest severity)")

with right:
    st.subheader("ℹ️ Model Information")
    st.write(f"**Active model:** `{mode}`")

    if metadata:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test R²", f"{metadata.get('test_r2', 0):.3f}")
        with col2:
            st.metric("Test RMSE", f"{metadata.get('test_rmse', 0):.3f}")
        with col3:
            st.metric("Test MAE", f"{metadata.get('test_mae', 0):.3f}")
        
        if metadata.get("best_model_name"):
            st.write(f"**Best model:** {metadata.get('best_model_name')}")

    st.info(
        "📌 **Disclaimer:** This project is for educational and demonstration purposes only. "
        "It should not be used as medical advice or a clinical decision support system. "
        "Always consult healthcare professionals for medical concerns."
    )
    
    st.divider()
    st.caption("Built with ❤️ using Streamlit and Machine Learning")