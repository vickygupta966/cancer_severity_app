from __future__ import annotations

import pickle
from pathlib import Path
import requests 

import numpy as np
import pandas as pd
import streamlit as st


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
    """Download and load the legacy model from Hugging Face"""
    
    # Download model if not exists
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
    
    # Load and return the model
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
            return model
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


def predict(model, input_data: dict) -> float:
    """Make prediction using the legacy model"""
    
    # Mappings for categorical variables
    gender_map = {"Female": 0, "Male": 1, "Other": 2}
    country_map = {
        "UK": 0, "China": 1, "Pakistan": 2, "Brazil": 3, "USA": 4,
        "Australia": 5, "Canada": 6, "France": 7, "Germany": 8, "India": 9,
    }
    cancer_type_map = {
        "Lung": 0, "Leukemia": 1, "Breast": 2, "Colon": 3, "Skin": 4,
        "Liver": 5, "Cervical": 6, "Prostate": 7,
    }
    stage_map = {"Stage 0": 0, "Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}

    # Convert input to numeric values
    features = np.array([[
        input_data["Age"],
        gender_map[input_data["Gender"]],
        country_map[input_data["Country_Region"]],
        input_data["Year"],
        input_data["Genetic_Risk"],
        input_data["Air_Pollution"],
        input_data["Alcohol_Use"],
        input_data["Smoking"],
        input_data["Obesity_Level"],
        cancer_type_map[input_data["Cancer_Type"]],
        stage_map[input_data["Cancer_Stage"]],
    ]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    return float(np.clip(prediction, 0, 10))


# Streamlit UI
st.set_page_config(page_title="Cancer Severity Prediction", page_icon="🔬", layout="wide")

st.title("🎗️ Cancer Severity Prediction")
st.caption("ML model for predicting cancer severity based on patient data")

# Load model
model = load_model()

# Create input form
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

# Create input dictionary
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

# Display current input
st.subheader("📊 Current Patient Input")
input_df = pd.DataFrame([input_data])
st.dataframe(input_df, hide_index=True, use_container_width=True)

# Predict button
if st.button("🔮 Predict Severity", type="primary", use_container_width=True):
    with st.spinner("Calculating prediction..."):
        score = predict(model, input_data)
        label, state = severity_band(score)

    # Display result
    if state == "error":
        st.error(f"🚨 {label}: {score:.2f} / 10")
    elif state == "warning":
        st.warning(f"⚠️ {label}: {score:.2f} / 10")
    else:
        st.success(f"✅ {label}: {score:.2f} / 10")

    st.progress(min(max(score / 10, 0), 1))
    st.caption(f"Score range: 0 (lowest severity) - 10 (highest severity)")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **Disclaimer:** This tool is for educational purposes only. "
    "Always consult healthcare professionals for medical advice."
)