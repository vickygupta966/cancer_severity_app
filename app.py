import streamlit as st
import pickle
import numpy as np
import gdown
import os

# ------------------ DOWNLOAD MODEL ------------------

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1FkOc55IjpeVL9-6Cf_vS-zAmPiJ2z8tr"
    gdown.download(url, MODEL_PATH, quiet=False)

# ------------------ LOAD MODEL ------------------

model = pickle.load(open(MODEL_PATH, 'rb'))

# ------------------ UI ------------------

st.title("Cancer Severity Prediction 🔬")
st.write("Predict cancer severity based on patient data.")

st.header("Enter Patient Details")

# ------------------ MAPPINGS ------------------

# Gender
gender_map = {"Female": 0, "Male": 1}
gender = gender_map[st.selectbox("Gender", list(gender_map.keys()))]

# Country
country_map = {
    "UK": 0,
    "China": 1,
    "Pakistan": 2,
    "Brazil": 3,
    "USA": 4
}
country = country_map[st.selectbox("Country", list(country_map.keys()))]

# Cancer Type
cancer_type_map = {
    "Lung": 0,
    "Leukemia": 1,
    "Breast": 2,
    "Colon": 3,
    "Skin": 4,
    "Liver": 5
}
cancer_type = cancer_type_map[st.selectbox("Cancer Type", list(cancer_type_map.keys()))]

# Cancer Stage
cancer_stage_map = {
    "Stage 0": 0,
    "Stage I": 1,
    "Stage II": 2,
    "Stage III": 3,
    "Stage IV": 4
}
cancer_stage = cancer_stage_map[st.selectbox("Cancer Stage", list(cancer_stage_map.keys()))]

# ------------------ NUMERIC INPUTS ------------------

age = st.number_input("Age", 0, 100)
year = st.number_input("Year", 2015, 2024)

genetic_risk = st.slider("Genetic Risk", 0.0, 10.0)
air_pollution = st.slider("Air Pollution", 0.0, 10.0)
alcohol = st.slider("Alcohol Use", 0.0, 10.0)
smoking = st.slider("Smoking", 0.0, 10.0)
obesity = st.slider("Obesity Level", 0.0, 10.0)

# ------------------ PREDICTION ------------------

if st.button("Predict"):

    input_data = np.array([[age, gender, country, year,
                            genetic_risk, air_pollution, alcohol,
                            smoking, obesity, cancer_type,
                            cancer_stage]])

    prediction = model.predict(input_data)[0]

    if prediction > 7:
        st.error(f"High Cancer Severity ⚠️ ({prediction:.2f})")
    elif prediction > 4:
        st.warning(f"Moderate Severity ⚠️ ({prediction:.2f})")
    else:
        st.success(f"Low Severity ✅ ({prediction:.2f})")