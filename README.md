# Cancer Severity Prediction

This project predicts a cancer severity score from patient demographics, cancer context, and risk-factor inputs.

## What Was Improved

- Uses a full `scikit-learn` `Pipeline` instead of manual label mappings.
- Uses `ColumnTransformer` with numeric and categorical preprocessing.
- Compares multiple models with cross-validation.
- Tunes the best model with `RandomizedSearchCV`.
- Saves the model, preprocessing, metrics, and metadata together.
- Streamlit app can load the advanced pipeline directly.
- Legacy `model.pkl` still works as a fallback.

## Train The Advanced Model

Run this from the project folder:

```powershell
python src/train_advanced_model.py --data "C:\path\to\global_cancer_patients_2015_2024.csv"
```

The script creates:

```text
artifacts/cancer_severity_pipeline.joblib
artifacts/metrics.json
artifacts/feature_importance.csv
```

## Run The App

```powershell
streamlit run app.py
```

The app prefers the advanced artifact:

```text
artifacts/cancer_severity_pipeline.joblib
```

If that file is missing, it falls back to:

```text
model.pkl
```

## Important Note

This project is for machine-learning demonstration and learning. It is not a medical diagnosis or clinical decision tool.
