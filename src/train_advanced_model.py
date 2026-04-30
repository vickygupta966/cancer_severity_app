from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Target_Severity_Score"
DROP_COLUMNS = ["Patient_ID", "Survival_Years", "Treatment_Cost_USD", "Age_Group"]

CATEGORICAL_FEATURES = ["Gender", "Country_Region", "Cancer_Type", "Cancer_Stage"]
NUMERIC_FEATURES = [
    "Age",
    "Year",
    "Genetic_Risk",
    "Air_Pollution",
    "Alcohol_Use",
    "Smoking",
    "Obesity_Level",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class ModelReport:
    best_model_name: str
    test_r2: float
    test_rmse: float
    test_mae: float
    train_rows: int
    test_rows: int
    features: list[str]
    target: str
    top_features: list[dict[str, float | str]]
    cv_summary: dict[str, dict[str, float]]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    data.columns = [col.strip().replace(" ", "_") for col in data.columns]

    missing = [col for col in FEATURES + [TARGET] if col not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return data


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_models(random_state: int) -> dict[str, Any]:
    return {
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=random_state,
            max_iter=250,
            learning_rate=0.05,
            l2_regularization=0.01,
        ),
        "random_forest": RandomForestRegressor(
            random_state=random_state,
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            random_state=random_state,
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
    }


def make_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )


def evaluate_candidates(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> tuple[str, dict[str, dict[str, float]]]:
    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
    }

    summaries: dict[str, dict[str, float]] = {}
    for name, model in candidate_models(random_state).items():
        scores = cross_validate(
            make_pipeline(model),
            x_train,
            y_train,
            cv=5,
            scoring=scoring,
            n_jobs=-1,
        )

        summaries[name] = {
            "cv_r2_mean": float(scores["test_r2"].mean()),
            "cv_r2_std": float(scores["test_r2"].std()),
            "cv_mae_mean": float(-scores["test_neg_mae"].mean()),
            "cv_rmse_mean": float(-scores["test_neg_rmse"].mean()),
        }

    best_name = max(summaries, key=lambda key: summaries[key]["cv_r2_mean"])
    return best_name, summaries


def tune_best_model(name: str, random_state: int) -> RandomizedSearchCV:
    if name == "hist_gradient_boosting":
        estimator = make_pipeline(HistGradientBoostingRegressor(random_state=random_state))
        param_distributions = {
            "model__max_iter": [150, 250, 350, 500],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [10, 20, 35, 50],
            "model__l2_regularization": [0.0, 0.01, 0.05, 0.1],
        }
    elif name == "extra_trees":
        estimator = make_pipeline(ExtraTreesRegressor(random_state=random_state, n_jobs=-1))
        param_distributions = {
            "model__n_estimators": [250, 400, 600],
            "model__max_depth": [None, 10, 20, 35],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.7, 1.0],
        }
    else:
        estimator = make_pipeline(RandomForestRegressor(random_state=random_state, n_jobs=-1))
        param_distributions = {
            "model__n_estimators": [250, 400, 600],
            "model__max_depth": [None, 10, 20, 35],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.7, 1.0],
        }

    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )


def prepare_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    clean_data = data.copy()

    for col in NUMERIC_FEATURES + [TARGET]:
        clean_data[col] = pd.to_numeric(clean_data[col], errors="coerce")

    clean_data = clean_data.dropna(subset=[TARGET])
    clean_data[TARGET] = clean_data[TARGET].clip(0, 10)

    x = clean_data[FEATURES]
    y = clean_data[TARGET]
    return x, y


def feature_importance(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> list[dict[str, float | str]]:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=8,
        random_state=random_state,
        scoring="r2",
        n_jobs=-1,
    )

    rows = []
    for feature, importance, std in zip(FEATURES, result.importances_mean, result.importances_std):
        rows.append(
            {
                "feature": feature,
                "importance_mean": float(importance),
                "importance_std": float(std),
            }
        )

    return sorted(rows, key=lambda row: row["importance_mean"], reverse=True)[:15]


def train(data_path: Path, output_dir: Path, random_state: int) -> ModelReport:
    data = load_data(data_path)
    x, y = prepare_xy(data)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    best_name, cv_summary = evaluate_candidates(x_train, y_train, random_state)
    search = tune_best_model(best_name, random_state)
    search.fit(x_train, y_train)

    best_model: Pipeline = search.best_estimator_
    predictions = np.clip(best_model.predict(x_test), 0, 10)

    report = ModelReport(
        best_model_name=best_name,
        test_r2=float(r2_score(y_test, predictions)),
        test_rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
        test_mae=float(mean_absolute_error(y_test, predictions)),
        train_rows=int(len(x_train)),
        test_rows=int(len(x_test)),
        features=FEATURES,
        target=TARGET,
        top_features=feature_importance(best_model, x_test, y_test, random_state),
        cv_summary=cv_summary,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": best_model,
        "metadata": {
            **asdict(report),
            "best_params": search.best_params_,
            "data_path": str(data_path),
        },
    }

    joblib.dump(artifact, output_dir / "cancer_severity_pipeline.joblib")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(artifact["metadata"], file, indent=2)

    pd.DataFrame(report.top_features).to_csv(output_dir / "feature_importance.csv", index=False)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an advanced cancer severity prediction pipeline.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to global_cancer_patients_2015_2024 CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where model and metrics will be saved.",
    )
    parser.add_argument("--random-state", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train(args.data, args.output_dir, args.random_state)

    print("\nTraining complete")
    print(f"Best model: {report.best_model_name}")
    print(f"Test R2: {report.test_r2:.4f}")
    print(f"Test RMSE: {report.test_rmse:.4f}")
    print(f"Test MAE: {report.test_mae:.4f}")
    print(f"Saved model: {args.output_dir / 'cancer_severity_pipeline.joblib'}")
    print(f"Saved metrics: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
