"""
Training utilities for multiple regressors.
Saves each fitted Pipeline to  models/<ModelName>_model.pkl
"""

import pandas as pd, numpy as np, joblib, os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# ----------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent   # …/Group26_Project
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)          # create if missing
# ----------------------------------------------------------------------

def build_preprocessor(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.difference(
        ["SurvivalMonths"]
    )
    cat_cols = df.select_dtypes(include=["object"]).columns
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preproc

def train_all_models(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns="SurvivalMonths")
    y = df["SurvivalMonths"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df["Stage"]
    )

    pre = build_preprocessor(df)

    models = {
        "ElasticNet":  ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "SVR":         SVR(C=10, gamma="scale"),
        "DecisionTree":DecisionTreeRegressor(max_depth=5, random_state=42),
        "LightGBM":    LGBMRegressor(n_estimators=200, learning_rate=0.05),
        "XGBoost":     XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4),
    }

    scores = {}
    for name, est in models.items():
        pipe = Pipeline([("prep", pre), ("model", est)])
        pipe.fit(Xtr, ytr)

        y_pred = pipe.predict(Xte)
        scores[name] = {
            "MAE": mean_absolute_error(yte, y_pred),
            "MSE": mean_squared_error(yte, y_pred),
        }

        # --- save -------------------------------------------------------
        out_path = MODEL_DIR / f"{name}_model.pkl"
        joblib.dump(pipe, out_path)
        print(f"✔ Saved {name} → {out_path.relative_to(BASE_DIR)}")

    return pd.DataFrame(scores).T

# Allow CLI usage -------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(BASE_DIR / "data" / "synthetic_data.csv")
    print(train_all_models(df))