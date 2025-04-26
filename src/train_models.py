"""
Training utilities for multiple regressors.
Saves each fitted Pipeline to models/<ModelName>_model.pkl
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

# Set up paths and directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
VAE_MODEL_DIR = BASE_DIR / "models" / "vae"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VAE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def build_preprocessor(df: pd.DataFrame):
    """Build preprocessing pipeline for numerical and categorical features"""
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.difference(["SurvivalMonths"])
    cat_cols = df.select_dtypes(include=["object"]).columns
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preproc

def train_all_models(df: pd.DataFrame, use_vae: bool = False) -> pd.DataFrame:
    """Train multiple regression models and return their performance metrics"""
    X = df.drop(columns="SurvivalMonths")
    y = df["SurvivalMonths"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df["Stage"]
    )

    pre = build_preprocessor(df)

    # Define models with their hyperparameters
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

        # Save trained model
        out_dir = VAE_MODEL_DIR if use_vae else MODEL_DIR
        out_path = out_dir / f"{name}_model.pkl"
        joblib.dump(pipe, out_path)
        print(f"✔ Saved {name} → {out_path.relative_to(BASE_DIR)}")

    return pd.DataFrame(scores).T

# Allow CLI usage -------------------------------------------------------
if __name__ == "__main__":
    # Train on original data
    print("Training models on original data...")
    original_df = pd.read_csv(BASE_DIR / "data" / "synthetic_data.csv")
    original_scores = train_all_models(original_df, use_vae=False)
    print("\nOriginal data model scores:")
    print(original_scores)
    
    # Train on VAE-augmented data
    print("\nTraining models on VAE-augmented data...")
    vae_df = pd.read_csv(BASE_DIR / "data" / "vae_augmented_data.csv")
    vae_scores = train_all_models(vae_df, use_vae=True)
    print("\nVAE-augmented data model scores:")
    print(vae_scores)