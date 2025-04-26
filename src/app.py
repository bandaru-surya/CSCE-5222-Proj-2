# ----------------------------------------------------------------------
# Streamlit Dashboard – Pancreatic Cancer Survival Prediction
# ----------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths and directories
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
VAE_MODEL_DIR = BASE_DIR / "models" / "vae"
RESULTS_DIR = BASE_DIR / "results"
DEFAULT_CSV = DATA_DIR / "synthetic_data.csv"
VAE_CSV = DATA_DIR / "vae_augmented_data.csv"

VAE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Pancreatic Cancer Survival Predictor", layout="wide")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
apply_fe = st.sidebar.checkbox("Apply feature‑engineering pipeline", value=True)
use_aug = st.sidebar.checkbox("Use VAE‑augmented models", value=False)

# Load appropriate dataset
if use_aug and VAE_CSV.exists():
    df = pd.read_csv(VAE_CSV)
    st.sidebar.success("Using VAE-augmented dataset.")
else:
    df = pd.read_csv(DEFAULT_CSV)
    st.sidebar.info("Using original dataset.")

# Model selection
model_dir = VAE_MODEL_DIR if use_aug else MODEL_DIR
avail_models = [p.stem.replace("_model", "") for p in model_dir.glob("*_model.pkl")]
chosen = st.sidebar.multiselect("Select model(s) to run", avail_models, default=avail_models[:3])
run_btn = st.sidebar.button("🚀 Run prediction")

# Single patient prediction form
with st.sidebar.expander("🔍 Single patient (optional)"):
    single_mode = st.checkbox("Enable form prediction")
    if single_mode:
        patient = {}
        for col in df.columns:
            if col == "SurvivalMonths": continue
            if df[col].dtype == "object":
                patient[col] = st.selectbox(col, sorted(df[col].unique()))
            else:
                min_,max_ = float(df[col].min()), float(df[col].max())
                patient[col] = st.slider(col, min_, max_, float(np.median(df[col])))
        if st.button("Predict for patient"):
            mdl_path = model_dir / f"{chosen[0]}_model.pkl"
            pipe = joblib.load(mdl_path)
            pred = pipe.predict(pd.DataFrame([patient]))[0]
            st.success(f"Predicted survival: **{pred:.1f} months**")

# Main dashboard
st.title("Pancreatic Cancer Survival Dashboard")
st.subheader("Dataset preview")
st.dataframe(df.head())

c1, c2 = st.columns(2)
with c1: st.metric("Patients", len(df))
with c2: st.metric("Features", df.shape[1]-1)

# Run predictions and show results
if run_btn:
    from train_models import build_preprocessor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    X = df.drop(columns="SurvivalMonths", errors="ignore")
    y = df["SurvivalMonths"] if "SurvivalMonths" in df else None

    results = []
    feature_importances = {}
    for m in chosen:
        mdl_path = model_dir / f"{m}_model.pkl"
        if not mdl_path.exists():
            st.error(f"{m} model not found. Retrain first."); st.stop()
        pipe = joblib.load(mdl_path)
        pred = pipe.predict(X)
        mae = mean_absolute_error(y, pred) if y is not None else np.nan
        mse = mean_squared_error(y, pred) if y is not None else np.nan
        results.append({"Model": m, "MAE": mae, "MSE": mse})

        # Get feature importance for tree-based models
        inner = pipe.named_steps["model"]
        if hasattr(inner, "feature_importances_"):
            pre = pipe.named_steps["prep"]
            feat_names = list(pre.transformers_[0][2]) + list(pre.named_transformers_["cat"].get_feature_names_out())
            feature_importances[m] = pd.Series(inner.feature_importances_, index=feat_names)

    res_df = pd.DataFrame(results).set_index("Model")
    st.subheader("Performance")
    st.dataframe(res_df.style.format("{:.2f}"))

    # Visualize results
    st.subheader("MAE Comparison")
    fig, ax = plt.subplots(figsize=(6,3))
    res_df["MAE"].plot(kind="barh", ax=ax, color="#4c8eda")
    ax.set_xlabel("MAE (months)")
    st.pyplot(fig)

    # Show feature importance for each model
    if feature_importances:
        st.subheader("Feature Importance")
        cols = st.columns(len(feature_importances))
        for (model_name, fi), col in zip(feature_importances.items(), cols):
            with col:
                fig, ax = plt.subplots(figsize=(4,3))
                fi.sort_values(ascending=True).tail(10).plot(kind="barh", ax=ax)
                ax.set_title(f"{model_name}")
                st.pyplot(fig)

    # Show t-SNE visualization if available
    tsne_img = RESULTS_DIR / "tsne_plot.png"
    if tsne_img.exists():
        st.subheader("t‑SNE 2‑D Projection")
        st.image(str(tsne_img))

    # Allow downloading results
    csv_dl = res_df.to_csv().encode()
    st.download_button("Download metrics CSV", csv_dl, file_name="model_metrics.csv", mime="text/csv")
else:
    st.caption("← Configure options in the sidebar and click **Run prediction**.")