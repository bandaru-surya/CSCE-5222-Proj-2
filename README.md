
# Group 26 – Pancreatic Cancer Survival Prediction

This repository contains code and resources for predicting pancreatic cancer patient survival times using feature engineering and Variational Autoencoder (VAE)–based data augmentation.

## Directory Structure
```
src/               Core Python modules
data/              Synthetic (and anonymized real) datasets
models/            Saved trained model objects
results/           Plots and evaluation metrics
presentation/      Slides (PPTX & PDF)
Group26_Code.pdf   Code snapshot required for submission
```

## Quick Start
```bash
pip install -r requirements.txt
python src/synthetic_data.py           # generate synthetic dataset
python -m pip install streamlit torch lightgbm xgboost joblib
python -c "from src.train_models import train_all_models; import pandas as pd, os; df=pd.read_csv('data/synthetic_data.csv'); train_all_models(df)"
streamlit run src/app.py
```

## Authors
* Sai Sri Surya Bandaru
