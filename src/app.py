
import streamlit as st, pandas as pd, joblib, os
st.title('Pancreatic Cancer Survival Predictor')

uploaded = st.file_uploader('Upload patient CSV', type='csv')
if uploaded:
    data = pd.read_csv(uploaded)
else:
    st.info('Using sample synthetic dataset.')
    data = pd.read_csv('data/synthetic_data.csv')

model_choice = st.selectbox('Choose model', ['ElasticNet','SVR','DecisionTree','LightGBM','XGBoost'])
model_path = f'models/{model_choice}_model.pkl'
if not os.path.exists(model_path):
    st.error('Model file not found. Train models first.')
else:
    model = joblib.load(model_path)
    pred = model.predict(data.drop(columns=['SurvivalMonths'], errors='ignore'))
    st.write('Predicted survival months:', pred)
