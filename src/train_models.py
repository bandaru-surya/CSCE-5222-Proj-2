import pandas as pd, numpy as np, joblib
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

def build_preprocessor(df):
    num_cols = df.select_dtypes(include=['float64','int64']).columns.drop('SurvivalMonths')
    cat_cols = df.select_dtypes(include=['object']).columns
    preproc = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    return preproc, num_cols, cat_cols

def train_all_models(df):
    X = df.drop(columns=['SurvivalMonths'])
    y = df['SurvivalMonths']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preproc,_,_ = build_preprocessor(df)
    models = {
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'SVR': SVR(C=10, gamma='scale'),
        'DecisionTree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=200, learning_rate=0.05),
        'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4)
    }
    results = {}
    for name, model in models.items():
        pipe = Pipeline([('pre', preproc), ('model', model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            'MAE': mean_absolute_error(y_test, pred),
            'MSE': mean_squared_error(y_test, pred)
        }
        joblib.dump(pipe, f'../models/{name}_model.pkl')
    return pd.DataFrame(results).T
