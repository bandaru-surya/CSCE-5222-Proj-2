
"Feature engineering: correlation filter, RFE, PCA"
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA

def remove_correlated(df: pd.DataFrame, threshold: float = 0.85):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

def select_features_rfe(X, y, n_features=5):
    selector = RFE(Lasso(alpha=0.01), n_features_to_select=n_features).fit(X, y)
    selected = X.columns[selector.support_]
    return selected

def pca_transform(X, n_components=2):
    pca = PCA(n_components=n_components).fit(X)
    X_pca = pca.transform(X)
    return X_pca, pca
