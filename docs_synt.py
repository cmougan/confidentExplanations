# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from tools.xaiUtils import SelectiveAbstentionExplanations
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# %%
X, y = make_blobs(n_samples=2_000, centers=2, n_features=2, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, stratify=y, test_size=0.5, random_state=0
)
# %%
detector = SelectiveAbstentionExplanations(
    model=XGBClassifier(n_estimators=20), gmodel=LogisticRegression(), cov=0.8
)
detector.fit(X_tr, y_tr)
# %%
roc_auc_score(detector.create_error(X_te, y_te), detector.gpredict_proba(X_te)[:, 1])

# %%
