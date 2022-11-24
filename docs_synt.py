# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from tools.xaiUtils import SelectiveAbstentionExplanations
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# %%

for disp in [0.001, 0.1, 0.2]:
    x1, x2 = np.random.multivariate_normal([0, 0], [[disp, 0], [0, disp]], 1000).T
    x3, x4 = np.random.multivariate_normal([1, 1], [[disp, 0], [0, disp]], 1000).T
    plt.figure(figsize=(10, 10))
    plt.scatter(x1, x2, alpha=0.2)
    plt.scatter(x3, x4, alpha=0.2)
    plt.show()

    X1 = pd.DataFrame([x1, x2]).T
    X1["target"] = 0
    X2 = pd.DataFrame([x3, x4]).T
    X2["target"] = 1
    X = pd.concat([X1, X2])
    X.columns = ["var1", "var2", "target"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.drop(columns="target"),
        X["target"],
        stratify=X["target"],
        test_size=0.5,
        random_state=0,
    )

    detector = SelectiveAbstentionExplanations(
        model=XGBClassifier(n_estimators=20), gmodel=LogisticRegression(), cov=0.9
    )
    detector.fit(X_tr, y_tr)

    print(
        roc_auc_score(
            detector.create_error(X_te, y_te), detector.gpredict_proba(X_te)[:, 1]
        )
    )

# %%
