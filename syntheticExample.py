# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from tools.xaiUtils import SelectiveAbstentionExplanations
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from tools.xaiUtils import PlugInRule

# %%
cov = 0.9
plot = False
res_plugIn = []
res_sax = []
res_sax_actual_cov = []
res_plugin_actual_cov = []
values = np.linspace(0.1, 0.9, 9)
for disp in values:
    # Create synthetic iid data that depends on a dispersion parameter
    x1, x2 = np.random.multivariate_normal([0, 0], [[disp, 0], [0, disp]], 1000).T
    x3, x4 = np.random.multivariate_normal([1, 1], [[disp, 0], [0, disp]], 1000).T
    if plot:
        plt.figure(figsize=(10, 10))
        plt.scatter(x1, x2, alpha=0.2)
        plt.scatter(x3, x4, alpha=0.2)
        plt.show()
    # Convert to dataframe, add labels, and split into train and test
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
    # Fit our detector
    detector = SelectiveAbstentionExplanations(
        model=LogisticRegression(), gmodel=XGBClassifier(), cov=cov
    )
    detector.fit(X_tr, y_tr)
    # Fit plug in
    plugIn = PlugInRule(model=LogisticRegression())
    plugIn.fit(X_tr, y_tr)

    # Evaluation
    # SAX
    # TODO : name all predictions equally so we can reuse the same code
    ## Accuracy over accepted instances
    selected = detector.gpredict(X_te).astype(bool)
    preds = detector.fpredict(X_te).astype(bool)
    res_sax.append(
        accuracy_score(y_te[selected], preds[selected]),
    )
    res_sax_actual_cov.append(sum(selected) / len(selected))
    if accuracy_score(y_te, preds) > accuracy_score(y_te[selected], preds[selected]):
        print("SAX Accuracy over accepted instances is worse than overall accuracy")
    # Plug in
    selected = plugIn.predict(X_te).astype(bool)
    preds = plugIn.model.predict(X_te).astype(bool)
    res_plugIn.append(
        accuracy_score(y_te[selected], preds[selected]),
    )
    res_plugin_actual_cov.append(sum(selected) / len(selected))
    if accuracy_score(y_te, preds) > accuracy_score(y_te[selected], preds[selected]):
        print("PlugIn Accuracy over accepted instances is worse than overall accuracy")


# %%
plt.plot()
plt.title("Accuracy over accepted instances with coverage = {}".format(cov))
plt.plot(values, res_plugIn, label="PlugIn")
plt.plot(values, res_sax, label="SAX")
plt.legend()
plt.xlabel("Dispersion")
plt.ylabel("Accuracy")
plt.show()
# %%
plt.plot()
plt.title("Actual coverage = {}".format(cov))
plt.plot(values, res_plugin_actual_cov, label="PlugIn")
plt.plot(values, res_sax_actual_cov, label="SAX")
plt.legend()
plt.xlabel("Dispersion")
plt.ylabel("Accuracy")
plt.show()
# %%
