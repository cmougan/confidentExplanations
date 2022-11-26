# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from tools.xaiUtils import SelectiveAbstentionExplanations
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
import numpy as np
from tools.PlugIn import PlugInRule

# %%
cov = 0.9
plot = False

res_sax = []
res_sax_actual_cov = []
res_sax_base = []
res_sax_base_actual_cov = []
res_plugin_actual_cov = []
res_plugIn = []
values = np.linspace(0.1, 2, 10)
for disp in values:
    # Create synthetic iid data that depends on a dispersion parameter
    X, y = make_blobs(n_samples=2000, centers=2, n_features=2, cluster_std=disp)
    df = pd.DataFrame(X, columns=["Var%d" % (i + 1) for i in range(X.shape[1])])
    df["label"] = y

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.5,
    )

    # Fit our detector
    detector = SelectiveAbstentionExplanations(
        model=LogisticRegression(), gmodel=XGBClassifier(), cov=cov
    )
    detector.fit(X_tr, y_tr)
    # Detector baseline on prediction space instead of explanation space
    det_pred = SelectiveAbstentionExplanations(
        model=LogisticRegression(),
        gmodel=XGBClassifier(),
        cov=cov,
        use_explanation_space=False,
    )
    det_pred.fit(X_tr, y_tr)
    # Fit plug in
    plugIn = PlugInRule(model=LogisticRegression())
    plugIn.fit(X_tr, y_tr)

    # Evaluation
    # TODO : name all predictions equally so we can reuse the same code
    # SAX
    ## Accuracy over accepted instances
    selected = detector.gpredict(X_te)
    preds = detector.fpredict(X_te).astype(bool)
    res_sax.append(
        accuracy_score(y_te[selected], preds[selected]),
    )
    res_sax_actual_cov.append(sum(selected) / len(selected))
    # Detector baseline
    selected_base = det_pred.gpredict(X_te).astype(bool)
    preds_base = det_pred.fpredict(X_te).astype(bool)
    res_sax_base.append(
        accuracy_score(y_te[selected_base], preds[selected_base]),
    )
    res_sax_base_actual_cov.append(sum(selected_base) / len(selected_base))
    # Plug in
    selected_plug = plugIn.predict(X_te).astype(bool)
    preds_plug = plugIn.model.predict(X_te).astype(bool)
    res_plugIn.append(
        accuracy_score(y_te[selected_plug], preds[selected_plug]),
    )
    res_plugin_actual_cov.append(sum(selected_plug) / len(selected_plug))

    # Plots
    if plot:
        plt.figure(figsize=(10, 10))
        X1 = df[df["label"] == 1]
        X0 = df[df["label"] == 0]
        plt.scatter(X1["Var1"], X1["Var2"], alpha=0.1, label="Class0")
        plt.scatter(X0["Var1"], X0["Var2"], alpha=0.1, label="Class1")
        plt.scatter(
            X_te[~selected_plug][:, 0],
            X_te[~selected_plug][:, 1],
            alpha=0.5,
            label="PlugIn",
        )
        plt.scatter(
            X_te[~selected][:, 0],
            X_te[~selected][:, 1],
            alpha=0.5,
            marker="+",
            label="SAX",
        )
        plt.scatter(
            X_te[~selected_base][:, 0],
            X_te[~selected_base][:, 1],
            alpha=0.5,
            marker=".",
            label="SAX Base",
        )
        plt.legend()
        plt.show()
# %%
plt.figure(figsize=(10, 10))
plt.title("Accuracy over accepted instances with coverage = {}".format(cov))
plt.plot(values, res_plugIn, label="PlugIn")
plt.plot(values, res_sax, label="SAX")
plt.plot(values, res_sax_base, label="SAX Base")
plt.legend()
plt.xlabel("Dispersion")
plt.ylabel("Accuracy")
plt.show()
# %%
plt.plot()
plt.title("Actual coverage = {}".format(cov))
plt.plot(values, res_plugin_actual_cov, label="PlugIn")
plt.plot(values, res_sax_actual_cov, label="SAX")
plt.plot(values, res_sax_base_actual_cov, label="SAX Base")
plt.legend()
plt.xlabel("Dispersion")
plt.ylabel("Accuracy")
plt.show()

# %%
