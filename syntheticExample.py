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
    X, y = make_blobs(
        n_samples=2000, centers=2, n_features=2, random_state=0, cluster_std=disp
    )
    df = pd.DataFrame(X, columns=["var1", "var2"])
    df["label"] = y

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        stratify=y,
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
    selected_plug = plugIn.predict(X_te).astype(bool)
    preds_plug = plugIn.model.predict(X_te).astype(bool)
    res_plugIn.append(
        accuracy_score(y_te[selected_plug], preds[selected_plug]),
    )
    res_plugin_actual_cov.append(sum(selected_plug) / len(selected_plug))
    if accuracy_score(y_te, preds) > accuracy_score(
        y_te[selected_plug], preds[selected_plug]
    ):
        print("PlugIn Accuracy over accepted instances is worse than overall accuracy")

    # Plots
    plt.figure()
    X1 = df[df["label"] == 1]
    X0 = df[df["label"] == 0]
    plt.scatter(X1["var1"], X1["var2"], alpha=0.1, label="Class0")
    plt.scatter(X0["var1"], X0["var2"], alpha=0.1, label="Class1")
    plt.scatter(
        X_te[~selected_plug][:, 0],
        X_te[~selected_plug][:, 1],
        alpha=0.5,
        label="PlugIn",
    )
    plt.scatter(
        X_te[~selected][:, 0], X_te[~selected][:, 1], alpha=0.5, marker="+", label="SAX"
    )
    plt.legend()
    plt.show()


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
