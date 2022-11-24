from tools.xaiUtils import PlugInRule
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import numpy as np

X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
X_ood, y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)


def test_doc_examples():
    """
    Test it works
    """
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_blobs
    from xgboost import XGBRegressor
    from tools.xaiUtils import PlugInRule
    from sklearn.metrics import accuracy_score

    X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    # X_ood,y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

    clf = PlugInRule(model=XGBRegressor())
    clf.fit(X_tr, y_tr)
    # scores = clf.predict_proba(X_te)[:,1]
    preds = clf.predict(X_te)
    bands = clf.qband(X_te)
    for level in range(len(clf.quantiles)+1):
        selected = bands >= level
        coverage = len(y[selected])/len(y)
        acc = accuracy_score(y_te[selected], preds[selected])
        print("target coverage is: {}".format(1-clf.quantiles[level]))
        print("coverage is: {}".format(coverage))
        print("selective accuracy is: {}".format(acc))

def test_plugin_fitted():
    """
    Check that no NaNs are present in the shap values.
    """
    from sklearn.utils.validation import check_is_fitted
    X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    # X_ood,y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

    clf = PlugInRule(model=XGBRegressor())
    clf.fit(X_tr, y_tr)
    assert check_is_fitted(clf.clf_base) is None

def test_thetas_estimated():
    """
    Check that thetas are estimated after fit
    """
    X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    # X_ood,y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

    clf = PlugInRule(model=XGBRegressor())
    clf.fit(X_tr, y_tr)
    assert clf.thetas is not None
