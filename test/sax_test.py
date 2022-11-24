from tools.xaiUtils import SelectiveAbstentionExplanations
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np


X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
X_ood, y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)


def test_return_shapDF():
    """
    If X is dataframe, return shap values as dataframe.
    """
    XX = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])

    esd = SelectiveAbstentionExplanations(
        model=LogisticRegression(), gmodel=LogisticRegression()
    )
    esd.fit_model(XX, y)
    ex = esd.get_explanations(XX)
    assert all([a == b for a, b in zip(ex.columns, XX.columns)])


def test_supported_models():
    """
    Check that models are supported.
    """
    for model in [XGBClassifier(), LogisticRegression()]:
        for gmodel in [XGBClassifier(), LogisticRegression()]:
            assert (
                type(SelectiveAbstentionExplanations(model=model, gmodel=gmodel))
                is SelectiveAbstentionExplanations
            )


def test_not_supported_models():
    """
    Check that models are not supported.
    """

    from sklearn.neural_network import MLPClassifier
    import pytest

    with pytest.raises(ValueError):
        SelectiveAbstentionExplanations(
            model=MLPClassifier(), gmodel=LogisticRegression()
        )
    with pytest.raises(ValueError):
        SelectiveAbstentionExplanations(
            model=LinearRegression(), gmodel=MLPClassifier()
        )


def test_doc_examples():
    """
    TODO : add a test for the doc examples.
    """


def test_no_nan():
    """
    Check that no NaNs are present in the shap values.
    """
    esd = SelectiveAbstentionExplanations(
        model=LogisticRegression(), gmodel=LogisticRegression()
    )
    esd.fit_model(X, y)
    ex = esd.get_explanations(X)
    assert not np.any(np.isnan(ex))


def test_get_coefs_linear():
    """
    Check that the coefficients are returned correctly for the linear regression.
    """
    esd = SelectiveAbstentionExplanations(
        model=LogisticRegression(), gmodel=LogisticRegression()
    )
    esd.fit(X, y)
    coefs = esd.get_linear_coefs()
    # Assert shape
    assert coefs.shape[1] == X.shape[1]
    # Assert that there is non NaNs
    assert not np.any(np.isnan(coefs))
    # Check when we call the full methods
    coefs = esd.get_coefs()
    # Assert shape
    assert coefs.shape[1] == X.shape[1]
    # Assert that there is non NaNs
    assert not np.any(np.isnan(coefs))


def test_get_coefs_pipeline():
    """
    Check that the coefficients are returned correctly for the linear regression pipeline.
    TODO : add a test for the case of a pipeline for F.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    esd = SelectiveAbstentionExplanations(
        model=LogisticRegression(),
        gmodel=Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())]),
    )
    esd.fit(X, y)
    coefs = esd.get_coefs()
    # Assert shape
    assert coefs.shape[1] == X.shape[1]
    # Assert that there is non NaNs
    assert not np.any(np.isnan(coefs))


def test_get_model_types():
    """
    Check that the model types are returned correctly.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    esd = SelectiveAbstentionExplanations(
        model=LogisticRegression(), gmodel=LogisticRegression()
    )
    assert esd.get_gmodel_type(), esd.get_model_type() == ("linear", "linear")
    # Case of pipeline
    esd = SelectiveAbstentionExplanations(
        model=Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())]),
        gmodel=Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())]),
    )
    assert esd.get_gmodel_type(), esd.get_model_type() == ("linear", "linear")


def test_predict_is_bool():
    """
    Check that the prediction is a boolean.
    """
    esd = SelectiveAbstentionExplanations(
        model=LogisticRegression(), gmodel=LogisticRegression()
    )
    esd.fit(X, y)
    assert esd.gpredict(X).dtype == bool
