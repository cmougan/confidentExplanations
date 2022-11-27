from tools.xaiUtils import SelectiveAbstentionExplanations
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import pytest
from sklearn.tree import DecisionTreeClassifier

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


def test_check_is_predicting_something():
    """
    Check that all predictions are not zero
    """
    # TODO


def test_check_use_explanations():
    """
    Check that the use_explanations method works.
    """
    esd0 = SelectiveAbstentionExplanations(
        model=LogisticRegression(),
        gmodel=LogisticRegression(),
        use_explanation_space=False,
    )
    esd0.fit(X, y)
    assert esd0.get_explanations(X).shape[1] == 1
    esd1 = SelectiveAbstentionExplanations(
        model=LogisticRegression(),
        gmodel=LogisticRegression(),
    )
    esd1.fit(X, y)
    assert esd0.get_explanations(X).shape[1] != esd1.get_explanations(X).shape[1]


def test_check_use_explanations_learns():
    """
    Check that the use_explanations method learns.
    # TODO
    """


def test_decision_tree():
    # TODO the test doest not pass if fmodel is DecisionTreeClassifier
    fmodel = LogisticRegression()
    gmodel = DecisionTreeClassifier()
    # Fit our detector
    detector = SelectiveAbstentionExplanations(model=fmodel, gmodel=gmodel)
    detector.fit(X_tr, y_tr)
    assert detector.gpredict(X_te).shape[0] == X_te.shape[0]


def test_check_explanation_space_dim():
    detector = SelectiveAbstentionExplanations(
        model=LogisticRegression(),
        gmodel=LogisticRegression(),
    )
    for i in [2, 5, 10]:
        X, y = make_blobs(n_samples=100, centers=2, n_features=i, random_state=0)
        detector.fit(X, y)
        assert detector.get_explanations(X).shape == (100, i)
