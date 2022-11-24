from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
import shap
import copy


class ShapEstimator(BaseEstimator, ClassifierMixin):
    """
    A ShapValues estimator based on tree explainer.
    Returns the explanations of the data provided self.predict(X)

    Example:

    import xgboost
    from sklearn.model_selection import cross_val_predict
    X, y = shap.datasets.boston()
    se = ShapEstimator(model=xgboost.XGBRegressor())
    shap_pred = cross_val_predict(se, X, y, cv=3)
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        self.model.fit(self.X_, self.y_)
        return self

    def predict(self, X, dataframe: bool = False):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        check_array(X)

        explainer = shap.Explainer(self.model)
        shap_values = explainer(X).values
        if dataframe:
            shap_values = pd.DataFrame(shap_values, columns=X.columns)
            shap_values = shap_values.add_suffix("_shap")

        return shap_values


class SelectiveAbstentionExplanations(BaseEstimator, ClassifierMixin):
    """
    Given a model

    Example
    -------
    >>> import pandas as pd
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import make_blobs
    >>> from tools.xaiUtils import ExplanationShiftDetector
    >>> from xgboost import XGBRegressor
    >>> from sklearn.linear_model import LogisticRegression

    >>> X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> X_ood,y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

    >>> detector = ExplanationShiftDetector(model=XGBRegressor(),gmodel=LogisticRegression())
    >>> detector.fit(X_tr, y_tr, X_ood)
    >>> detector.get_auc_val()
    # 0.76
    >>> detector.fit(X_tr, y_tr, X_te)
    >>> detector.get_auc_val()
    # 0.5
    """

    def __init__(self, model, gmodel, cov: float = 0.95, cv=None):
        self.model = model
        self.gmodel = gmodel
        self.cov = cov
        self.cv = cv
        # Supported F Models
        self.supported_tree_models = ["XGBClassifier", "LGBMClassifier"]
        self.supported_linear_models = [
            "LogisticRegression",
        ]
        self.supported_models = (
            self.supported_tree_models + self.supported_linear_models
        )
        # Supported detectors
        self.supported_linear_detectors = [
            "LogisticRegression",
        ]
        self.supported_tree_detectors = ["XGBClassifier", "LGBMClassifier"]
        self.supported_detectors = (
            self.supported_linear_detectors + self.supported_tree_detectors
        )

        # Check if models are supported
        if self.get_model_type() not in self.supported_models:
            raise ValueError(
                "Model not supported. Supported models are: {} got {}".format(
                    self.supported_models, self.model.__class__.__name__
                )
            )
        if self.get_gmodel_type() not in self.supported_detectors:
            raise ValueError(
                "gmodel not supported. Supported models are: {} got {}".format(
                    self.supported_detectors, self.gmodel.__class__.__name__
                )
            )

    def get_gmodel_type(self):
        if self.gmodel.__class__.__name__ == "Pipeline":
            return self.gmodel.steps[-1][1].__class__.__name__
        else:
            return self.gmodel.__class__.__name__

    def get_model_type(self):
        if self.model.__class__.__name__ == "Pipeline":
            return self.model.steps[-1][1].__class__.__name__
        else:
            return self.model.__class__.__name__

    def fit(self, X_source, y_source):

        # Check that X and y have correct shape
        check_X_y(X_source, y_source)
        self.X_tr, self.X_val, self.y_tr, self.y_val = train_test_split(
            X_source, y_source, random_state=0, test_size=0.5
        )

        # Fit model F
        self.fit_model(self.X_tr, self.y_tr)

        # Get explanations
        self.S_val = self.get_explanations(self.X_val)

        # Create error
        self.e_val = self.create_error_first(self.X_val, self.y_val)

        # Fit SAX
        self.gmodel.fit(self.S_val, self.e_val)

        return self

    def create_error_first(self, X, y):
        preds = self.fpredict_proba(X)[:, 1]
        self.theta = np.quantile(np.abs(preds - y), q=self.cov)
        return np.where(np.abs(preds - y) < self.theta, 1, 0)

    def create_error(self, X, y):
        preds = self.fpredict_proba(X)[:, 1]
        return np.where(np.abs(preds - y) < self.theta, 1, 0)

    def fpredict(self, X):
        return self.model.predict(X)

    def fpredict_proba(self, X):
        return self.model.predict_proba(X)

    def gpredict(self, X):
        return self.gmodel.predict(self.get_explanations(X)).astype(bool)

    def gpredict_proba(self, X):
        return self.gmodel.predict_proba(self.get_explanations(X))

    def explanation_predict(self, X):
        return self.gmodel.predict(X)

    def explanation_predict_proba(self, X):
        return self.gmodel.predict_proba(X)

    def fit_model(self, X, y):
        self.model.fit(X, y)

    def get_explanations(self, X):
        # Determine the type of SHAP explainer to use
        if self.get_model_type() in self.supported_tree_models:
            self.explainer = shap.Explainer(self.model)
        elif self.get_model_type() in self.supported_linear_models:
            self.explainer = shap.LinearExplainer(
                self.model, X, feature_dependence="correlation_dependent"
            )
        else:
            raise ValueError(
                "Model not supported. Supported models are: {}, got {}".format(
                    self.supported_models, self.model.__class__.__name__
                )
            )

        shap_values = self.explainer(X)
        # Name columns
        if isinstance(X, pd.DataFrame):
            columns_name = X.columns
        else:
            columns_name = ["Shap%d" % (i + 1) for i in range(X.shape[1])]

        exp = pd.DataFrame(
            data=shap_values.values,
            columns=columns_name,
        )
        return exp

    def get_auc_val(self):
        """
        Returns the AUC of the explanation shift detector on the validation set of the explanation space
        Example
        -------
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_blobs
        from tools.xaiUtils import ExplanationShiftDetector
        from xgboost import XGBRegressor
        from sklearn.linear_model import LogisticRegression

        # Create data
        X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
        X_ood,y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

        detector = ExplanationShiftDetector(model=XGBRegressor(),gmodel=LogisticRegression())
        detector.fit(X_tr, y_tr, X_ood)
        detector.get_auc_val()
        # 0.76

        """
        return roc_auc_score(
            self.y_shap_te, self.explanation_predict_proba(self.X_shap_te)[:, 1]
        )

    def get_coefs(self):
        if self.gmodel.__class__.__name__ == "Pipeline":
            if (
                self.gmodel.steps[-1][1].__class__.__name__
                in self.supported_linear_models
            ):
                return self.gmodel.steps[-1][1].coef_
            else:
                raise ValueError(
                    "Pipeline model not supported. Supported models are: {}, got {}".format(
                        self.supported_linear_models,
                        self.gmodel.steps[-1][1].__class__.__name__,
                    )
                )
        else:
            return self.get_linear_coefs()

    def get_linear_coefs(self):
        if self.gmodel.__class__.__name__ in self.supported_linear_models:
            return self.gmodel.coef_
        else:
            raise ValueError(
                "Detector model not supported. Supported models ar linear: {}, got {}".format(
                    self.supported_linear_detector, self.model.__class__.__name__
                )
            )


# pluginrule
class PlugInRule(BaseEstimator, ClassifierMixin):
    """
    Given a model

    Example
    -------
    >>> import pandas as pd
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import make_blobs
    >>> from xgboost import XGBRegressor
    >>> from sklearn.linear_model import LogisticRegression
    >>> from tools.xaiUtils import PlugInRule

    >>> X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> X_ood,y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

    >>> clf = PlugInRule(model=XGBRegressor())
    >>> clf.fit(X_tr, y_tr)
    >>> clf.predict(X_te)


    """

    def __init__(
        self,
        model,
        seed: int = 42,
    ):
        self.seed = seed
        self.model = model

    def fit(self, X, y):
        # Dimensionality check
        check_X_y(X, y)

        # Split the data and save hold out sets
        X_train, self.X_hold, y_train, self.y_hold = train_test_split(
            X, y, stratify=y, random_state=self.seed, test_size=0.1
        )
        # Fit the model
        self.model.fit(X_train, y_train)

    def compute_theta(self, q: int = 0.9):
        # Compute the scores
        scores = np.max(self.model.predict_proba(self.X_hold), axis=1)

        # TODO : check if the quantile is a list and save a list

        # Compute the theta
        self.theta = np.quantile(scores, q)

    def predict(self, X, cov: int = 0.9):

        # TODO check if the below function has been called
        self.compute_theta(1 - cov)

        probas = self.model.predict_proba(X)
        confs = np.max(probas, axis=1)
        return confs > self.theta
